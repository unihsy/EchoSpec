#!/usr/bin/env bash
# ==============================================================================
# EchoSpec — Step 2: Two-Stage Training (Verification Warm-Start + lambda-GRPO)
# ==============================================================================
# Trains an EchoSpec search agent across any supported instruction backbone:
#   - Qwen/Qwen2.5-7B-Instruct          (default)
#   - meta-llama/Llama-3.1-8B-Instruct
#   - Qwen/Qwen2.5-3B-Instruct
#
# Default 8-GPU layout:
#   GPU 0     : GPU-resident e5-base-v2 retrieval server
#   GPUs 1..4 : 4 x vLLM rollout servers with runtime LoRA hot-loading
#   GPUs 5..7 : Distributed PyTorch LoRA SFT + async GRPO trainer (torchrun)
#
# 4-GPU single-node override example:
#   RETR_GPU=0 ROLLOUT_GROUPS="1;2" TRAIN_GPUS="3" bash scripts/train_echospec.sh
# ==============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
OUT_DIR="${OUT_DIR:-./outputs/echospec_qwen25_7b_s1}"
RUN_TAG="${RUN_TAG:-$(basename "${OUT_DIR}")}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen2.5-32B-Instruct}"
SFT_JSONL="${SFT_JSONL:-${DATA_DIR}/spec/sft_trajectories.jsonl}"
SEED="${SEED:-1}"

# Hardware layout
RETR_GPU="${RETR_GPU:-0}"
RETR_PORT="${RETR_PORT:-8290}"
TEACHER_GROUPS="${TEACHER_GROUPS:-1,2;3,4}"  # two TP=2 servers for 32B teacher synthesis
ROLLOUT_GROUPS="${ROLLOUT_GROUPS:-1;2;3;4}"  # four TP=1 rollout servers for RL
TRAIN_GPUS="${TRAIN_GPUS:-5,6,7}"
MASTER_PORT="${MASTER_PORT:-29501}"

# EchoSpec hyperparameters (shared across Qwen2.5-3B, Qwen2.5-7B, and Llama-3.1-8B)
LAM="${LAM:-0.1}"
GROUND_ALPHA="${GROUND_ALPHA:-0.25}"
ACQ_BETA="${ACQ_BETA:-0.3}"
EXCLUDE_SEEN="${EXCLUDE_SEEN:-1}"
SFT_STEPS="${SFT_STEPS:-60}"
RL_STEPS="${RL_STEPS:-200}"

mkdir -p "${OUT_DIR}/logs"
ARROW="${DATA_DIR}/wiki18/wiki18_contents.arrow"
E5_NPY="${DATA_DIR}/wiki18/wiki18_e5_fp16.npy"

cleanup() {
  [[ -n "${VLLM_PID:-}" ]] && kill -TERM "${VLLM_PID}" 2>/dev/null || true
  [[ -n "${RETR_PID:-}" ]] && kill -TERM "${RETR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[1/3] Launching e5-base-v2 retrieval server on GPU ${RETR_GPU} (:${RETR_PORT})..."
CUDA_VISIBLE_DEVICES="${RETR_GPU}" HIP_VISIBLE_DEVICES="${RETR_GPU}" \
  python -m echospec.retrieval \
    --corpus "wiki18=${ARROW}" \
    --emb "wiki18=${E5_NPY}" \
    --device "cuda:0" \
    --port "${RETR_PORT}" \
    > "${OUT_DIR}/logs/retriever.log" 2>&1 &
RETR_PID=$!

for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:${RETR_PORT}/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:${RETR_PORT}/health" >/dev/null

if [[ ! -f "${SFT_JSONL}" ]]; then
  echo "[2/3] Synthesizing Stage-1 chain-guided verification trajectories with ${TEACHER_MODEL}..."
  EP_SYNTH="${OUT_DIR}/endpoints_synth.json"
  rm -f "${EP_SYNTH}"
  python -m echospec.serve \
    --groups "${TEACHER_GROUPS}" \
    --model "${TEACHER_MODEL}" \
    --log_dir "${OUT_DIR}/logs/vllm_synth" \
    --out_json "${EP_SYNTH}" \
    --first_port 8310 &
  VLLM_PID=$!
  until [[ -f "${EP_SYNTH}" ]]; do sleep 3; kill -0 "${VLLM_PID}"; done

  python -m echospec.synth \
    --train_jsonl "${DATA_DIR}/spec/spec_train.jsonl" \
    --endpoints_json "${EP_SYNTH}" \
    --out "${SFT_JSONL}" \
    --served_model "${TEACHER_MODEL}" \
    --tokenizer "${BASE_MODEL}" \
    --retriever_url "http://127.0.0.1:${RETR_PORT}" \
    --per_item 3 \
    --samples 4 \
    --temperature 0.8 \
    --exclude_seen "${EXCLUDE_SEEN}" \
    --family_probs "inject:0.6,delete:0.15,consistent:0.1,dup:0.15" \
    --m_probs "1:0.1,3:0.2,5:0.3,8:0.4" \
    --seed 0

  kill -TERM "${VLLM_PID}" && wait "${VLLM_PID}" || true
  unset VLLM_PID
else
  echo "[2/3] Reusing existing Stage-1 verification trajectories at ${SFT_JSONL}"
fi

echo "[3/3] Starting vLLM rollout servers (${BASE_MODEL}) with runtime LoRA updating..."
EP_RL="${OUT_DIR}/endpoints_rl.json"
rm -f "${EP_RL}"
python -m echospec.serve \
  --groups "${ROLLOUT_GROUPS}" \
  --model "${BASE_MODEL}" \
  --log_dir "${OUT_DIR}/logs/vllm_rl" \
  --out_json "${EP_RL}" \
  --first_port 8410 \
  --lora &
VLLM_PID=$!
until [[ -f "${EP_RL}" ]]; do sleep 3; kill -0 "${VLLM_PID}"; done

IFS=',' read -r -a GPU_ARR <<< "${TRAIN_GPUS}"
NPROC="${#GPU_ARR[@]}"

echo "[3/3] Launching Stage-1 LoRA warm-start (${SFT_STEPS} steps) + Stage-2 lambda-GRPO (${RL_STEPS} steps) on GPUs ${TRAIN_GPUS}..."
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" HIP_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  torchrun --standalone --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
    -m echospec.train \
      --run_tag "${RUN_TAG}" \
      --base_model "${BASE_MODEL}" \
      --train_jsonl "${DATA_DIR}/spec/spec_train.jsonl" \
      --dev_jsonl "${DATA_DIR}/spec/spec_dev.jsonl" \
      --sft_jsonl "${SFT_JSONL}" \
      --sft_steps "${SFT_STEPS}" \
      --sft_batch 64 \
      --sft_lr 1e-4 \
      --families "identity,dup,inject,consistent,rewrite,delete" \
      --family_probs "dup:0.1,inject:0.4,consistent:0.15,rewrite:0.1,delete:0.25" \
      --m_probs "1:0.1,2:0.1,3:0.2,5:0.25,8:0.35" \
      --lam "${LAM}" \
      --ground_alpha "${GROUND_ALPHA}" \
      --acq_beta "${ACQ_BETA}" \
      --adv_norm "none" \
      --kl_beta 0.001 \
      --lr 1e-5 \
      --lora_rank 64 \
      --items_per_step 12 \
      --group_size 6 \
      --steps "${RL_STEPS}" \
      --exclude_seen "${EXCLUDE_SEEN}" \
      --endpoints_json "${EP_RL}" \
      --retriever_url "http://127.0.0.1:${RETR_PORT}" \
      --out_dir "${OUT_DIR}" \
      --seed "${SEED}"

echo "[done] Final LoRA adapter saved to ${OUT_DIR}/final_adapter"
