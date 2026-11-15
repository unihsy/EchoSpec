#!/usr/bin/env bash
# ==============================================================================
# EchoSpec — Step 3: Unified Evaluation (Transformation Suites, QA & Cross-Retriever)
# ==============================================================================
# Evaluates a trained EchoSpec LoRA adapter standalone and composed with
# orthogonal retrieval filters (TrustRAG, Semantic Dedup) across:
#   - Test Set A (Qwen3-32B, 10 worlds per item) & Test Set B (Phi-4, 10 worlds)
#   - 7 Open-Domain QA Benchmarks (NQ, TriviaQA, PopQA, HotpotQA, 2Wiki, MuSiQue, Bamboogle)
#   - (Optional) Zero-shot cross-retriever transfer to BGE-base-en-v1.5 and Okapi BM25
#
# Usage:
#   ADAPTER_DIR=./outputs/echospec_qwen25_7b_s1/final_adapter bash scripts/evaluate_echospec.sh
#   RETRIEVER=bge  ADAPTER_DIR=./outputs/echospec_qwen25_7b_s1/final_adapter bash scripts/evaluate_echospec.sh
#   RETRIEVER=bm25 ADAPTER_DIR=./outputs/echospec_qwen25_7b_s1/final_adapter bash scripts/evaluate_echospec.sh
# ==============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
ADAPTER_DIR="${ADAPTER_DIR:-./outputs/echospec_qwen25_7b_s1/final_adapter}"
RETRIEVER="${RETRIEVER:-e5}"   # e5 | bge | bm25
EVAL_DIR="${EVAL_DIR:-./outputs/eval_${RETRIEVER}}"
RETR_GPU="${RETR_GPU:-0}"
RETR_PORT="${RETR_PORT:-8290}"
SERVE_GROUPS="${SERVE_GROUPS:-1;2;3;4}"

mkdir -p "${EVAL_DIR}/logs"
ARROW="${DATA_DIR}/wiki18/wiki18_contents.arrow"

cleanup() {
  [[ -n "${VLLM_PID:-}" ]] && kill -TERM "${VLLM_PID}" 2>/dev/null || true
  [[ -n "${RETR_PID:-}" ]] && kill -TERM "${RETR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

case "${RETRIEVER}" in
  e5)
    CORPUS_NAME="wiki18"
    RETR_ARGS=(--corpus "wiki18=${ARROW}" --emb "wiki18=${DATA_DIR}/wiki18/wiki18_e5_fp16.npy")
    ;;
  bge)
    CORPUS_NAME="wiki18_bge"
    RETR_ARGS=(--corpus "wiki18_bge=${ARROW}" --emb "wiki18_bge=${DATA_DIR}/wiki18/wiki18_bge_fp16.npy" --encoder_map "wiki18_bge=BAAI/bge-base-en-v1.5")
    ;;
  bm25)
    CORPUS_NAME="wiki18_bm25"
    RETR_ARGS=(--corpus "wiki18_bm25=${ARROW}" --bm25 "wiki18_bm25=${DATA_DIR}/wiki18/bm25")
    ;;
  *)
    echo "Unsupported RETRIEVER='${RETRIEVER}' (expected: e5 | bge | bm25)" >&2
    exit 1
    ;;
esac

echo "[1/3] Starting ${RETRIEVER} retrieval server on GPU ${RETR_GPU} (:${RETR_PORT})..."
CUDA_VISIBLE_DEVICES="${RETR_GPU}" HIP_VISIBLE_DEVICES="${RETR_GPU}" \
  python -m echospec.retrieval \
    "${RETR_ARGS[@]}" \
    --device "cuda:0" \
    --port "${RETR_PORT}" \
    > "${EVAL_DIR}/logs/retriever.log" 2>&1 &
RETR_PID=$!

for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:${RETR_PORT}/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:${RETR_PORT}/health" >/dev/null

echo "[2/3] Starting vLLM inference servers (${BASE_MODEL}) on GPUs ${SERVE_GROUPS}..."
EP_EVAL="${EVAL_DIR}/endpoints_eval.json"
rm -f "${EP_EVAL}"
python -m echospec.serve \
  --groups "${SERVE_GROUPS}" \
  --model "${BASE_MODEL}" \
  --log_dir "${EVAL_DIR}/logs/vllm" \
  --out_json "${EP_EVAL}" \
  --first_port 8510 \
  --lora &
VLLM_PID=$!
until [[ -f "${EP_EVAL}" ]]; do sleep 3; kill -0 "${VLLM_PID}"; done

METHODS_JSON="${EVAL_DIR}/methods.json"
cat > "${METHODS_JSON}" <<EOF
[
  {
    "name": "echospec",
    "base_model": "${BASE_MODEL}",
    "lora_path": "${ADAPTER_DIR}",
    "prompt_style": "neutral",
    "corpus": "${CORPUS_NAME}",
    "exclude_seen": true
  },
  {
    "name": "echospec_no_exclude_seen",
    "base_model": "${BASE_MODEL}",
    "lora_path": "${ADAPTER_DIR}",
    "prompt_style": "neutral",
    "corpus": "${CORPUS_NAME}",
    "exclude_seen": false
  },
  {
    "name": "echospec_trustrag",
    "base_model": "${BASE_MODEL}",
    "lora_path": "${ADAPTER_DIR}",
    "prompt_style": "neutral",
    "corpus": "${CORPUS_NAME}",
    "exclude_seen": true,
    "retrieval_filter": "trustrag"
  },
  {
    "name": "echospec_semantic_dedup",
    "base_model": "${BASE_MODEL}",
    "lora_path": "${ADAPTER_DIR}",
    "prompt_style": "neutral",
    "corpus": "${CORPUS_NAME}",
    "exclude_seen": true,
    "retrieval_filter": "novelty"
  }
]
EOF

BENCHES=("${DATA_DIR}/spec/spec_test_A.jsonl")
[[ -f "${DATA_DIR}/spec/spec_test_B.jsonl" ]] && BENCHES+=("${DATA_DIR}/spec/spec_test_B.jsonl")
for F in "${DATA_DIR}"/qa/l4_*.jsonl; do
  [[ -f "${F}" ]] && BENCHES+=("${F}")
done

echo "[3/3] Evaluating across ${#BENCHES[@]} suites -> ${EVAL_DIR}/summary.json..."
python -m echospec.evaluate \
  --endpoints_json "${EP_EVAL}" \
  --retriever_url "http://127.0.0.1:${RETR_PORT}" \
  --methods_json "${METHODS_JSON}" \
  --benches "${BENCHES[@]}" \
  --out_dir "${EVAL_DIR}"

echo "[done] Aggregated metrics written to ${EVAL_DIR}/summary.json"
