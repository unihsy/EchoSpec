#!/usr/bin/env bash
# ==============================================================================
# EchoSpec — Step 1: Corpus, Retrieval Indices & Transformation Worlds
# ==============================================================================
# Reconstructs all benchmark and training artifacts from public sources:
#   1. 21M-passage Wikipedia dump (wiki-18) + e5-base-v2 fp16 embeddings
#   2. (Optional) BGE-base-en-v1.5 dense index & Okapi BM25 inverted index
#   3. HotpotQA + MuSiQue multi-hop base items & 7 open-domain QA test sets
#   4. Symmetric claim/rewrite generations (Generator A: Qwen3-32B, B: Phi-4)
#   5. Iterative retrieval-verified deletion masks + paired 10-world test suites
#
# Usage:
#   DATA_DIR=./data bash scripts/prepare_data.sh
#   BUILD_CROSS_RETRIEVERS=1 DATA_DIR=./data bash scripts/prepare_data.sh
# ==============================================================================
set -euo pipefail

DATA_DIR="${DATA_DIR:-./data}"
WORK_DIR="${WORK_DIR:-${DATA_DIR}/_scratch}"
RETR_GPU="${RETR_GPU:-0}"
RETR_PORT="${RETR_PORT:-8290}"
GEN_GROUPS_A="${GEN_GROUPS_A:-1,2;3,4}"   # e.g. two TP=2 replicas for 32B generator
GEN_GROUPS_B="${GEN_GROUPS_B:-1;2;3;4}"   # e.g. four TP=1 replicas for 14B Phi-4
GEN_MODEL_A="${GEN_MODEL_A:-Qwen/Qwen3-32B}"
GEN_MODEL_B="${GEN_MODEL_B:-microsoft/phi-4}"
BUILD_CROSS_RETRIEVERS="${BUILD_CROSS_RETRIEVERS:-0}"

mkdir -p "${DATA_DIR}/wiki18" "${DATA_DIR}/qa" "${DATA_DIR}/spec" "${WORK_DIR}/logs"

cleanup() {
  [[ -n "${VLLM_PID:-}" ]] && kill -TERM "${VLLM_PID}" 2>/dev/null || true
  [[ -n "${RETR_PID:-}" ]] && kill -TERM "${RETR_PID}" 2>/dev/null || true
}
trap cleanup EXIT

echo "[1/5] Building wiki-18 Arrow corpus and e5-base-v2 fp16 index..."
python -m echospec.wiki18 \
  --out "${DATA_DIR}/wiki18" \
  --work "${WORK_DIR}/wiki18"

ARROW="${DATA_DIR}/wiki18/wiki18_contents.arrow"
E5_NPY="${DATA_DIR}/wiki18/wiki18_e5_fp16.npy"

if [[ "${BUILD_CROSS_RETRIEVERS}" == "1" ]]; then
  echo "[1b/5] Building cross-retriever indices (BAAI/bge-base-en-v1.5 & Okapi BM25)..."
  python -m echospec.wiki18 --build_dense "${ARROW}" "${DATA_DIR}/wiki18/wiki18_bge_fp16.npy" "BAAI/bge-base-en-v1.5"
  python -m echospec.wiki18 --build_bm25 "${ARROW}" "${DATA_DIR}/wiki18/bm25" --work "${WORK_DIR}/bm25"
fi

echo "[2/5] Downloading HotpotQA + MuSiQue base splits and 7 open-domain QA benchmarks..."
python -m echospec.qa_data --out "${DATA_DIR}/spec" --base
python -m echospec.qa_data --out "${DATA_DIR}/qa" --l4 --l4_max 1000

echo "[3/5] Starting GPU-resident e5-base-v2 retrieval server on GPU ${RETR_GPU} (:${RETR_PORT})..."
CUDA_VISIBLE_DEVICES="${RETR_GPU}" HIP_VISIBLE_DEVICES="${RETR_GPU}" \
  python -m echospec.retrieval \
    --corpus "wiki18=${ARROW}" \
    --emb "wiki18=${E5_NPY}" \
    --device "cuda:0" \
    --port "${RETR_PORT}" \
    > "${WORK_DIR}/logs/retriever.log" 2>&1 &
RETR_PID=$!

for _ in $(seq 1 180); do
  curl -fsS "http://127.0.0.1:${RETR_PORT}/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://127.0.0.1:${RETR_PORT}/health" >/dev/null

echo "[3b/5] Verifying primary-evidence retrievability and iterative deletion masks..."
for SPLIT in train dev test; do
  python -m echospec.transforms verify \
    --base "${DATA_DIR}/spec/base_${SPLIT}.jsonl" \
    --out "${DATA_DIR}/spec/mask_${SPLIT}.jsonl" \
    --retriever_url "http://127.0.0.1:${RETR_PORT}"
done

echo "[4/5] Generating symmetric world passages with Generator A (${GEN_MODEL_A})..."
EP_A="${WORK_DIR}/endpoints_gen_a.json"
rm -f "${EP_A}"
python -m echospec.serve \
  --groups "${GEN_GROUPS_A}" \
  --model "${GEN_MODEL_A}" \
  --log_dir "${WORK_DIR}/logs/gen_a" \
  --out_json "${EP_A}" \
  --first_port 8310 &
VLLM_PID=$!
until [[ -f "${EP_A}" ]]; do sleep 3; kill -0 "${VLLM_PID}"; done

for SPLIT in train dev test; do
  python -m echospec.transforms generate \
    --base "${DATA_DIR}/spec/base_${SPLIT}.jsonl" \
    --out "${DATA_DIR}/spec/gen_A_${SPLIT}.jsonl" \
    --endpoints_json "${EP_A}"
done
kill -TERM "${VLLM_PID}" && wait "${VLLM_PID}" || true
unset VLLM_PID

if [[ -n "${GEN_MODEL_B}" ]]; then
  echo "[4b/5] Generating held-out cross-generator passages with Generator B (${GEN_MODEL_B})..."
  EP_B="${WORK_DIR}/endpoints_gen_b.json"
  rm -f "${EP_B}"
  python -m echospec.serve \
    --groups "${GEN_GROUPS_B}" \
    --model "${GEN_MODEL_B}" \
    --log_dir "${WORK_DIR}/logs/gen_b" \
    --out_json "${EP_B}" \
    --first_port 8320 &
  VLLM_PID=$!
  until [[ -f "${EP_B}" ]]; do sleep 3; kill -0 "${VLLM_PID}"; done

  python -m echospec.transforms generate \
    --base "${DATA_DIR}/spec/base_test.jsonl" \
    --out "${DATA_DIR}/spec/gen_B_test.jsonl" \
    --endpoints_json "${EP_B}"
  kill -TERM "${VLLM_PID}" && wait "${VLLM_PID}" || true
  unset VLLM_PID
fi

echo "[5/5] Assembling paired transformation worlds and running TF-IDF style audit..."
GEN_B_ARG=()
[[ -f "${DATA_DIR}/spec/gen_B_test.jsonl" ]] && GEN_B_ARG=(--gen_b "${GEN_MODEL_B}")
python -m echospec.transforms assemble \
  --root "${DATA_DIR}/spec" \
  --gen_a "${GEN_MODEL_A}" \
  "${GEN_B_ARG[@]}"

echo "[done] Data & transformation worlds ready under ${DATA_DIR}"
