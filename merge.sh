#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

INPUT_ROOT="${ANANT_INPUT_ROOT:-/kaggle/input/notebooks/aaravmaloo/anant-coder}"
INPUT_OUTPUT="${INPUT_ROOT}/output_anant"
WORK_OUTPUT="${ANANT_OUTPUT_DIR:-/kaggle/working/output_anant}"
ARTIFACT_NAME="${ANANT_ARTIFACT_NAME:-anant-14b-coder}"
GGUF_OUT="${ANANT_GGUF_OUT:-/kaggle/working/${ARTIFACT_NAME}-F16.gguf}"
MERGED_DIR="${WORK_OUTPUT}/merged/${ARTIFACT_NAME}-F16"
LLAMA_CPP_DIR="/kaggle/working/llama.cpp"

export ANANT_WORK_DIR="/kaggle/working"
export ANANT_OUTPUT_DIR="${WORK_OUTPUT}"
export HF_HOME="${HF_HOME:-/kaggle/temp/hf_cache}"
export ANANT_MERGE_METHOD="${ANANT_MERGE_METHOD:-stream}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

echo "[setup] input=${INPUT_OUTPUT}"
test -d "${INPUT_OUTPUT}/adapters/${ARTIFACT_NAME}" || {
  echo "missing adapter: ${INPUT_OUTPUT}/adapters/${ARTIFACT_NAME}" >&2
  exit 1
}

echo "[clean] Kaggle input is read-only; cleaning working dirs only"
rm -rf "${WORK_OUTPUT}" "${LLAMA_CPP_DIR}" /kaggle/working/*.gguf
mkdir -p "${WORK_OUTPUT}/adapters" "${WORK_OUTPUT}/logs" /kaggle/temp

echo "[copy] adapter -> working"
cp -a "${INPUT_OUTPUT}/adapters/${ARTIFACT_NAME}" "${WORK_OUTPUT}/adapters/"
if [[ -d "${INPUT_OUTPUT}/logs" ]]; then
  cp -a "${INPUT_OUTPUT}/logs/." "${WORK_OUTPUT}/logs/" || true
fi

echo "[setup] deps"
pip install -q -U transformers peft accelerate sentencepiece huggingface_hub safetensors gguf protobuf
pip uninstall -y -q torchao || true

echo "[merge] streaming LoRA -> merged F16 HF"
python lora_merge.py 2>&1 | tee -a "${WORK_OUTPUT}/logs/merge.log"

echo "[gguf] clone llama.cpp"
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"

echo "[gguf] convert F16 -> ${GGUF_OUT}"
python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" "${MERGED_DIR}" --outfile "${GGUF_OUT}" --outtype f16

echo "[clean] remove merged HF shards"
rm -rf "${MERGED_DIR}" "${WORK_OUTPUT}/merged" "${LLAMA_CPP_DIR}"

echo "[done] GGUF:"
ls -lh "${GGUF_OUT}"
