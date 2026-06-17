#!/usr/bin/env bash
set -euo pipefail

# Anant-Code Training Orchestrator
# Optimized for Kaggle 2x T4 GPUs

cd "$(dirname "$0")"

# --- REQUIRED: SET YOUR HF TOKEN ---
export HUGGING_FACE_HUB_TOKEN="${HUGGING_FACE_HUB_TOKEN:-YOUR_TOKEN_HERE}"
# -----------------------------------

export ANANT_WORK_DIR="/kaggle/working"
export ANANT_OUTPUT_DIR="/kaggle/working/output_anant"
export HF_HOME="/kaggle/temp/hf_cache"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="0,1"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export ANANT_SEQ_LEN="${ANANT_SEQ_LEN:-1024}"
export ANANT_LORA_R="${ANANT_LORA_R:-8}"
export ANANT_LORA_ALPHA="${ANANT_LORA_ALPHA:-16}"
export ANANT_DEVICE_MAP="${ANANT_DEVICE_MAP:-auto}"
export ANANT_MAX_MEMORY_GPU="${ANANT_MAX_MEMORY_GPU:-10GiB}"
export ANANT_MAX_MEMORY_CPU="${ANANT_MAX_MEMORY_CPU:-24GiB}"
export ANANT_MERGE_METHOD="${ANANT_MERGE_METHOD:-stream}"
export ANANT_KEEP_MERGED="${ANANT_KEEP_MERGED:-0}"

mkdir -p "${ANANT_OUTPUT_DIR}/logs" "${ANANT_OUTPUT_DIR}/gguf" "${HF_HOME}" /kaggle/temp
: > "${ANANT_OUTPUT_DIR}/logs/train.log"

echo "[setup] Detecting environment..."
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Kaggle GPU T4 x2 recommended." >&2
fi

GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l || echo 0)"
echo "[setup] Detected ${GPU_COUNT} GPUs"

# 1. Install dependencies & Login
echo "[setup] Installing dependencies and authenticating..."
pip install -q -U transformers peft datasets bitsandbytes accelerate sentencepiece huggingface_hub gguf protobuf
pip uninstall -y -q torchao || true

if [[ "${HUGGING_FACE_HUB_TOKEN}" != "YOUR_TOKEN_HERE" ]]; then
    hf auth login --token "${HUGGING_FACE_HUB_TOKEN}"
else
    echo "WARNING: HUGGING_FACE_HUB_TOKEN not set. Training Qwen3-14B might fail if license is not accepted."
fi

# 2. Run Training
echo "[train] Starting LoRA training (Seq Len: ${ANANT_SEQ_LEN})..."
if [[ "${ANANT_DEVICE_MAP}" == "auto" ]]; then
  python -u lora_train.py 2>&1 | tee -a "${ANANT_OUTPUT_DIR}/logs/train.log"
else
  torchrun --standalone --nproc_per_node="${GPU_COUNT}" lora_train.py 2>&1 | tee -a "${ANANT_OUTPUT_DIR}/logs/train.log"
fi

# 3. Merge Adapters
echo "[merge] Merging LoRA adapters into base model (F16, method=${ANANT_MERGE_METHOD})..."
python lora_merge.py 2>&1 | tee -a "${ANANT_OUTPUT_DIR}/logs/train.log"

# 4. GGUF Conversion (F16)
echo "[gguf] Cloning llama.cpp and converting to GGUF F16..."
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git /kaggle/temp/llama.cpp
CONVERT_SCRIPT="/kaggle/temp/llama.cpp/convert_hf_to_gguf.py"

ARTIFACT_NAME="anant-14b-coder"
MERGED_DIR="${ANANT_OUTPUT_DIR}/merged/${ARTIFACT_NAME}-F16"
GGUF_F16="${ANANT_OUTPUT_DIR}/gguf/${ARTIFACT_NAME}-F16.gguf"

python "${CONVERT_SCRIPT}" "${MERGED_DIR}" --outfile "${GGUF_F16}" --outtype f16

if [[ "${ANANT_KEEP_MERGED}" != "1" ]]; then
  echo "[cleanup] Removing merged HF model to save disk. Set ANANT_KEEP_MERGED=1 to keep it."
  rm -rf "${MERGED_DIR}"
fi

echo "[final] Training pipeline complete."
echo "Artifacts available in ${ANANT_OUTPUT_DIR}"
ls -lh "${ANANT_OUTPUT_DIR}/gguf"
