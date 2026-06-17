#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

INPUT_ROOT="${ANANT_INPUT_ROOT:-/kaggle/input/notebooks/aaravmaloo/anant-coder}"
INPUT_OUTPUT="${INPUT_ROOT}/output_anant"
WORK_OUTPUT="${ANANT_OUTPUT_DIR:-/kaggle/working/output_anant}"
ARTIFACT_NAME="${ANANT_ARTIFACT_NAME:-anant-14b-coder}"
BASE_MODEL="${ANANT_BASE_MODEL:-Qwen/Qwen3-14B}"
GGUF_TYPE="${ANANT_GGUF_TYPE:-q8_0}"
GGUF_OUT="${ANANT_GGUF_OUT:-/kaggle/working/${ARTIFACT_NAME}-${GGUF_TYPE}.gguf}"
LLAMA_CPP_DIR="/kaggle/working/llama.cpp"

export ANANT_WORK_DIR="/kaggle/working"
export ANANT_OUTPUT_DIR="${WORK_OUTPUT}"
export HF_HOME="${HF_HOME:-/kaggle/temp/hf_cache}"
export ANANT_ADAPTER_DIR="${WORK_OUTPUT}/adapters/${ARTIFACT_NAME}"
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

echo "[gguf] clone llama.cpp"
git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMA_CPP_DIR}"

echo "[patch] direct LoRA merge during GGUF conversion"
python - <<'PY'
from pathlib import Path

base_py = Path("/kaggle/working/llama.cpp/conversion/base.py")
text = base_py.read_text()

inject = r'''

_ANANT_LORA = None

def _anant_base_weight_name(adapter_prefix: str) -> str:
    prefix = "base_model.model."
    if adapter_prefix.startswith(prefix):
        adapter_prefix = adapter_prefix[len(prefix):]
    return f"{adapter_prefix}.weight"

def _anant_load_lora():
    global _ANANT_LORA
    if _ANANT_LORA is not None:
        return _ANANT_LORA

    adapter_dir = os.environ.get("ANANT_ADAPTER_DIR")
    if not adapter_dir:
        _ANANT_LORA = ({}, 1.0)
        return _ANANT_LORA

    from safetensors.torch import load_file

    adapter_path = Path(adapter_dir) / "adapter_model.safetensors"
    config_path = Path(adapter_dir) / "adapter_config.json"
    raw = load_file(str(adapter_path), device="cpu")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    scale = float(cfg["lora_alpha"]) / float(cfg["r"])

    tmp = {}
    for key, tensor in raw.items():
        if ".lora_A." in key:
            prefix, suffix = key.split(".lora_A.", 1)
            tmp.setdefault(_anant_base_weight_name(prefix), {})[f"A:{suffix}"] = tensor
        elif ".lora_B." in key:
            prefix, suffix = key.split(".lora_B.", 1)
            tmp.setdefault(_anant_base_weight_name(prefix), {})[f"B:{suffix}"] = tensor

    lora = {}
    for name, parts in tmp.items():
        for suffix in sorted({k[2:] for k in parts}):
            a = parts.get(f"A:{suffix}")
            b = parts.get(f"B:{suffix}")
            if a is not None and b is not None:
                lora[name] = (a.float(), b.float())
                break

    print(f"[anant-lora] loaded {len(lora)} LoRA tensors from {adapter_dir}", flush=True)
    _ANANT_LORA = (lora, scale)
    return _ANANT_LORA

def _anant_wrap_lora(name, gen):
    lora, scale = _anant_load_lora()
    if name not in lora:
        return gen

    def wrapped():
        base = LazyTorchTensor.to_eager(gen())
        a, b = lora[name]
        delta = (b @ a).to(dtype=base.dtype) * scale
        return base + delta

    return wrapped
'''

if "_anant_wrap_lora" not in text:
    marker = "try:\n    from mistral_common"
    text = text.replace(marker, inject + "\n" + marker)

text = text.replace(
    "if titem := self.filter_tensors((name, data_gen)):",
    "data_gen = _anant_wrap_lora(name, data_gen)\n                if titem := self.filter_tensors((name, data_gen)):",
)
base_py.write_text(text)
PY

echo "[gguf] remote base + adapter -> ${GGUF_OUT} (${GGUF_TYPE})"
python "${LLAMA_CPP_DIR}/convert_hf_to_gguf.py" "${BASE_MODEL}" --remote --outfile "${GGUF_OUT}" --outtype "${GGUF_TYPE}" 2>&1 | tee -a "${WORK_OUTPUT}/logs/merge.log"

echo "[clean] remove temp files"
rm -rf "${LLAMA_CPP_DIR}" "${WORK_OUTPUT}/adapters"

echo "[done] GGUF:"
ls -lh "${GGUF_OUT}"
