import gc
import json
import os
import shutil
from importlib import metadata
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from config import AnantConfig


def _patch_incompatible_torchao() -> None:
    try:
        version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return

    major, minor, *_ = (int(part) for part in version.split(".")[:2])
    if (major, minor) >= (0, 16):
        return

    import peft.tuners.lora.torchao as peft_torchao

    print(f"[merge] disabling incompatible torchao {version}; FP16 merge does not need it")
    peft_torchao.is_torchao_available = lambda: False


def _base_weight_name(adapter_prefix: str) -> str:
    prefix = "base_model.model."
    if adapter_prefix.startswith(prefix):
        adapter_prefix = adapter_prefix[len(prefix):]
    return f"{adapter_prefix}.weight"


def _adapter_suffix(key: str, marker: str) -> str:
    return key.split(marker, 1)[1]


def _load_lora_weights(adapter_dir: str) -> dict[str, dict[str, torch.Tensor]]:
    adapter_path = Path(adapter_dir) / "adapter_model.safetensors"
    if not adapter_path.exists():
        raise RuntimeError(f"Missing adapter safetensors: {adapter_path}")

    raw = load_file(str(adapter_path), device="cpu")
    pairs: dict[str, dict[str, torch.Tensor]] = {}

    for key, tensor in raw.items():
        if ".lora_A." in key:
            prefix = key.split(".lora_A.", 1)[0]
            suffix = _adapter_suffix(key, ".lora_A.")
            pairs.setdefault(_base_weight_name(prefix), {})[f"A:{suffix}"] = tensor
        elif ".lora_B." in key:
            prefix = key.split(".lora_B.", 1)[0]
            suffix = _adapter_suffix(key, ".lora_B.")
            pairs.setdefault(_base_weight_name(prefix), {})[f"B:{suffix}"] = tensor

    merged: dict[str, dict[str, torch.Tensor]] = {}
    for weight_name, parts in pairs.items():
        suffixes = sorted({key[2:] for key in parts})
        for suffix in suffixes:
            a = parts.get(f"A:{suffix}")
            b = parts.get(f"B:{suffix}")
            if a is not None and b is not None:
                merged[weight_name] = {"A": a, "B": b}
                break

    if not merged:
        raise RuntimeError("No LoRA A/B tensors found in adapter.")

    return merged


def _load_base_shards(base_dir: str) -> tuple[list[Path], dict[str, str] | None]:
    base_path = Path(base_dir)
    index_path = base_path / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open("r", encoding="utf-8") as f:
            index = json.load(f)
        shard_names = sorted(set(index["weight_map"].values()))
        return [base_path / name for name in shard_names], index

    single = base_path / "model.safetensors"
    if single.exists():
        return [single], None

    raise RuntimeError(f"No safetensors checkpoint found in {base_dir}")


def _scaling(adapter_dir: str) -> float:
    with open(Path(adapter_dir) / "adapter_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    return float(cfg["lora_alpha"]) / float(cfg["r"])


def _save_configs(base_model_id: str, adapter_dir: str, out_dir: str) -> None:
    AutoConfig.from_pretrained(base_model_id).save_pretrained(out_dir)
    try:
        GenerationConfig.from_pretrained(base_model_id).save_pretrained(out_dir)
    except Exception:
        pass

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, use_fast=True)
    tokenizer.save_pretrained(out_dir)


def stream_merge() -> None:
    cfg = AnantConfig()
    if not os.path.isdir(cfg.adapter_dir):
        raise RuntimeError(f"Missing LoRA adapter directory: {cfg.adapter_dir}")

    print(f"[merge] downloading/locating base safetensors: {cfg.base_model_id}")
    base_dir = snapshot_download(
        cfg.base_model_id,
        allow_patterns=[
            "*.safetensors",
            "*.safetensors.index.json",
            "config.json",
            "generation_config.json",
            "tokenizer*",
            "*.model",
            "*.json",
        ],
    )

    lora = _load_lora_weights(cfg.adapter_dir)
    scale = _scaling(cfg.adapter_dir)
    base_shards, base_index = _load_base_shards(base_dir)

    out_dir = Path(cfg.merged_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge] streaming merge {len(lora)} LoRA tensors over {len(base_shards)} shard(s)")
    weight_map: dict[str, str] = {}
    total_size = 0

    for idx, shard_path in enumerate(base_shards, start=1):
        print(f"[merge] shard {idx}/{len(base_shards)}: {shard_path.name}", flush=True)
        tensors = load_file(str(shard_path), device="cpu")
        out_tensors = {}

        for name, tensor in tensors.items():
            if name in lora:
                a = lora[name]["A"].float()
                b = lora[name]["B"].float()
                tensor = tensor + (b @ a).to(dtype=tensor.dtype) * scale
            out_tensors[name] = tensor
            weight_map[name] = shard_path.name
            total_size += tensor.numel() * tensor.element_size()

        save_file(out_tensors, str(out_dir / shard_path.name), metadata={"format": "pt"})
        del tensors, out_tensors
        gc.collect()

    if base_index is not None:
        index = {
            "metadata": {"total_size": total_size},
            "weight_map": weight_map,
        }
        with (out_dir / "model.safetensors.index.json").open("w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, sort_keys=True)

    _save_configs(cfg.base_model_id, cfg.adapter_dir, str(out_dir))
    print(f"[merge] merged model saved -> {out_dir}")


def peft_merge() -> None:
    cfg = AnantConfig()
    if not os.path.isdir(cfg.adapter_dir):
        raise RuntimeError(f"Missing LoRA adapter directory: {cfg.adapter_dir}")

    print(f"[merge] loading base model to CPU: {cfg.base_model_id}")
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map="cpu",
    )

    print(f"[merge] loading adapter: {cfg.adapter_dir}")
    _patch_incompatible_torchao()
    model = PeftModel.from_pretrained(base, cfg.adapter_dir)

    print("[merge] merging and unloading on CPU...")
    model = model.merge_and_unload()

    print(f"[merge] saving merged model to: {cfg.merged_dir}")
    tok = AutoTokenizer.from_pretrained(cfg.adapter_dir, use_fast=True)
    os.makedirs(cfg.merged_dir, exist_ok=True)

    model.save_pretrained(
        cfg.merged_dir,
        safe_serialization=True,
        max_shard_size=os.getenv("ANANT_MERGE_SHARD_SIZE", "2GB"),
    )
    tok.save_pretrained(cfg.merged_dir)
    print(f"[merge] merged model saved -> {cfg.merged_dir}")


def main() -> None:
    method = os.getenv("ANANT_MERGE_METHOD", "stream").strip().lower()
    if method == "peft":
        peft_merge()
    elif method == "stream":
        stream_merge()
    else:
        raise ValueError("ANANT_MERGE_METHOD must be 'stream' or 'peft'")


if __name__ == "__main__":
    main()
