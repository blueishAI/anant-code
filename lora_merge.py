import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from config import AnantConfig


def main() -> None:
    cfg = AnantConfig()

    if not os.path.isdir(cfg.adapter_dir):
        raise RuntimeError(f"Missing LoRA adapter directory: {cfg.adapter_dir}")

    print(f"[merge] loading base model to CPU: {cfg.base_model_id}")
    # Loading to CPU (30GB+ RAM on Kaggle) to guarantee no OOM during merge_and_unload
    base = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    
    print(f"[merge] loading adapter: {cfg.adapter_dir}")
    model = PeftModel.from_pretrained(base, cfg.adapter_dir)
    
    print("[merge] merging and unloading on CPU (this may take a few minutes)...")
    model = model.merge_and_unload()

    print(f"[merge] saving merged model to: {cfg.merged_dir}")
    tok = AutoTokenizer.from_pretrained(cfg.adapter_dir, use_fast=True)
    os.makedirs(cfg.merged_dir, exist_ok=True)
    
    model.save_pretrained(cfg.merged_dir, safe_serialization=True)
    tok.save_pretrained(cfg.merged_dir)
    print(f"[merge] merged model saved -> {cfg.merged_dir}")


if __name__ == "__main__":
    main()
