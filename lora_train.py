import json
import os
import gc
from itertools import islice
from typing import Dict, List, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
import bitsandbytes as bnb
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from peft import prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import AnantConfig


def _cuda_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    return torch.float16


def _memory_snapshot() -> str:
    parts = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(idx)
            parts.append(f"cuda:{idx} free={free / 1024**3:.1f}GiB total={total / 1024**3:.1f}GiB")
    return "; ".join(parts) if parts else "cuda=unavailable"


def _qlora_device_map(local_rank: int):
    return {"": local_rank}


def _clean_messages(messages: List[Dict], system_prompt: str) -> List[Dict[str, str]]:
    cleaned = []
    has_system = False
    for m in messages:
        role = str(m.get("role", "user")).strip().lower()
        content = str(m.get("content", "")).strip()
        if role == "system":
            role = "system"
            has_system = True
        elif role == "assistant":
            role = "assistant"
        else:
            role = "user"
        if content:
            cleaned.append({"role": role, "content": content})
    
    if not has_system:
        cleaned.insert(0, {"role": "system", "content": system_prompt})
    return cleaned


def _tokenize_chat(tokenizer, messages: List[Dict], max_length: int, system_prompt: str) -> Dict[str, List[int]]:
    messages = _clean_messages(messages, system_prompt)
    if not messages:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
    )
    labels = [-100] * len(input_ids)

    prefix: List[Dict[str, str]] = []
    prev_len = 0
    for message in messages:
        current = prefix + [message]
        current_ids = tokenizer.apply_chat_template(
            current,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=max_length,
        )
        current_len = min(len(current_ids), len(input_ids))
        if message["role"] == "assistant" and current_len > prev_len:
            labels[prev_len:current_len] = input_ids[prev_len:current_len]
        prefix = current
        prev_len = current_len
        if prev_len >= len(input_ids):
            break

    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


def _format_example(example: Dict[str, Any], cfg: AnantConfig) -> List[Dict[str, str]]:
    # Standard Chat Format
    if cfg.messages_column in example and example[cfg.messages_column]:
        return example[cfg.messages_column]
    
    # CodeSearchNet
    if "whole_func_string" in example:
        return [
            {"role": "user", "content": f"Complete this code:\n{example.get('func_code_string', '')}"},
            {"role": "assistant", "content": example.get("whole_func_string", "")}
        ]
    
    # The Stack
    if "content" in example and "lang" in example:
        return [
            {"role": "user", "content": f"Write some {example['lang']} code."},
            {"role": "assistant", "content": example["content"]}
        ]

    # XLam (Function Calling) - Formatting into XML schema
    if "query" in example and "answers" in example:
        # XLam answers are usually JSON strings or structured tool calls
        # We wrap them in our <tool_call> XML format to train the model on our specific protocol
        try:
            raw_answer = example["answers"]
            # If it's already structured, we try to preserve it
            if isinstance(raw_answer, str) and raw_answer.strip().startswith("{"):
                formatted_answer = f"<tool_call>\n{raw_answer.strip()}\n</tool_call>"
            else:
                formatted_answer = str(raw_answer)
        except:
            formatted_answer = str(example["answers"])

        return [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": formatted_answer}
        ]

    # Fallback
    prompt = example.get("prompt") or example.get("instruction") or example.get("input")
    response = example.get("response") or example.get("output")
    if prompt and response:
        return [{"role": "user", "content": str(prompt)}, {"role": "assistant", "content": str(response)}]
    
    return []


def _load_training_dataset(cfg: AnantConfig, tokenizer):
    dataset_ids = [item.strip() for item in cfg.dataset_id.split(",") if item.strip()]
    splits = [item.strip() for item in cfg.dataset_split.split(",") if item.strip()]
    if len(splits) == 1 and len(dataset_ids) > 1:
        splits = splits * len(dataset_ids)

    datasets = []
    per_dataset_max = cfg.max_samples // len(dataset_ids)
    for dataset_id, split in zip(dataset_ids, splits):
        print(f"[data] loading {dataset_id} ({split})")
        try:
            # Stream to save local disk/RAM before slicing
            ds = load_dataset(dataset_id, split=split, streaming=True)
            ds_list = list(islice(ds, per_dataset_max))
            ds = Dataset.from_list(ds_list)
            
            ds = ds.map(
                lambda example: _tokenize_chat(tokenizer, _format_example(example, cfg), cfg.seq_len, cfg.system_prompt),
                remove_columns=ds.column_names,
                desc=f"Tokenizing {dataset_id}"
            )
            datasets.append(ds)
        except Exception as e:
            print(f"[data] error loading {dataset_id}: {e}")

    if not datasets:
        raise ValueError("No datasets loaded successfully.")
        
    return concatenate_datasets(datasets) if len(datasets) > 1 else datasets[0]


def _collate_batch(tokenizer, examples: List[Dict]) -> Dict[str, torch.Tensor]:
    labels = [example["labels"] for example in examples]
    features = [{"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]} for example in examples]
    batch = tokenizer.pad(features, padding=True, return_tensors="pt")

    max_len = batch["input_ids"].shape[1]
    padded_labels = []
    for row in labels:
        pad_len = max_len - len(row)
        padded_labels.append(row + [-100] * pad_len)
    batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)
    return batch


def main() -> None:
    cfg = AnantConfig()
    os.makedirs(cfg.adapter_dir, exist_ok=True)

    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    rank = int(os.getenv("RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
    
    if rank == 0:
        print(f"[lora-cuda] artifact={cfg.artifact_name}")
        print(f"[lora-cuda] base_model={cfg.base_model_id}")
        print(f"[lora-cuda] seq_len={cfg.seq_len}")
        print(f"[lora-cuda] memory_before_load={_memory_snapshot()}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=_cuda_dtype(),
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model_id,
        torch_dtype=_cuda_dtype(),
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map=_qlora_device_map(local_rank),
    )
    
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.enable_input_require_grads()
    model.train()
    
    if rank == 0:
        print(f"[lora-cuda] memory_after_lora={_memory_snapshot()}")

    ds = _load_training_dataset(cfg, tokenizer)
    ds = ds.filter(lambda row: len(row["input_ids"]) > 0 and any(l != -100 for l in row["labels"]))
    
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        ds,
        batch_size=cfg.micro_batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        drop_last=True,
        collate_fn=lambda examples: _collate_batch(tokenizer, examples),
    )

    optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=cfg.lora_lr)
    
    step = 0
    optimizer.zero_grad(set_to_none=True)
    
    while step < cfg.lora_steps:
        if sampler is not None:
            sampler.set_epoch(step)
            
        for i, batch in enumerate(loader):
            step += 1
            input_ids = batch["input_ids"].to(f"cuda:{local_rank}")
            attention_mask = batch["attention_mask"].to(f"cuda:{local_rank}")
            labels = batch["labels"].to(f"cuda:{local_rank}")

            with torch.cuda.amp.autocast(dtype=_cuda_dtype()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if not torch.isfinite(loss):
                print(f"[lora] ERROR: Non-finite loss at step {step}")
                continue

            (loss / cfg.grad_accum_steps).backward()

            if step % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if step % cfg.log_every == 0 and rank == 0:
                print(f"[lora] step={step}/{cfg.lora_steps} loss={float(loss):.4f} mem={_memory_snapshot()}")

            if step % cfg.save_every == 0 and rank == 0:
                model.save_pretrained(os.path.join(cfg.adapter_dir, f"checkpoint-{step}"))

            if step >= cfg.lora_steps:
                break

    if rank == 0:
        model.save_pretrained(cfg.adapter_dir)
        tokenizer.save_pretrained(cfg.adapter_dir)
        print(f"[lora] training finished -> {cfg.adapter_dir}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
