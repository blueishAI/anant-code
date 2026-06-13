import os
from dataclasses import dataclass


@dataclass
class AnantConfig:
    # Artifact identity
    param_label: str = os.getenv("ANANT_PARAM_LABEL", "14b")
    variant: str = os.getenv("ANANT_VARIANT", "coder")

    # Hugging Face source
    base_model_id: str = os.getenv("ANANT_BASE_MODEL", "Qwen/Qwen3-14B")

    # LoRA training schedule
    seq_len: int = int(os.getenv("ANANT_SEQ_LEN", "4096"))  # Increased for complex code
    micro_batch_size: int = int(os.getenv("ANANT_MICRO_BATCH", "1"))
    grad_accum_steps: int = int(os.getenv("ANANT_GRAD_ACCUM", "8")) # Increased for 4096 seq_len stability
    log_every: int = int(os.getenv("ANANT_LOG_EVERY", "5"))
    save_every: int = int(os.getenv("ANANT_SAVE_EVERY", "250"))
    lora_r: int = int(os.getenv("ANANT_LORA_R", "32"))
    lora_alpha: int = int(os.getenv("ANANT_LORA_ALPHA", "64"))
    lora_dropout: float = float(os.getenv("ANANT_LORA_DROPOUT", "0.05"))
    lora_lr: float = float(os.getenv("ANANT_LORA_LR", "1e-4"))
    lora_steps: int = int(os.getenv("ANANT_LORA_STEPS", "2000"))

    # Data
    dataset_id: str = os.getenv("ANANT_DATASET", "bigcode/the-stack-smol,microsoft/code_search_net,Salesforce/xlam-function-calling-60k")
    dataset_split: str = os.getenv("ANANT_DATASET_SPLIT", "train,train,train")
    max_samples: int = int(os.getenv("ANANT_MAX_SAMPLES", "50000"))
    messages_column: str = os.getenv("ANANT_MESSAGES_COLUMN", "messages")

    # Paths
    work_dir: str = os.getenv("ANANT_WORK_DIR", "/kaggle/working")
    output_dir: str = os.getenv("ANANT_OUTPUT_DIR", "/kaggle/working/output_anant")
    hf_cache: str = os.getenv("HF_HOME", "/kaggle/temp/hf_cache")

    @property
    def artifact_name(self) -> str:
        return f"anant-{self.param_label}-{self.variant}"

    @property
    def adapter_dir(self) -> str:
        return os.path.join(self.output_dir, "adapters", self.artifact_name)

    @property
    def merged_dir(self) -> str:
        return os.path.join(self.output_dir, "merged", f"{self.artifact_name}-F16")

    @property
    def gguf_dir(self) -> str:
        return os.path.join(self.output_dir, "gguf")

    @property
    def system_prompt(self) -> str:
        return (
            "You are Anant-Code, a high-performance autonomous coding agent. "
            "Your goal is to assist with complex programming tasks, system architecture, and automation.\n\n"
            "## Tool Usage Protocol\n"
            "You have access to specialized tools. To use a tool, you MUST use the following XML format:\n"
            "<tool_call>\n"
            "{\"name\": \"tool_name\", \"arguments\": {\"arg1\": \"value1\"}}\n"
            "</tool_call>\n\n"
            "After a tool call, you will receive a <tool_response> with the result. "
            "Available tools: WriteFile, ReadFile, ReadFiles, WriteFiles, WebSearch, RunCommand, "
            "ListDirectory, EditFile, GrepSearch, MoveFile, DeleteFile, GetFileTree, BrowserTool, "
            "AskUser, MemoryWrite, MemoryRead, RunTests, GitTool.\n\n"
            "## Guidelines\n"
            "- Be concise and prioritize efficient, modern code.\n"
            "- Always use tools when they are the most effective path to a solution.\n"
            "- If a task is ambiguous, use AskUser or perform research with WebSearch/ReadFile.\n"
            "- Ensure all code is well-documented and adheres to best practices."
        )
