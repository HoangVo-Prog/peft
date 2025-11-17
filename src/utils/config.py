from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List


@dataclass
class RunConfig:
    # Task control
    all: bool = False                      # True nếu muốn chạy tất cả GLUE tasks
    task_name: str = "sst2"                # cola sst2 mrpc qqp stsb mnli qnli rte wnli

    # Model and I/O
    model_name: str = "roberta-large"
    output_dir: str = "./outputs"

    # Sequence length cho bước tokenize, quan trọng cho VRAM
    max_seq_length: int = 128              # 128 là hợp lý cho QQP, MNLI trên P100 16GB

    # Training hyperparams
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 16  # 16 cho roberta large + LoRA khá an toàn
    per_device_eval_batch_size: int = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    logging_steps: int = 100
    seed: int = 42

    # Distributed
    ddp: bool = False

    # Saving and eval
    save_strategy: str = "epoch"           # "epoch" hoặc "steps"
    eval_strategy: str = "epoch"           # "epoch" hoặc "steps"
    save_total_limit: int = 1

    # Có bật gradient cho toàn bộ model hay không
    # Với LoRA nên để False, chỉ train adapter
    gradient_enable: bool = False

    # Precision
    fp16: bool = True                      # P100 dùng fp16 là hợp
    bf16: bool = False                     # để False cho P100 cho chắc

    # W&B settings
    wandb_enable: bool = True
    wandb_project: Optional[str] = "glue"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_offline_fallback: bool = True


@dataclass
class LoRAArgs:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    bias: str = "none"  # "none", "all", hoặc "lora_only"

    # Cho roberta = attention projection
    target_modules: List[str] = field(
        default_factory=lambda: ["query", "key", "value"]
    )

    # Thường cần lưu classifier để head vẫn finetune
    modules_to_save: List[str] = field(
        default_factory=lambda: ["classifier"]
    )

    seed: int = 42

    # Với roberta large trên GLUE nên bật gradient checkpointing
    gradient_checkpointing: bool = True


GLUE_SENTENCE_KEYS: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

GLUE_TASKS = tuple(GLUE_SENTENCE_KEYS.keys())


def is_regression_task(task: str) -> bool:
    return task.lower() == "stsb"
