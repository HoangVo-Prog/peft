from dataclasses import dataclass
from typing import Optional, Tuple, Dict




@dataclass
class RunConfig:
    task_name: str = "sst2" # cola sst2 mrpc qqp stsb mnli qnli rte wnli
    model_name: str = None 
    output_dir: str = "./outputs"
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 64
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    logging_steps: int = 100
    seed: int = 42
    save_strategy: str = "epoch" # or "steps"
    eval_strategy: str = "epoch" # or "steps"
    save_total_limit: int = 2
    fp16: bool = True
    bf16: bool = False
    # W&B settings
    wandb_enable: bool = True
    wandb_project: Optional[str] = "glue"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_offline_fallback: bool = True




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




def is_regression_task(task: str) -> bool:
    return task.lower() == "stsb"