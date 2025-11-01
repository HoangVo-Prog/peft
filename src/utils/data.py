from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, DataCollatorWithPadding


from ..utils.config import GLUE_SENTENCE_KEYS, is_regression_task




def load_glue_and_tokenizer(task: str, model_name: str):
    task = task.lower()
    if task not in GLUE_SENTENCE_KEYS:
        raise ValueError(f"Unknown GLUE task: {task}")


    raw: DatasetDict = load_dataset("glue", task)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


    sent1_key, sent2_key = GLUE_SENTENCE_KEYS[task]


    def preprocess(batch):
        if sent2_key is None:
            tokenized = tokenizer(batch[sent1_key], truncation=True)
        else:
            tokenized = tokenizer(batch[sent1_key], batch[sent2_key], truncation=True)
        if "label" in batch:
            tokenized["labels"] = batch["label"]
        return tokenized


    encoded = raw.map(preprocess, batched=True, remove_columns=raw["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)


    if is_regression_task(task):
        num_labels = 1
        label_list = None
    else:
        label_list = raw["train"].features["label"].names
        num_labels = len(label_list)


    return encoded, tokenizer, collator, num_labels, label_list