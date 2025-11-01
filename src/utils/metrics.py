import numpy as np
import evaluate

from ..utils.config import is_regression_task


def get_best_metric_for_task(task: str) -> str:
    task = task.lower()
    if task == "cola":
        return "matthews_correlation"
    if task == "stsb":
        return "combined_score"
    if task in {"mrpc", "qqp"}:
        return "f1"
    return "accuracy"


def build_compute_metrics(task: str):
    task = task.lower()
    metric = evaluate.load("glue", task)
    is_reg = is_regression_task(task)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if is_reg:
            preds = np.squeeze(preds)
            res = metric.compute(predictions=preds, references=labels)
            res["combined_score"] = float((res.get("pearson", 0.0) + res.get("spearmanr", 0.0)) / 2.0)
            return res
        preds = np.argmax(preds, axis=1)
        res = metric.compute(predictions=preds, references=labels)
        if task in {"mrpc", "qqp"}:
            res["combined_score"] = float((res.get("f1", 0.0) + res.get("accuracy", 0.0)) / 2.0)
        return res

    return compute_metrics