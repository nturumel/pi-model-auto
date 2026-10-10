"""Parameterized variant of pi-model-auto/training/train_cost_sensitive.py for experiments.

Same data, split, labels and cost-sensitive loss as the fork's recipe; adds a choice of base model,
epochs, learning rate, and evaluation on the held-out test split (never used for model selection).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

DATASET_ID = "anasnassar/llm-query-complexity-benchmark"
DATASET_REVISION = "e111dc811aea9103010db741db9af0505240e77c"
LABELS = ["LOW", "MEDIUM", "HIGH"]
L2I = {label: i for i, label in enumerate(LABELS)}
COSTS = [[0.0, 0.15, 0.50], [1.0, 0.0, 0.15], [5.0, 1.0, 0.0]]


def load(split):
    path = hf_hub_download(DATASET_ID, f"{split}.jsonl", repo_type="dataset", revision=DATASET_REVISION)
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def routing_metrics(y, p):
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro")),
        "high_recall": float((p[y == 2] == 2).mean()),
        "underroute_rate": float((p < y).mean()),
        "severe_underroute_rate": float((p + 1 < y).mean()),
    }


class CostSensitiveTrainer(Trainer):
    def __init__(self, *args, risk_lambda: float, **kwargs):
        super().__init__(*args, **kwargs)
        self.risk_lambda = risk_lambda

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        ce = F.cross_entropy(outputs.logits, labels)
        costs = outputs.logits.new_tensor(COSTS)
        risk = (outputs.logits.softmax(-1) * costs[labels]).sum(-1).mean()
        loss = ce + self.risk_lambda * risk
        return (loss, outputs) if return_outputs else loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--risk-lambda", type=float, default=0.5)
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    records = load("train")
    train_rows, val_rows = train_test_split(
        records, test_size=0.2, random_state=42, stratify=[r["ground_truth"] for r in records])
    test_rows = load("test")

    tok = AutoTokenizer.from_pretrained(args.base, revision=args.revision)

    def ds(rows):
        enc = tok([r["text"] for r in rows], truncation=True, max_length=128)
        enc["labels"] = [L2I[r["ground_truth"]] for r in rows]
        return Dataset.from_dict(dict(enc))

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, revision=args.revision, num_labels=3,
        id2label=dict(enumerate(LABELS)), label2id=L2I, ignore_mismatched_sizes=True)

    def compute_metrics(result):
        m = routing_metrics(result.label_ids, result.predictions.argmax(-1))
        m["routing_score"] = m["macro_f1"] - 4.0 * m["severe_underroute_rate"] - 0.5 * m["underroute_rate"]
        return m

    targs = TrainingArguments(
        output_dir=str(args.output / "checkpoints"), learning_rate=args.lr, weight_decay=0.01,
        warmup_ratio=0.1, num_train_epochs=args.epochs, per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=64, dataloader_pin_memory=False, eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="routing_score", greater_is_better=True, report_to=[], seed=args.seed, data_seed=args.seed)
    trainer = CostSensitiveTrainer(
        model=model, args=targs, train_dataset=ds(train_rows), eval_dataset=ds(val_rows),
        processing_class=tok, compute_metrics=compute_metrics, risk_lambda=args.risk_lambda)
    t0 = time.time()
    trainer.train()
    train_secs = time.time() - t0
    val = trainer.evaluate()
    pred = trainer.predict(ds(test_rows))
    logits = pred.predictions
    y = pred.label_ids
    probs = torch.tensor(logits).softmax(-1).numpy()
    floor = np.where(probs.max(-1) < 0.45, 2, logits.argmax(-1))
    test = {"argmax": routing_metrics(y, logits.argmax(-1)), "floor045": routing_metrics(y, floor)}

    best = args.output / "best"
    trainer.save_model(best)
    tok.model_max_length = 128
    tok.save_pretrained(best)
    np.save(args.output / "val_logits.npy", trainer.predict(ds(val_rows)).predictions)
    np.save(args.output / "test_logits.npy", logits)
    out = {"args": {k: str(v) for k, v in vars(args).items()}, "train_secs": round(train_secs),
           "val": val, "test": test}
    (args.output / "metrics.json").write_text(json.dumps(out, indent=2))
    print("RESULT", json.dumps({"base": args.base, "lambda": args.risk_lambda, "epochs": args.epochs,
                                "lr": args.lr, "secs": round(train_secs), "test": test["floor045"]}))


if __name__ == "__main__":
    main()
