from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    # Reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    # Create folder if needed
    if path:
        os.makedirs(path, exist_ok=True)


def save_json(path: str, obj: Dict[str, Any]) -> None:
    # Save dict to json (handles numpy)
    def convert(o):
        if isinstance(o, np.generic):
            return o.item()
        raise TypeError

    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=convert)


def get_device(device_str: str) -> torch.device:
    # Pick cpu/cuda
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        # Add batch value
        self.total += float(value) * n
        self.count += int(n)

    @property
    def avg(self) -> float:
        # Mean value
        return self.total / max(1, self.count)


@torch.no_grad()
def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Basic binary metrics
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = (2 * prec * rec) / max(1e-12, (prec + rec))

    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def set_requires_grad(module: torch.nn.Module, flag: bool) -> None:
    # Freeze/unfreeze module
    for p in module.parameters():
        p.requires_grad = flag


def count_trainable_params(module: torch.nn.Module) -> int:
    # Count trainable params
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))
