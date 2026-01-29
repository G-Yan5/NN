from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_curve
from torch.utils.data import DataLoader
from transformers import RobertaTokenizerFast
from transformers import logging as hf_logging

from .data import Collator, YelpMultimodalDataset
from .models import EfficientNetImageEncoder, FusionClassifier, RobertaTextEncoder
from .utils import compute_binary_metrics, ensure_dir, get_device, save_json

hf_logging.set_verbosity_error()


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--tune_threshold", action="store_true")
    ap.add_argument("--use_tuned_threshold", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = get_device(cfg["project"]["device"])
    outputs_dir = cfg["paths"]["outputs_dir"]
    ensure_dir(outputs_dir)

    # Load checkpoint
    ckpt_path = args.ckpt or os.path.join(outputs_dir, "best_model.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

    # Tokenizer (for pad id)
    tokenizer = RobertaTokenizerFast.from_pretrained(cfg["model"]["text"]["name"])

    # Dataset + loader
    ds = YelpMultimodalDataset(
        csv_path=cfg["paths"]["processed_csv"],
        images_dir=cfg["paths"]["images_dir"],
        split=args.split,
        tokenizer_name=cfg["model"]["text"]["name"],
        max_length=int(cfg["model"]["text"]["max_length"]),
        image_size=224,
    )

    dl = DataLoader(
        ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["project"]["num_workers"]),
        collate_fn=Collator(pad_token_id=tokenizer.pad_token_id),
    )

    # Build model
    model = FusionClassifier(
        RobertaTextEncoder(cfg["model"]["text"]["name"]),
        EfficientNetImageEncoder(cfg["model"]["image"]["name"]),
        hidden_dim=int(cfg["model"]["fusion"]["hidden_dim"]),
        dropout=float(cfg["model"]["fusion"]["dropout"]),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Run inference
    criterion = nn.BCEWithLogitsLoss()
    all_probs, all_true = [], []
    total_loss, total_n = 0.0, 0

    with torch.no_grad():
        for batch in dl:
            logits = model(
                batch.input_ids.to(device),
                batch.attention_mask.to(device),
                batch.pixel_values.to(device),
            )

            y = batch.label.to(device).float()
            loss = criterion(logits, y)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu().numpy())
            all_true.append(y.cpu().numpy())

            total_loss += loss.item() * y.size(0)
            total_n += y.size(0)

    y_true = np.concatenate(all_true)
    y_prob = np.concatenate(all_probs)

    # Pick threshold
    threshold = 0.5

    if args.tune_threshold:
        # Tune on val only
        assert args.split == "val", "Tune threshold only on val"

        prec, rec, th = precision_recall_curve(y_true, y_prob)
        f1 = 2 * prec * rec / (prec + rec + 1e-8)
        idx = int(np.argmax(f1))
        threshold = float(th[idx])

        save_json(os.path.join(outputs_dir, "tuned_threshold.json"), {"threshold": threshold})
        print(f"Tuned threshold (val): {threshold:.3f}")

    if args.use_tuned_threshold:
        # Load saved threshold
        tpath = os.path.join(outputs_dir, "tuned_threshold.json")
        threshold = float(json.load(open(tpath, "r", encoding="utf-8"))["threshold"])
        print(f"Using tuned threshold: {threshold:.3f}")

    # Metrics
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_binary_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / max(1, total_n)
    metrics["threshold"] = float(threshold)

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({args.split})")
    plt.xlabel("Pred")
    plt.ylabel("True")
    plt.tight_layout()

    cm_path = os.path.join(outputs_dir, f"confusion_{args.split}.png")
    plt.savefig(cm_path)
    plt.close()

    # Save metrics
    save_json(os.path.join(outputs_dir, f"metrics_{args.split}.json"), metrics)

    print(f"\nSplit: {args.split}")
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
