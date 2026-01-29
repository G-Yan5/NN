from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import precision_recall_curve  # (kept if you later need it)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import RobertaTokenizerFast
from transformers import logging as hf_logging

from .data import Collator, YelpMultimodalDataset
from .models import EfficientNetImageEncoder, FusionClassifier, RobertaTextEncoder
from .utils import (
    AverageMeter,
    compute_binary_metrics,
    count_trainable_params,
    ensure_dir,
    get_device,
    save_json,
    set_requires_grad,
    set_seed,
)

hf_logging.set_verbosity_error()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def unfreeze_last_roberta_layers(roberta_model: nn.Module, n_layers: int) -> None:
    # Unfreeze last N transformer layers
    if n_layers <= 0:
        return
    layers = roberta_model.encoder.layer
    n_layers = min(n_layers, len(layers))
    for layer in layers[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True


def unfreeze_last_efficientnet_blocks(effnet_backbone: nn.Module, n_blocks: int) -> None:
    # Unfreeze last N EfficientNet blocks
    if n_blocks <= 0:
        return
    feats = effnet_backbone.features
    n_blocks = min(n_blocks, len(feats))
    for block in feats[-n_blocks:]:
        for p in block.parameters():
            p.requires_grad = True


def build_optimizer_two_lr(
    model: FusionClassifier,
    head_lr: float,
    backbone_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    # Two LRs: head (high), unfrozen backbone (low)
    head_params = [p for p in model.head.parameters() if p.requires_grad]

    backbone_params: List[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("head."):
            continue
        backbone_params.append(p)

    groups = [{"params": head_params, "lr": head_lr, "weight_decay": weight_decay}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})

    return torch.optim.AdamW(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["project"]["seed"]))

    device = get_device(cfg["project"]["device"])
    print(f"Device: {device.type}")

    outputs_dir = cfg["paths"]["outputs_dir"]
    ensure_dir(outputs_dir)

    processed_csv = cfg["paths"]["processed_csv"]
    images_dir = cfg["paths"]["images_dir"]

    # Data & tokenizer
    text_name = cfg["model"]["text"]["name"]
    max_length = int(cfg["model"]["text"]["max_length"])
    tokenizer = RobertaTokenizerFast.from_pretrained(text_name)

    ds_train = YelpMultimodalDataset(processed_csv, images_dir, "train", text_name, max_length, image_size=224)
    ds_val = YelpMultimodalDataset(processed_csv, images_dir, "val", text_name, max_length, image_size=224)

    collate = Collator(pad_token_id=tokenizer.pad_token_id)

    nw = int(cfg["project"]["num_workers"])
    use_pw = nw > 0

    dl_train = DataLoader(
        ds_train,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_pw,
        prefetch_factor=2 if use_pw else None,
        collate_fn=collate,
    )

    dl_val = DataLoader(
        ds_val,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=nw,
        pin_memory=(device.type == "cuda"),
        persistent_workers=use_pw,
        prefetch_factor=2 if use_pw else None,
        collate_fn=collate,
    )

    # Model
    model = FusionClassifier(
        text_encoder=RobertaTextEncoder(text_name, dropout=float(cfg["model"]["text"]["dropout"])),
        image_encoder=EfficientNetImageEncoder(
            cfg["model"]["image"]["name"],
            pretrained=bool(cfg["model"]["image"]["pretrained"]),
            dropout=float(cfg["model"]["image"]["dropout"]),
        ),
        hidden_dim=int(cfg["model"]["fusion"]["hidden_dim"]),
        dropout=float(cfg["model"]["fusion"]["dropout"]),
    ).to(device)

    # Resume 
    ckpt_path = os.path.join(outputs_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"], strict=True)
            print(f"Loaded: {ckpt_path}")

    # Loss (with capped pos_weight)
    df = pd.read_csv(processed_csv)
    train_y = df[df["split"] == "train"]["label"].values
    neg = int((train_y == 0).sum())
    pos = int((train_y == 1).sum())

    raw_ratio = neg / max(1, pos)
    pos_weight_value = min(raw_ratio, 2.0)

    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Train labels: neg={neg}, pos={pos}, pos_weight={pos_weight_value:.2f}")

    # Freeze then unfreeze
    warmup_epochs = int(cfg["train"].get("warmup_epochs", 5))
    unfreeze_roberta_layers_n = int(cfg["train"].get("unfreeze_roberta_layers", 1))
    unfreeze_effnet_blocks_n = int(cfg["train"].get("unfreeze_efficientnet_blocks", 1))
    backbone_lr = float(cfg["train"].get("backbone_lr", 2e-5))

    head_lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])

    # Stage-1: head only
    set_requires_grad(model.text_encoder, False)
    set_requires_grad(model.image_encoder, False)
    set_requires_grad(model.head, True)

    opt = build_optimizer_two_lr(model, head_lr=head_lr, backbone_lr=backbone_lr, weight_decay=wd)
    print(f"Stage-1: warmup_epochs={warmup_epochs}, head_lr={head_lr:g}")

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Loop
    best_metric_name = cfg["train"]["save_best_metric"]
    best_score = -1.0
    patience = int(cfg["train"]["early_stop_patience"])
    bad_epochs = 0
    total_epochs = int(cfg["train"]["epochs"])
    stage2_started = False

    train_loss_hist, val_loss_hist = [], []
    train_acc_hist, val_acc_hist = [], []
    train_f1_hist, val_f1_hist = [], []

    for epoch in range(1, total_epochs + 1):
        # Stage-2 switch
        if (not stage2_started) and (epoch == warmup_epochs + 1):
            stage2_started = True

            set_requires_grad(model.text_encoder, False)
            set_requires_grad(model.image_encoder, False)
            set_requires_grad(model.head, True)

            unfreeze_last_roberta_layers(model.text_encoder.model, unfreeze_roberta_layers_n)
            unfreeze_last_efficientnet_blocks(model.image_encoder.backbone, unfreeze_effnet_blocks_n)

            opt = build_optimizer_two_lr(model, head_lr=head_lr, backbone_lr=backbone_lr, weight_decay=wd)
            scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

            print(
                f"Stage-2: roberta_last={unfreeze_roberta_layers_n}, "
                f"effnet_last={unfreeze_effnet_blocks_n}, backbone_lr={backbone_lr:g}"
            )

        # Train
        model.train()
        loss_meter = AverageMeter()

        train_correct = 0
        train_total = 0
        train_tp = train_fp = train_fn = 0

        for batch in tqdm(dl_train, desc=f"Epoch {epoch}/{total_epochs} [train]", leave=True, dynamic_ncols=True):
            opt.zero_grad(set_to_none=True)

            input_ids = batch.input_ids.to(device)
            attention_mask = batch.attention_mask.to(device)
            pixel_values = batch.pixel_values.to(device)
            y = batch.label.to(device).float()

            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(input_ids, attention_mask, pixel_values)
                loss = criterion(logits, y)

            probs = torch.sigmoid(logits)
            pred = (probs >= 0.5).long()
            y_int = y.long()

            train_tp += int(((pred == 1) & (y_int == 1)).sum().item())
            train_fp += int(((pred == 1) & (y_int == 0)).sum().item())
            train_fn += int(((pred == 0) & (y_int == 1)).sum().item())

            train_correct += int((pred == y_int).sum().item())
            train_total += int(y.size(0))

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            loss_meter.update(loss.item(), n=y.size(0))

        train_acc = train_correct / max(1, train_total)
        train_precision = train_tp / max(1, train_tp + train_fp)
        train_recall = train_tp / max(1, train_tp + train_fn)
        train_f1 = (2 * train_precision * train_recall) / max(1e-12, train_precision + train_recall)

        # Val
        model.eval()
        all_true, all_pred = [], []
        val_loss_meter = AverageMeter()

        with torch.no_grad():
            n_val = len(dl_val)
            for i, batch in enumerate(dl_val, start=1):
                input_ids = batch.input_ids.to(device)
                attention_mask = batch.attention_mask.to(device)
                pixel_values = batch.pixel_values.to(device)
                y = batch.label.to(device).float()

                logits = model(input_ids, attention_mask, pixel_values)
                loss = criterion(logits, y)
                val_loss_meter.update(loss.item(), n=y.size(0))

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                pred = (probs >= 0.5).astype(int)

                all_pred.append(pred)
                all_true.append(y.detach().cpu().numpy().astype(int))

                if (i % 20 == 0) or (i == n_val):
                    print(f"  val: {i}/{n_val}", end="\r")
        print()

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        metrics = compute_binary_metrics(y_true, y_pred)
        score = float(metrics[best_metric_name])

        # Save history
        train_loss_hist.append(loss_meter.avg)
        val_loss_hist.append(val_loss_meter.avg)
        train_acc_hist.append(float(train_acc))
        val_acc_hist.append(float(metrics["acc"]))
        train_f1_hist.append(float(train_f1))
        val_f1_hist.append(float(metrics["f1"]))

        trainable = count_trainable_params(model)
        stage_tag = "stage2" if stage2_started else "stage1"
        print(
            f"Epoch {epoch} ({stage_tag}): "
            f"train_loss={loss_meter.avg:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} "
            f"val_loss={val_loss_meter.avg:.4f} val_acc={metrics['acc']:.4f} val_f1={metrics['f1']:.4f} "
            f"trainable={trainable}"
        )

        # Save best
        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "best_score": best_score, "config": cfg},
                os.path.join(outputs_dir, "best_model.pt"),
            )
            save_json(os.path.join(outputs_dir, "best_val_metrics.json"), metrics)
            print(f"  Saved best ({best_metric_name}={best_score:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping (patience={patience}).")
                break

    # Save curves
    history = {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "train_acc": train_acc_hist,
        "val_acc": val_acc_hist,
        "train_f1": train_f1_hist,
        "val_f1": val_f1_hist,
    }

    hist_path = os.path.join(outputs_dir, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    epochs_r = range(1, len(train_loss_hist) + 1)

    # Loss plot
    plt.figure()
    plt.plot(epochs_r, train_loss_hist, label="Train loss")
    plt.plot(epochs_r, val_loss_hist, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curves")
    plt.legend()
    plt.grid(True)
    loss_png = os.path.join(outputs_dir, "loss_curves.png")
    plt.savefig(loss_png, dpi=150, bbox_inches="tight")
    plt.close()

    # F1 plot
    plt.figure()
    plt.plot(epochs_r, train_f1_hist, label="Train F1")
    plt.plot(epochs_r, val_f1_hist, label="Val F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 Curves")
    plt.legend()
    plt.grid(True)
    f1_png = os.path.join(outputs_dir, "f1_curves.png")
    plt.savefig(f1_png, dpi=150, bbox_inches="tight")
    plt.close()

    # Acc plot
    plt.figure()
    plt.plot(epochs_r, train_acc_hist, label="Train accuracy")
    plt.plot(epochs_r, val_acc_hist, label="Val accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curves")
    plt.legend()
    plt.grid(True)
    acc_png = os.path.join(outputs_dir, "acc_curves.png")
    plt.savefig(acc_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Saved history: {hist_path}")
    print(f"Saved plots: {loss_png}, {f1_png}, {acc_png}")


if __name__ == "__main__":
    main()
