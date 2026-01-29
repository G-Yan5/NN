User: complete this README.md file in brief

Workspace: Collecting workspace information```markdown
# Yelp Multimodal Review Credibility Classification

A PyTorch-based multimodal deep learning system for predicting review credibility using text and images from Yelp reviews.

## Overview

This project combines RoBERTa (text encoder) and EfficientNet (image encoder) to classify whether a review is credible based on:
- Review text
- Associated business photos
- Review metadata (useful votes, funny, cool)

## Project Structure

```
src/                          # Main pipeline
├── build_raw_labels.py       # Build raw labels from Yelp JSON
├── preprocess.py             # Train/val/test split & labeling
├── data.py                   # Dataset and DataLoader
├── models.py                 # RoBERTa + EfficientNet + Fusion
├── train.py                  # Two-stage training (head → fine-tune)
├── eval.py                   # Evaluation & threshold tuning
└── utils.py                  # Helpers (metrics, seed, device)

configs/
└── yelp_roberta_efficientnet.yaml  # Training configuration


## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Build Raw Labels
```bash
python -m src.build_raw_labels --config configs/yelp_roberta_efficientnet.yaml
```

### 3. Preprocess Data
```bash
python -m src.preprocess --config configs/yelp_roberta_efficientnet.yaml
```

### 4. Train Model
```bash
python -m src.train --config configs/yelp_roberta_efficientnet.yaml
```

### 5. Evaluate
```bash
python -m src.eval --config configs/yelp_roberta_efficientnet.yaml --split test --use_tuned_threshold
```

## Model Architecture

- **Text Encoder**: RoBERTa-base (768-dim)
- **Image Encoder**: EfficientNet-B0 (1280-dim)
- **Fusion Head**: 2-layer MLP (512 hidden)
- **Output**: Binary classification (credible/non-credible)

## Training Strategy

**Stage 1** (warmup): Freeze encoders, train head only  
**Stage 2** (fine-tune): Unfreeze last N layers of RoBERTa + EfficientNet, use differential LR

## Configuration

Edit yelp_roberta_efficientnet.yaml to adjust:
- Model names & hyperparameters
- Training epochs, batch size, learning rates
- Data paths
- Split ratios

## Outputs

- best_model.pt – Best checkpoint, can be infer from (https://github.com/G-Yan5/NN/releases/tag/BestModel)
- metrics_val.json, `metrics_test.json` – Performance metrics
- training_history.json – Loss/F1/Acc curves
- loss_curves.png, `f1_curves.png`, `acc_curves.png` – Training plots
- `outputs/confusion_*.png` – Confusion matrices

## Requirements

See requirements.txt for dependencies (PyTorch, Transformers, scikit-learn, etc.).

## Notes

- Uses `BCEWithLogitsLoss` with class-weighted `pos_weight`
- Supports mixed precision (AMP) on CUDA
- Early stopping on validation F1 score
- Reproducible: fixed seed for all random operations

