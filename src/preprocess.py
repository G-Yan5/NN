from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

from .utils import ensure_dir, set_seed


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_label_and_filter(
    df: pd.DataFrame,
    useful_col: str,
    funny_col: str,
    cool_col: str,
    pos_votes: int = 3,
) -> pd.DataFrame:
    # Label by votes, drop ambiguous
    votes = df[useful_col].fillna(0) + df[funny_col].fillna(0) + df[cool_col].fillna(0)
    votes = votes.astype(int)

    keep = (votes == 0) | (votes >= pos_votes)
    df = df.loc[keep].copy()
    df["label"] = (votes.loc[keep] >= pos_votes).astype(int)

    return df


def filter_min_photos_per_business(df: pd.DataFrame, business_col: str, min_photos: int) -> pd.DataFrame:
    # Keep businesses with enough photos
    counts = df.groupby(business_col, dropna=False).size().rename("photo_count")
    df = df.join(counts, on=business_col)
    return df[df["photo_count"] >= min_photos].drop(columns=["photo_count"])


def split_by_business_id(
    df: pd.DataFrame,
    business_col: str,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> pd.DataFrame:
    # Split by business_id (no leakage)
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "splits must sum to 1"

    gss1 = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    idx_train, idx_temp = next(gss1.split(df, groups=df[business_col]))
    df_train = df.iloc[idx_train].copy()
    df_temp = df.iloc[idx_temp].copy()

    temp_total = val_frac + test_frac
    val_within_temp = val_frac / temp_total

    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_within_temp, random_state=seed + 1)
    idx_val, idx_test = next(gss2.split(df_temp, groups=df_temp[business_col]))
    df_val = df_temp.iloc[idx_val].copy()
    df_test = df_temp.iloc[idx_test].copy()

    df_train["split"] = "train"
    df_val["split"] = "val"
    df_test["split"] = "test"

    return pd.concat([df_train, df_val, df_test], axis=0).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    raw_csv = cfg["paths"]["raw_labels_csv"]
    processed_csv = cfg["paths"]["processed_csv"]
    ensure_dir(os.path.dirname(processed_csv))

    cols = cfg["preprocess"]["columns"]
    business_col = cols["business_id"]
    image_col = cols["image_path"]
    text_col = cols["text"]
    useful_col = cols["useful"]
    funny_col = cols["funny"]
    cool_col = cols["cool"]

    df = pd.read_csv(raw_csv)

    # Select columns
    keep = [c for c in [cols.get("review_id"), business_col, image_col, text_col, useful_col, funny_col, cool_col] if c]
    keep = [c for c in keep if c in df.columns]
    df = df[keep].copy()

    # Clean types
    df[text_col] = df[text_col].fillna("").astype(str)
    df[image_col] = df[image_col].astype(str)

    # Build labels
    df = make_label_and_filter(df, useful_col, funny_col, cool_col, pos_votes=3)

    # Filter businesses
    min_photos = int(cfg["preprocess"]["min_photos_per_business"])
    df = filter_min_photos_per_business(df, business_col, min_photos)

    df = df.dropna(subset=[business_col, image_col]).reset_index(drop=True)

    # Split data
    split_cfg = cfg["preprocess"]["split"]
    df = split_by_business_id(
        df=df,
        business_col=business_col,
        train_frac=float(split_cfg["train"]),
        val_frac=float(split_cfg["val"]),
        test_frac=float(split_cfg["test"]),
        seed=seed,
    )

    df.to_csv(processed_csv, index=False)
    print(f"Saved processed CSV: {processed_csv}")
    print(df["split"].value_counts().to_dict())
    print("Label distribution:", df["label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
