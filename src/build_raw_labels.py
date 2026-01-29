
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, Iterable, List

import pandas as pd
import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def iter_json_lines(path: str) -> Iterable[Dict[str, Any]]:
    # Read JSONL
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def choose_review_per_business(reviews_df: pd.DataFrame, strategy: str, seed: int) -> pd.DataFrame:
    # Pick 1 review per business
    if strategy not in {"random", "max_votes", "longest"}:
        raise ValueError(f"Unknown strategy: {strategy}")

    df = reviews_df.copy()
    df["votes_sum"] = df["useful"].fillna(0) + df["funny"].fillna(0) + df["cool"].fillna(0)
    df["text_len"] = df["text"].fillna("").astype(str).str.len()

    if strategy == "random":
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        chosen = df.groupby("business_id", as_index=False).first()
        return chosen.drop(columns=["votes_sum", "text_len"])

    if strategy == "max_votes":
        df = df.sort_values(["business_id", "votes_sum", "text_len"], ascending=[True, False, False])
        chosen = df.groupby("business_id", as_index=False).first()
        return chosen.drop(columns=["votes_sum", "text_len"])

    df = df.sort_values(["business_id", "text_len", "votes_sum"], ascending=[True, False, False])
    chosen = df.groupby("business_id", as_index=False).first()
    return chosen.drop(columns=["votes_sum", "text_len"])


def main() -> None:
    ap = argparse.ArgumentParser(description="Build labels.csv from Yelp reviews + photos")
    ap.add_argument("--config", required=True, help="Experiment YAML")
    ap.add_argument(
        "--review_strategy",
        default="max_votes",
        choices=["random", "max_votes", "longest"],
        help="How to choose the review",
    )
    ap.add_argument("--max_photos", type=int, default=0, help="Cap rows (0 = no cap)")
    ap.add_argument("--verify_images", action="store_true", help="Check image files exist")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    seed = int(cfg["project"]["seed"])

    raw_csv = cfg["paths"]["raw_labels_csv"]
    images_dir = cfg["paths"]["images_dir"]

    # Input files
    review_json = os.path.join("data", "Yelp JSON", "yelp_academic_dataset_review.json")
    photos_json = os.path.join("data", "Yelp Photos", "photos.json")

    if not os.path.exists(review_json):
        raise FileNotFoundError(f"Missing review JSON: {review_json}")
    if not os.path.exists(photos_json):
        raise FileNotFoundError(f"Missing photos JSON: {photos_json}")

    # Load reviews
    reviews_rows: List[Dict[str, Any]] = []
    for r in iter_json_lines(review_json):
        reviews_rows.append(
            {
                "review_id": r.get("review_id"),
                "business_id": r.get("business_id"),
                "text": r.get("text", "") or "",
                "useful": int(r.get("useful", 0) or 0),
                "funny": int(r.get("funny", 0) or 0),
                "cool": int(r.get("cool", 0) or 0),
            }
        )

    reviews_df = pd.DataFrame(reviews_rows).dropna(subset=["business_id"]).reset_index(drop=True)
    if len(reviews_df) == 0:
        raise RuntimeError("No reviews loaded.")

    chosen_reviews = choose_review_per_business(reviews_df, strategy=args.review_strategy, seed=seed)

    # Load photos
    photos_rows: List[Dict[str, Any]] = []
    for p in iter_json_lines(photos_json):
        photo_id = p.get("photo_id")
        business_id = p.get("business_id")
        if not photo_id or not business_id:
            continue
        photos_rows.append(
            {
                "photo_id": photo_id,
                "business_id": business_id,
                "photo_label": p.get("label", None),
            }
        )

    photos_df = pd.DataFrame(photos_rows).dropna(subset=["business_id", "photo_id"]).reset_index(drop=True)
    if len(photos_df) == 0:
        raise RuntimeError("No photos loaded.")

    # Join by business_id
    merged = photos_df.merge(chosen_reviews, on="business_id", how="inner")

    # Build image path
    merged["image_path"] = merged["photo_id"].astype(str) + ".jpg"

    # Drop missing images 
    if args.verify_images:
        full_paths = merged["image_path"].apply(lambda p: os.path.join(images_dir, p))
        exists_mask = full_paths.apply(os.path.exists)
        missing = int((~exists_mask).sum())
        if missing > 0:
            print(f"[WARN] Missing images: {missing}. Dropping them.")
        merged = merged[exists_mask].reset_index(drop=True)

    # Cap rows 
    if args.max_photos and args.max_photos > 0 and len(merged) > args.max_photos:
        merged = merged.sample(n=args.max_photos, random_state=seed).reset_index(drop=True)

    # Save
    out = merged[
        [
            "business_id",
            "review_id",
            "photo_id",
            "image_path",
            "text",
            "useful",
            "funny",
            "cool",
            "photo_label",
        ]
    ].copy()

    ensure_dir(os.path.dirname(raw_csv))
    out.to_csv(raw_csv, index=False)

    print(f"Saved: {raw_csv}")
    print(f"Rows: {len(out)}")
    print("Unique businesses:", out["business_id"].nunique())
    if len(out) > 0:
        print("Example row:", out.iloc[0].to_dict())


if __name__ == "__main__":
    main()
