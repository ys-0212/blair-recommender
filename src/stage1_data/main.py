"""Stage 1 — parse raw files, 5-core filter, write parquet outputs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.utils.config import ensure_dirs, get_path, load_config
from src.stage1_data.parsers import parse_meta, parse_reviews, parse_interaction_csv
from src.stage1_data.cleaner import (
    apply_k_core,
    clean_meta,
    clean_reviews,
    derive_train_from_reviews,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _write_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    """Write a DataFrame to a snappy-compressed parquet file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    size_mb = path.stat().st_size / 1_048_576
    logger.info("Wrote %s -> %s  (%.1f MB, %d rows)", label, path, size_mb, len(df))


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    # -- metadata --
    raw_meta_path = get_path(cfg, "raw_meta")
    meta_fields = cfg["stage1"]["meta_fields"]

    meta_raw = parse_meta(raw_meta_path, meta_fields)
    meta_clean = clean_meta(meta_raw)

    meta_out = get_path(cfg, "meta_clean")
    _write_parquet(meta_clean, meta_out, "meta_clean")

    # -- reviews + 5-core --
    raw_reviews_path = get_path(cfg, "raw_reviews")
    review_fields = cfg["stage1"]["review_fields"]

    reviews_raw = parse_reviews(raw_reviews_path, review_fields)
    reviews_clean = clean_reviews(reviews_raw)

    # Restrict reviews to items that exist in cleaned meta
    valid_items = set(meta_clean["parent_asin"])
    before = len(reviews_clean)
    reviews_clean = reviews_clean[reviews_clean["parent_asin"].isin(valid_items)]
    logger.info(
        "Dropped %d reviews for items not in meta → %d remaining",
        before - len(reviews_clean), len(reviews_clean),
    )

    # 5-core filtering
    min_k = cfg["stage1"]["min_reviews_per_user"]  # same value used for items
    reviews_top5 = apply_k_core(
        reviews_clean,
        k=min_k,
        user_col="user_id",
        item_col="parent_asin",
    )

    reviews_out = get_path(cfg, "reviews_top5")
    _write_parquet(reviews_top5, reviews_out, "reviews_top5")

    # -- valid/test splits --
    valid_in = get_path(cfg, "raw_valid")
    test_in = get_path(cfg, "raw_test")

    valid_df = parse_interaction_csv(valid_in)
    test_df = parse_interaction_csv(test_in)

    # Restrict to users / items that survived 5-core filtering
    core_users = set(reviews_top5["user_id"])
    core_items = set(reviews_top5["parent_asin"])

    for split_name, split_df in [("valid", valid_df), ("test", test_df)]:
        before = len(split_df)
        split_df = split_df[
            split_df["user_id"].isin(core_users) &
            split_df["parent_asin"].isin(core_items)
        ]
        logger.info(
            "Split %s: %d → %d rows after restricting to 5-core users/items",
            split_name, before, len(split_df),
        )
        if split_name == "valid":
            valid_df = split_df
        else:
            test_df = split_df

    _write_parquet(valid_df, get_path(cfg, "valid"), "valid")
    _write_parquet(test_df, get_path(cfg, "test"), "test")

    # -- training split --
    raw_train_path = get_path(cfg, "raw_train")

    if raw_train_path.exists():
        logger.info("Found train CSV at %s — parsing ...", raw_train_path)
        try:
            train_df = parse_interaction_csv(raw_train_path)
            train_df = train_df[
                train_df["user_id"].isin(core_users) &
                train_df["parent_asin"].isin(core_items)
            ]
            logger.info("Loaded %d training interactions from CSV", len(train_df))
        except Exception as exc:
            logger.warning(
                "Failed to parse train CSV (%s) — falling back to derivation: %s",
                raw_train_path.name, exc,
            )
            train_df = derive_train_from_reviews(reviews_top5, valid_df, test_df)
    else:
        # Common case: the .crdownload file is incomplete
        logger.warning(
            "Train CSV not found at %s — deriving from reviews_top5.",
            raw_train_path,
        )
        # Check for the incomplete download to give a helpful message
        crdownload = raw_train_path.parent / (raw_train_path.name + ".crdownload")
        if crdownload.exists():
            logger.warning(
                "Incomplete download found: %s  "
                "Re-download Video_Games.train.csv.gz to use the official split.",
                crdownload.name,
            )
        train_df = derive_train_from_reviews(reviews_top5, valid_df, test_df)

    _write_parquet(train_df, get_path(cfg, "train"), "train")

    logger.info("stage1 done:")
    logger.info("  meta items   : %d", len(meta_clean))
    logger.info("  Reviews      : %d  (%d users, %d items)",
                len(reviews_top5),
                reviews_top5["user_id"].nunique(),
                reviews_top5["parent_asin"].nunique())
    logger.info("  Train        : %d interactions", len(train_df))
    logger.info("  Valid        : %d interactions", len(valid_df))
    logger.info("  Test         : %d interactions", len(test_df))


if __name__ == "__main__":
    run()
