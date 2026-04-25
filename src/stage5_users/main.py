"""Stage 5 — user profiles and voice documents (Colab for BLAIR encoding)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ensure_dirs, get_path, load_config
from src.stage5_users.profile_builder import build_user_profiles
from src.stage5_users.user_voice import build_user_voice_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    proc = get_path(cfg, "data_processed")

    profiles_path   = proc / "user_profiles.parquet"
    voice_docs_path = proc / "user_voice_docs.parquet"

    t0 = time.time()
    profiles = build_user_profiles(cfg)

    # Convert embedding ndarray columns to lists for parquet compatibility
    emb_cols = ["uniform_embedding", "recency_embedding",
                "rating_embedding", "combined_embedding"]
    for col in emb_cols:
        if col in profiles.columns:
            profiles[col] = profiles[col].apply(
                lambda v: v.tolist() if isinstance(v, np.ndarray) else v
            )

    profiles.to_parquet(profiles_path, index=False)
    logger.info("saved user_profiles.parquet: %d rows x %d cols  (%.1f s)",
                *profiles.shape, time.time() - t0)

    logger.info("loading reviews_nlp.parquet")
    rev_cols = [
        "user_id", "text", "timestamp", "rating",
        "sentiment_score", "sentiment_label", "helpful_vote", "verified_purchase",
        "aspect_gameplay", "aspect_graphics", "aspect_story",
        "aspect_controls", "aspect_value",
    ]
    try:
        reviews_nlp = pd.read_parquet(proc / "reviews_nlp.parquet", columns=rev_cols)
    except Exception:
        reviews_nlp = pd.read_parquet(proc / "reviews_nlp.parquet")
        reviews_nlp = reviews_nlp[[c for c in rev_cols if c in reviews_nlp.columns]]

    t1 = time.time()
    voice_docs_df = build_user_voice_documents(profiles, reviews_nlp, cfg)
    voice_docs_df.to_parquet(voice_docs_path, index=False)
    logger.info("saved user_voice_docs.parquet: %d rows x %d cols  (%.1f s)",
                *voice_docs_df.shape, time.time() - t1)

    print()
    print("user_profiles.parquet:")
    print(f"  Rows    : {len(profiles):,}")
    print(f"  Columns : {profiles.shape[1]}")
    tier_dist = profiles["cold_start_tier"].value_counts().sort_index()
    tier_names = {0: "Complete cold", 1: "Minimal", 2: "Developing", 3: "Warm"}
    print("  Cold-start tier distribution:")
    for tier, count in tier_dist.items():
        pct = 100 * count / len(profiles)
        print(f"    Tier {tier} ({tier_names.get(tier, '?'):<15}): {count:>7,}  ({pct:5.1f}%)")

    print()
    print("user_voice_docs.parquet:")
    print(f"  Rows    : {len(voice_docs_df):,}")
    print(f"  Columns : {voice_docs_df.columns.tolist()}")
    doc_lengths = voice_docs_df["voice_document"].str.len()
    print(f"  Doc length (chars): min={doc_lengths.min()}, mean={doc_lengths.mean():.0f}, max={doc_lengths.max()}")
    tier2_plus = int((profiles["cold_start_tier"] >= 2).sum())
    print(f"  Tier-2+ users (eligible for encoding): {tier2_plus:,}")

    print()
    print("Stage 5 local work complete.")
    print("Next: run notebooks/stage5_voice_encoding_colab.ipynb to encode voice docs.")


if __name__ == "__main__":
    run()
