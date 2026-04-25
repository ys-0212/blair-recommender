"""Stage 2 — VADER sentiment, aspect scoring, product aggregation, rich docs."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.utils.config import ensure_dirs, get_path, load_config
from src.stage2_nlp.sentiment import add_review_sentiment
from src.stage2_nlp.aspects import add_aspect_sentiment
from src.stage2_nlp.aggregator import aggregate_to_products
from src.stage2_nlp.rich_document import build_rich_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _write_parquet(df: pd.DataFrame, path: Path, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, compression="snappy")
    size_mb = path.stat().st_size / 1_048_576
    logger.info("Wrote %s -> %s  (%.1f MB, %d rows)", label, path.name, size_mb, len(df))


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    reviews_nlp_path = get_path(cfg, "reviews_nlp")

    # skip the 30-min VADER pass if already cached
    if reviews_nlp_path.exists():
        reviews = pd.read_parquet(reviews_nlp_path)
        logger.info("loaded cached reviews_nlp: %d rows", len(reviews))
    else:
        reviews_top5_path = get_path(cfg, "reviews_top5")
        if not reviews_top5_path.exists():
            raise FileNotFoundError(
                f"reviews_top5.parquet not found at {reviews_top5_path}. "
                "Run Stage 1 first: python -m src.stage1_data.main"
            )
        reviews = pd.read_parquet(reviews_top5_path)
        logger.info("loaded %d reviews", len(reviews))
        reviews = add_review_sentiment(reviews)
        aspect_keywords = cfg["stage2"]["aspect_keywords"]
        reviews = add_aspect_sentiment(reviews, aspect_keywords)
        _write_parquet(reviews, reviews_nlp_path, "reviews_nlp")

    # -- product aggregation --
    meta_path = get_path(cfg, "meta_clean")
    meta = pd.read_parquet(meta_path)
    logger.info("loaded %d meta records", len(meta))

    products_nlp = aggregate_to_products(reviews, meta, cfg=cfg)

    processed_dir = get_path(cfg, "data_processed")
    _write_parquet(products_nlp, processed_dir / "products_nlp.parquet", "products_nlp")

    products_rich = build_rich_documents(products_nlp, cfg)
    _write_parquet(products_rich, processed_dir / "products_rich.parquet", "products_rich")

    logger.info("stage2 done:")
    logger.info("  reviews NLP rows  : %d", len(reviews))
    logger.info("  Products NLP rows : %d  (%d columns)", len(products_nlp), len(products_nlp.columns))
    logger.info("  Products rich rows: %d", len(products_rich))
    logger.info(
        "  Token counts      : min=%d  mean=%.0f  max=%d",
        int(products_rich["token_count"].min()),
        float(products_rich["token_count"].mean()),
        int(products_rich["token_count"].max()),
    )


if __name__ == "__main__":
    run()
