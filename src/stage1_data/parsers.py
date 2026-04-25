"""Stage 1 parsers — low-level readers for raw Amazon Video Games files."""

from __future__ import annotations

import gzip
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _count_lines_gz(path: Path) -> int:
    """Line count for a gzip file (used for tqdm total)."""
    count = 0
    with gzip.open(path, "rb") as fh:
        for _ in fh:
            count += 1
    return count


def _iter_jsonl_gz(path: Path, fields: list[str]) -> pd.DataFrame:
    """Stream-parse a .jsonl.gz file, keeping only the specified fields."""
    records: list[dict[str, Any]] = []
    bad_lines = 0

    total = _count_lines_gz(path)
    logger.info("parsing %s (%d lines)", path.name, total)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in tqdm(fh, total=total, unit="line", desc=path.name):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            records.append({f: obj.get(f) for f in fields})

    if bad_lines:
        logger.warning("Skipped %d malformed lines in %s", bad_lines, path.name)

    return pd.DataFrame(records, columns=fields)


def parse_meta(path: Path, fields: list[str]) -> pd.DataFrame:
    """Parse meta_Video_Games.jsonl.gz; parent_asin must be in fields."""
    if "parent_asin" not in fields:
        raise ValueError("'parent_asin' must be included in meta fields.")
    df = _iter_jsonl_gz(path, fields)
    logger.info("Parsed %d raw meta records", len(df))
    return df


def parse_reviews(path: Path, fields: list[str]) -> pd.DataFrame:
    """Parse Video_Games.jsonl.gz; user_id and parent_asin must be in fields."""
    required = {"user_id", "parent_asin"}
    missing = required - set(fields)
    if missing:
        raise ValueError(f"Review fields must include: {missing}")
    df = _iter_jsonl_gz(path, fields)
    logger.info("Parsed %d raw review records", len(df))
    return df


def parse_interaction_csv(path: Path) -> pd.DataFrame:
    """Parse a pre-split interaction CSV (train/valid/test)."""
    compression = "gzip" if str(path).endswith(".gz") else None
    df = pd.read_csv(path, compression=compression, dtype=str)

    expected_cols = {"user_id", "parent_asin", "rating", "timestamp", "history"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Interaction CSV {path.name} missing columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )

    # Cast numeric columns
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    logger.info("Parsed %d rows from %s", len(df), path.name)
    return df
