"""Stage 2 — aspect-based sentiment (gameplay/graphics/story/controls/value)."""

from __future__ import annotations

import logging
import re
from typing import Any

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

_sia: SentimentIntensityAnalyzer | None = None


def _get_sia() -> SentimentIntensityAnalyzer:
    global _sia
    if _sia is None:
        _sia = SentimentIntensityAnalyzer()
    return _sia


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using punctuation heuristics (no NLTK)."""
    if not text:
        return []
    sentences = _SENT_SPLIT_RE.split(text.strip())
    # Remove very short fragments (< 3 chars) that carry no signal
    return [s for s in sentences if len(s) >= 3]


def _build_patterns(keywords: list[str]) -> re.Pattern:
    """Compile OR-pattern with word boundaries (avoids partial matches)."""
    escaped = [re.escape(kw) for kw in keywords]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def _aspect_score_for_text(text: str, pattern: re.Pattern) -> float:
    """Mean VADER compound for pattern-matching sentences; NaN if none match."""
    sentences = _split_sentences(text)
    matched_scores: list[float] = []
    sia = _get_sia()

    for sent in sentences:
        if pattern.search(sent):
            score = sia.polarity_scores(sent)["compound"]
            matched_scores.append(score)

    if not matched_scores:
        return float("nan")
    return float(np.mean(matched_scores))


def add_aspect_sentiment(
    df: pd.DataFrame,
    aspect_keywords: dict[str, list[str]],
    batch_log_every: int = 100_000,
) -> pd.DataFrame:
    """Add aspect_<name> columns (float32) for each aspect in aspect_keywords."""
    df = df.copy()
    texts = df["text"].fillna("").tolist()
    n = len(texts)

    # Pre-compile one regex pattern per aspect
    patterns = {
        aspect: _build_patterns(keywords)
        for aspect, keywords in aspect_keywords.items()
    }

    # Initialise score arrays as NaN
    scores: dict[str, np.ndarray] = {
        aspect: np.full(n, float("nan"), dtype=np.float32)
        for aspect in aspect_keywords
    }

    logger.info("aspect sentiment: %d reviews x %d aspects", n, len(patterns))

    for i, text in enumerate(texts):
        if i > 0 and i % batch_log_every == 0:
            logger.info("aspect sentiment: %d / %d", i, n)
        for aspect, pattern in patterns.items():
            scores[aspect][i] = _aspect_score_for_text(text, pattern)

    # attach columns and log coverage
    for aspect, arr in scores.items():
        col = f"aspect_{aspect}"
        df[col] = arr
        n_matched = int(np.sum(~np.isnan(arr)))
        logger.info(
            "  %s: %d / %d reviews had matching sentences (%.1f%%)",
            col, n_matched, n, 100 * n_matched / n if n else 0,
        )

    return df
