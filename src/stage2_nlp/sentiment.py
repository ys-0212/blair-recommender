"""Stage 2 — VADER full-text sentiment scoring."""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

# Singleton analyser — initialisation loads the VADER lexicon once
_sia: SentimentIntensityAnalyzer | None = None


def _get_sia() -> SentimentIntensityAnalyzer:
    global _sia
    if _sia is None:
        _sia = SentimentIntensityAnalyzer()
    return _sia


# VADER compound thresholds (standard cutpoints from the original paper)
_POS_THRESHOLD = 0.05
_NEG_THRESHOLD = -0.05


def compound_to_label(score: float) -> str:
    """Map a VADER compound score to a human-readable label."""
    if score >= _POS_THRESHOLD:
        return "positive"
    if score <= _NEG_THRESHOLD:
        return "negative"
    return "neutral"


def score_texts(
    texts: Sequence[str],
    batch_log_every: int = 100_000,
) -> tuple[np.ndarray, list[str]]:
    """Score texts with VADER; returns (float32 compound array, label list)."""
    sia = _get_sia()
    n = len(texts)
    scores = np.empty(n, dtype=np.float32)
    labels: list[str] = []

    for i, text in enumerate(texts):
        if i > 0 and i % batch_log_every == 0:
            logger.info("VADER sentiment: %d / %d done", i, n)
        t = str(text) if text else ""
        compound = sia.polarity_scores(t)["compound"]
        scores[i] = compound
        labels.append(compound_to_label(compound))

    logger.info("VADER sentiment: %d / %d done (complete)", n, n)
    return scores, labels


def add_review_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment_score and sentiment_label columns from the 'text' column."""
    df = df.copy()
    texts = df["text"].fillna("").tolist()
    scores, labels = score_texts(texts)
    df["sentiment_score"] = scores
    df["sentiment_label"] = labels
    logger.info(
        "Sentiment distribution — positive: %d, neutral: %d, negative: %d",
        labels.count("positive"),
        labels.count("neutral"),
        labels.count("negative"),
    )
    return df
