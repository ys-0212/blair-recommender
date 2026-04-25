"""Stage 2 rich document builder — constructs BLAIR-ready text per product."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def _f(val: Any, fmt: str = ".3f", default: str = "N/A") -> str:
    """Format a numeric value; return default for NaN / None."""
    if val is None:
        return default
    try:
        v = float(val)
        if np.isnan(v):
            return default
        return format(v, fmt)
    except (TypeError, ValueError):
        return default


def _s(val: Any, default: str = "N/A") -> str:
    """Return a cleaned string or default."""
    if val is None:
        return default
    if isinstance(val, float) and np.isnan(val):
        return default
    s = str(val).strip()
    return s if s else default


def _pct(val: Any, default: str = "N/A") -> str:
    """Format a [0,1] fraction as a percentage string."""
    if val is None:
        return default
    try:
        v = float(val)
        if np.isnan(v):
            return default
        return f"{v * 100:.1f}%"
    except (TypeError, ValueError):
        return default


def _section_all_na(*values: str) -> bool:
    """Return True if every provided value string is 'N/A'."""
    return all(v == "N/A" for v in values)


def _header(row: dict) -> str:
    price_str = _f(row.get("price"), ".2f")
    if price_str != "N/A":
        price_str = f"${price_str}"
    tier = _s(row.get("price_tier"))
    price_full = f"{price_str} ({tier})" if price_str != "N/A" else tier
    return (
        f"Title: {_s(row.get('title'))}\n"
        f"Brand: {_s(row.get('store'))} | Price: {price_full}\n"
        f"Category: {_s(row.get('full_category_path'))} | Leaf: {_s(row.get('leaf_category'))}\n"
        f"Market presence: {_f(row.get('days_on_market'), '.0f')} days"
        f" | {_s(row.get('rating_number'))} ratings"
    )


def _official_voice(row: dict) -> str | None:
    desc        = _s(row.get("description"))
    feats       = _s(row.get("features"))
    feat_count  = _s(row.get("feature_count"))
    richness    = _f(row.get("desc_richness_score"))
    tone        = _f(row.get("desc_sentiment"))
    if _section_all_na(desc, feats, richness, tone):
        return None
    lines = ["=== OFFICIAL PRODUCT VOICE ==="]
    if desc != "N/A":
        lines.append(f"Description: {desc}")
    if feats != "N/A":
        lines.append(f"Features: {feats} ({feat_count} features listed)")
    if richness != "N/A":
        lines.append(f"Listing quality score: {richness}")
    if tone != "N/A":
        lines.append(f"Official tone: {tone}")
    return "\n".join(lines)


def _community_voice(row: dict) -> str | None:
    rc    = _s(row.get("review_count"))
    ar    = _f(row.get("average_rating"), ".2f")
    pos   = _pct(row.get("pos_ratio"))
    neg   = _pct(row.get("neg_ratio"))
    neu   = _pct(row.get("neu_ratio"))
    cons  = _f(row.get("consensus_score"))
    cont  = _f(row.get("controversy_index"))
    vr    = _pct(row.get("verified_ratio"))
    arl   = _f(row.get("avg_review_length"), ".0f")
    hr    = _pct(row.get("helpful_ratio"))
    hws   = _f(row.get("helpfulness_weighted_sentiment"))
    if _section_all_na(ar, pos, cons):
        return None
    lines = [f"=== COMMUNITY COLLECTIVE VOICE ({rc} reviews) ==="]
    if ar != "N/A":
        lines.append(f"Overall rating: {ar}/5")
    if pos != "N/A":
        lines.append(f"Sentiment: {pos} positive | {neg} negative | {neu} neutral")
    if cons != "N/A":
        lines.append(f"Community consensus: {cons}")
    if cont != "N/A":
        lines.append(f"Controversy index: {cont}")
    if vr != "N/A":
        lines.append(f"Verified buyers: {vr}")
    if arl != "N/A":
        lines.append(f"Engagement: {arl} avg words per review")
    if hr != "N/A":
        lines.append(f"Helpfulness signal: {hr} reviews found helpful")
    if hws != "N/A":
        lines.append(f"Helpfulness-weighted sentiment: {hws}")
    return "\n".join(lines)


def _aspect_signals(row: dict) -> str | None:
    ASPECTS = ["gameplay", "graphics", "story", "controls", "value"]
    has_any = any(
        _f(row.get(f"mean_aspect_{a}")) != "N/A" for a in ASPECTS
    )
    if not has_any:
        return None
    lines = ["=== ASPECT DEEP SIGNALS ==="]
    for a in ASPECTS:
        score   = _f(row.get(f"mean_aspect_{a}"))
        cov     = _f(row.get(f"coverage_{a}"), ".1%")
        cons    = _f(row.get(f"{a}_consensus"))
        trend   = _s(row.get(f"{a}_trend"))
        if score == "N/A":
            continue
        cov_pct = _pct(row.get(f"coverage_{a}"))
        lines.append(
            f"{a.capitalize():10s} {score}"
            f" | coverage {cov_pct}"
            f" | consensus {cons}"
            f" | trend {trend}"
        )
    top_a   = _s(row.get("top_aspect"))
    worst_a = _s(row.get("worst_aspect"))
    if top_a != "N/A" or worst_a != "N/A":
        lines.append(f"Most praised: {top_a} | Most criticized: {worst_a}")
    return "\n".join(lines) if len(lines) > 1 else None


def _temporal_signals(row: dict) -> str | None:
    early_s = _f(row.get("early_sentiment"))
    mid_s   = _f(row.get("mid_sentiment"))
    recent_s= _f(row.get("recent_sentiment"))
    traj    = _s(row.get("sentiment_trajectory"))
    vel     = row.get("sentiment_velocity")
    if _section_all_na(early_s, mid_s, recent_s):
        return None
    early_p  = _s(row.get("early_period"))
    recent_p = _s(row.get("recent_period"))
    vel_str  = (f"{float(vel):+.3f}" if vel is not None and not (
        isinstance(vel, float) and np.isnan(vel)) else "N/A")
    summary  = _s(row.get("aspect_trend_summary"))
    lines = [
        "=== TEMPORAL SIGNALS ===",
        f"Early sentiment ({early_p}): {early_s}",
        f"Mid sentiment: {mid_s}",
        f"Recent sentiment ({recent_p}): {recent_s}",
        f"Trajectory: {traj}",
        f"Velocity: {vel_str}",
        f"Aspect shifts: {summary}",
    ]
    return "\n".join(lines)


def _reality_gap(row: dict) -> str | None:
    dt   = _f(row.get("desc_sentiment"))
    ct   = _f(row.get("mean_sentiment"))
    gap  = row.get("hype_gap_score")
    if _section_all_na(dt, ct):
        return None
    gap_str = (f"{float(gap):+.3f}" if gap is not None and not (
        isinstance(gap, float) and np.isnan(gap)) else "N/A")
    lines = [
        "=== REALITY GAP SIGNALS ===",
        f"Official tone: {dt} | Community tone: {ct}",
        f"Hype gap: {gap_str}",
        f"Most overhyped aspect: {_s(row.get('overhyped_aspect'))}",
        f"Most underrated aspect: {_s(row.get('underrated_aspect'))}",
        f"Hidden gem score: {_f(row.get('hidden_gem_score'))}",
    ]
    return "\n".join(lines)


def _category_signals(row: dict) -> str | None:
    leaf    = _s(row.get("leaf_category"))
    cat_sz  = _s(row.get("category_size"))
    r_pct   = row.get("category_rating_percentile")
    s_pct   = row.get("category_sentiment_percentile")
    c_ar    = _f(row.get("category_avg_rating"), ".2f")
    t_ar    = _f(row.get("average_rating"), ".2f")
    dist    = _f(row.get("distinctive_score"))
    if leaf == "Unknown" and _section_all_na(cat_sz, c_ar):
        return None
    r_top = (f"{100 - float(r_pct):.0f}%" if r_pct is not None and not (
        isinstance(r_pct, float) and np.isnan(r_pct)) else "N/A")
    s_top = (f"{100 - float(s_pct):.0f}%" if s_pct is not None and not (
        isinstance(s_pct, float) and np.isnan(s_pct)) else "N/A")
    lines = [
        f"=== CATEGORY SIGNALS ({leaf}) ===",
        f"Category size: {cat_sz} products",
        f"Rating rank: top {r_top} in category",
        f"Sentiment rank: top {s_top} in category",
        f"Category avg rating: {c_ar} | This product: {t_ar}",
        f"Outperforms category on: {_s(row.get('outperforms_aspects'))}",
        f"Underperforms category on: {_s(row.get('underperforms_aspects'))}",
        f"Distinctiveness score: {dist}",
    ]
    return "\n".join(lines)


def _vocabulary(row: dict) -> str | None:
    kws   = _s(row.get("top_tfidf_phrases"))
    uniq  = _s(row.get("distinctive_terms"))
    emo   = _s(row.get("emotion_vocabulary"))
    if _section_all_na(kws, uniq, emo):
        return None
    lines = [
        "=== COLLECTIVE VOCABULARY ===",
        f"Community keywords: {kws}",
        f"Unique to this product: {uniq}",
        f"Dominant emotions: {emo}",
    ]
    return "\n".join(lines)


_tokenizer_cache: Any = None


def _get_tokenizer(model_name: str) -> Any:
    global _tokenizer_cache
    if _tokenizer_cache is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer_cache = AutoTokenizer.from_pretrained(
                model_name, use_fast=True
            )
            logger.info("Loaded tokenizer: %s", model_name)
        except Exception as exc:
            logger.warning(
                "Could not load tokenizer '%s' (%s). "
                "Falling back to word-count approximation.", model_name, exc
            )
            _tokenizer_cache = None
    return _tokenizer_cache


def _count_tokens(text: str, tokenizer: Any) -> int:
    if tokenizer is None:
        return len(text.split())    # word-count approximation
    return len(tokenizer.encode(text, add_special_tokens=True))


def _truncate_to_budget(
    sections: list[str],
    priority: list[int],
    tokenizer: Any,
    max_tokens: int,
) -> str:
    """Greedily add sections in priority order until token budget exhausted."""
    kept = [""] * len(sections)

    # Always include core header (index 0)
    kept[0] = sections[0]
    budget  = max_tokens - _count_tokens(sections[0], tokenizer)

    for idx in priority[1:]:
        sec = sections[idx]
        if not sec:
            continue
        cost = _count_tokens(sec, tokenizer)
        if cost <= budget:
            kept[idx] = sec
            budget -= cost
        else:
            # Try to include a truncated version of this section
            words = sec.split()
            while words and _count_tokens(" ".join(words), tokenizer) > budget:
                words = words[: int(len(words) * 0.75)]
            if words and budget > 20:
                kept[idx] = " ".join(words) + " [truncated]"
                budget -= _count_tokens(kept[idx], tokenizer)
            # else: skip section entirely

    return "\n\n".join(s for s in kept if s)


def build_rich_document(row: dict | pd.Series, max_tokens: int = 512,
                        tokenizer: Any = None) -> str:
    """Build the rich text document for one product; truncates to max_tokens."""
    if isinstance(row, pd.Series):
        row = row.to_dict()

    sections = [
        _header(row),              # 0 — always kept
        _official_voice(row),      # 1
        _community_voice(row),     # 2
        _aspect_signals(row),      # 3
        _temporal_signals(row),    # 4
        _reality_gap(row),         # 5
        _category_signals(row),    # 6
        _vocabulary(row),          # 7
    ]
    # Replace None sections with ""
    sections = [s if s else "" for s in sections]

    # Priority order: indices sorted highest → lowest
    priority = [0, 1, 2, 3, 4, 5, 6, 7]

    full_text = "\n\n".join(s for s in sections if s)
    if _count_tokens(full_text, tokenizer) <= max_tokens:
        return full_text

    return _truncate_to_budget(sections, priority, tokenizer, max_tokens)


def build_rich_documents(products: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Build rich documents for all products; returns parent_asin/rich_text/token_count."""
    max_tokens = int(cfg.get("stage3", {}).get("max_seq_length", 512))
    model_name = cfg.get("stage3", {}).get("model_name", "hyp1231/blair-roberta-large")

    tokenizer = _get_tokenizer(model_name)
    logger.info("building %d rich documents (max_tokens=%d)", len(products), max_tokens)

    records: list[dict] = []
    log_every = 10_000

    for i, row in enumerate(products.itertuples(index=False)):
        row_dict = row._asdict()
        rich_text = build_rich_document(row_dict, max_tokens=max_tokens,
                                        tokenizer=tokenizer)
        tc = _count_tokens(rich_text, tokenizer)
        records.append({
            "parent_asin": row_dict["parent_asin"],
            "rich_text":   rich_text,
            "token_count": tc,
        })
        if (i + 1) % log_every == 0:
            logger.info("  %d / %d rich documents", i + 1, len(products))

    out = pd.DataFrame(records)
    tc = out["token_count"]
    n_over = int((tc > max_tokens).sum())
    logger.info(
        "Token count stats: min=%d, mean=%.0f, max=%d, "
        "over_budget=%d (%.1f%%)",
        int(tc.min()), float(tc.mean()), int(tc.max()),
        n_over, 100 * n_over / len(out),
    )
    return out
