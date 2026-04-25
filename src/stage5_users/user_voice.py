"""Stage 5 — structured user voice documents and BLAIR encoding."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import get_path, load_config

logger = logging.getLogger(__name__)

_TOKENIZER_CACHE: dict[str, Any] = {}


def _get_tokenizer(model_name: str = "hyp1231/blair-roberta-large") -> Any | None:
    """Return a cached tokenizer, or None if transformers is unavailable."""
    global _TOKENIZER_CACHE
    if model_name in _TOKENIZER_CACHE:
        return _TOKENIZER_CACHE[model_name]
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        _TOKENIZER_CACHE[model_name] = tok
        logger.info("Loaded tokenizer: %s", model_name)
        return tok
    except Exception as exc:
        logger.warning("Could not load tokenizer (%s) — using word-count fallback: %s", model_name, exc)
        _TOKENIZER_CACHE[model_name] = None
        return None


def _token_count(text: str, tokenizer: Any | None) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.encode(text, add_special_tokens=True))


def _truncate_to_budget(text: str, budget: int, tokenizer: Any | None) -> str:
    """Truncate text so it fits within token budget (approx)."""
    if _token_count(text, tokenizer) <= budget:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if _token_count(" ".join(words[:mid]), tokenizer) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return " ".join(words[:lo])


def _fmt(val: Any, fmt: str = ".3f", fallback: str = "n/a") -> str:
    """Format a value safely; return fallback for NaN/None."""
    if val is None:
        return fallback
    try:
        if math.isnan(float(val)):
            return fallback
    except (TypeError, ValueError):
        pass
    if fmt:
        try:
            return format(float(val), fmt)
        except (TypeError, ValueError):
            return str(val)
    return str(val)


def _critic_style(avg_rating: float) -> str:
    """Map average rating to harsh/balanced/generous label."""
    if avg_rating is None or math.isnan(avg_rating):
        return "balanced"
    if avg_rating <= 2.5:
        return "harsh"
    if avg_rating >= 4.0:
        return "generous"
    return "balanced"


def _section_interaction(row: pd.Series) -> str:
    lines = [
        "=== INTERACTION HISTORY ===",
        f"Total interactions: {int(row.get('interaction_count', 0))}",
        f"Active for: {_fmt(row.get('active_days'), '.0f')} days",
        f"Engagement intensity: {_fmt(row.get('interaction_velocity'), '.3f')} interactions/day",
        f"Rating pattern: avg {_fmt(row.get('avg_rating_given'), '.1f')}/5"
        f" (std {_fmt(row.get('rating_std'), '.2f')})",
        f"Critic style: {_critic_style(row.get('avg_rating_given', float('nan')))}",
    ]
    return "\n".join(lines)


def _section_categories(row: pd.Series) -> str:
    top_cats = row.get("top_categories", [])
    if not isinstance(top_cats, list) or not top_cats:
        top_cats_str = "n/a"
    else:
        top_cats_str = ", ".join(str(c) for c in top_cats[:3])
    lines = [
        "=== CATEGORY PREFERENCES ===",
        f"Favorite categories: {top_cats_str}",
        f"Taste diversity: {_fmt(row.get('category_diversity'), '.3f')}",
        f"Category entropy: {_fmt(row.get('category_entropy'), '.3f')}",
        f"Dominant genre: {row.get('dominant_category', 'n/a')}",
    ]
    return "\n".join(lines)


def _section_price(row: pd.Series) -> str:
    pref = row.get("preferred_price_tier", "n/a") or "n/a"
    lines = [
        "=== PRICE PREFERENCES ===",
        f"Preferred tier: {pref}",
        f"Price flexibility: {_fmt(row.get('price_diversity'), '.3f')}",
    ]
    return "\n".join(lines)


def _section_aspects(row: pd.Series) -> str:
    asp_names = ["gameplay", "graphics", "story", "controls", "value"]
    lines = ["=== ASPECT PREFERENCES ==="]
    for asp in asp_names:
        val = row.get(f"user_aspect_{asp}", float("nan"))
        lines.append(f"{asp.capitalize()} importance: {_fmt(val, '.3f')}")
    lines.append(f"Cares most about: {row.get('user_top_aspect', 'n/a') or 'n/a'}")
    lines.append(f"Least cares about: {row.get('user_worst_aspect', 'n/a') or 'n/a'}")
    return "\n".join(lines)


def _section_review_behavior(row: pd.Series) -> str:
    pos_pct = row.get("user_pos_ratio", float("nan"))
    try:
        pos_str = f"{float(pos_pct)*100:.1f}%"
    except (TypeError, ValueError):
        pos_str = "n/a"
    lines = [
        "=== REVIEW BEHAVIOR ===",
        f"Reviews written: {int(row.get('user_review_count', 0))}",
        f"Avg review length: {_fmt(row.get('user_avg_review_length'), '.0f')} words",
        f"Sentiment tendency: {_fmt(row.get('user_avg_sentiment'), '.3f')}",
        f"Positivity ratio: {pos_str}",
        f"Helpful votes received: {int(row.get('user_helpful_votes_received', 0) or 0)}",
    ]
    return "\n".join(lines)


def build_user_voice_document(
    user_row: pd.Series,
    user_reviews_df: pd.DataFrame,
    max_tokens: int = 512,
    tokenizer: Any | None = None,
) -> str:
    """Build a structured taste document; fills token budget with review text."""
    header = f"User taste profile:\n"

    # Priority-ordered sections (first = highest priority)
    sections = [
        _section_interaction(user_row),
        _section_categories(user_row),
        _section_aspects(user_row),
        _section_price(user_row),
        _section_review_behavior(user_row),
    ]

    # Build base doc from sections
    doc_parts = [header]
    for sec in sections:
        doc_parts.append(sec)

    base_doc = "\n\n".join(doc_parts)
    base_tokens = _token_count(base_doc, tokenizer)
    remaining = max_tokens - base_tokens - 10  # 10 token buffer

    # Append user's own words within remaining budget
    if remaining > 20 and not user_reviews_df.empty and "text" in user_reviews_df.columns:
        # Sort: most recent first, then highest rated
        sort_cols = [c for c in ["timestamp", "rating"] if c in user_reviews_df.columns]
        if sort_cols:
            user_reviews_sorted = user_reviews_df.sort_values(
                sort_cols, ascending=[False] * len(sort_cols)
            )
        else:
            user_reviews_sorted = user_reviews_df

        words_section = "=== USER'S OWN WORDS ==="
        collected_texts = []
        budget_left = remaining - _token_count(words_section, tokenizer) - 2

        for _, rev_row in user_reviews_sorted.iterrows():
            txt = str(rev_row.get("text", "") or "").strip()
            if not txt:
                continue
            tc = _token_count(txt, tokenizer)
            if tc <= budget_left:
                collected_texts.append(txt)
                budget_left -= tc + 1  # +1 for separator
            else:
                # Try truncating the last review to fit
                truncated = _truncate_to_budget(txt, budget_left, tokenizer)
                if truncated:
                    collected_texts.append(truncated)
                break

        if collected_texts:
            own_words = words_section + "\n" + " ".join(collected_texts)
            base_doc = base_doc + "\n\n" + own_words

    return base_doc


def build_user_voice_documents(
    profiles: pd.DataFrame,
    reviews_nlp: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Build voice documents for all tier-2+ users.

    Returns
    -------
    DataFrame[user_id, voice_document, tier]
    """
    model_name = cfg.get("stage3", {}).get("model_name", "hyp1231/blair-roberta-large")
    max_tokens = int(cfg.get("stage3", {}).get("max_seq_length", 512))

    logger.info("Loading tokenizer for voice document building ...")
    tokenizer = _get_tokenizer(model_name)

    # Only build for tier 2+ (tier 1 has too little signal for meaningful encoding)
    eligible = profiles[profiles["cold_start_tier"] >= 2].copy()
    logger.info(
        "Building voice documents for %d tier-2+ users (out of %d total) ...",
        len(eligible), len(profiles),
    )

    # Pre-group reviews by user
    if "user_id" in reviews_nlp.columns:
        rev_grouped = reviews_nlp.groupby("user_id", sort=False)
        rev_groups = {uid: grp for uid, grp in rev_grouped}
    else:
        rev_groups = {}

    docs = []
    for i, (_, row) in enumerate(eligible.iterrows()):
        uid = row["user_id"]
        user_revs = rev_groups.get(uid, pd.DataFrame())
        doc = build_user_voice_document(row, user_revs, max_tokens, tokenizer)
        docs.append({"user_id": uid, "voice_document": doc, "tier": int(row["cold_start_tier"])})

        if (i + 1) % 5000 == 0:
            logger.info("  ... %d / %d voice documents built", i + 1, len(eligible))

    logger.info("Voice document building complete: %d documents", len(docs))
    return pd.DataFrame(docs)


# ---------------------------------------------------------------------------
# BLAIR encoding
# ---------------------------------------------------------------------------

def encode_user_voices(
    voice_docs_df: pd.DataFrame,
    cfg: dict,
    output_dir: Path | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode user voice documents through hyp1231/blair-roberta-large.

    Only encodes tier-2+ users (passed in via voice_docs_df).
    Runs on CPU with a small batch size to manage RoBERTa-large memory.

    Saves:
      data/processed/user_voice_embeddings.npy
      data/processed/user_voice_ids.npy

    Returns
    -------
    embeddings : float32 array of shape (N, 1024), L2-normalised
    user_ids   : list of user_id strings in same order
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    if cfg is None:
        cfg = load_config()

    if output_dir is None:
        output_dir = get_path(cfg, "data_processed")

    model_name = cfg.get("stage3", {}).get("model_name", "hyp1231/blair-roberta-large")
    max_seq_len = int(cfg.get("stage3", {}).get("max_seq_length", 512))
    batch_size  = 8   # CPU-safe for RoBERTa-large (768M params)

    logger.info("Loading BLAIR model for user voice encoding: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device("cpu")
    model = model.to(device)
    logger.info("Model on CPU. Encoding %d user voices ...", len(voice_docs_df))

    docs     = voice_docs_df["voice_document"].tolist()
    user_ids = voice_docs_df["user_id"].tolist()
    n        = len(docs)
    emb_list = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch_docs = docs[start: start + batch_size]
            enc = tokenizer(
                batch_docs,
                padding=True,
                truncation=True,
                max_length=max_seq_len,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            cls_emb = out.last_hidden_state[:, 0, :].cpu().numpy().astype(np.float32)
            # L2 normalise each vector
            norms = np.linalg.norm(cls_emb, axis=1, keepdims=True)
            norms = np.where(norms < 1e-9, 1.0, norms)
            cls_emb = cls_emb / norms
            emb_list.append(cls_emb)

            if (start // batch_size + 1) % 100 == 0:
                logger.info("  ... encoded %d / %d", start + len(batch_docs), n)

    embeddings = np.concatenate(emb_list, axis=0)
    logger.info(
        "User voice encoding complete: shape=%s  norms min=%.4f max=%.4f",
        embeddings.shape,
        np.linalg.norm(embeddings, axis=1).min(),
        np.linalg.norm(embeddings, axis=1).max(),
    )

    # Save
    emb_path = output_dir / "user_voice_embeddings.npy"
    ids_path = output_dir / "user_voice_ids.npy"
    np.save(str(emb_path), embeddings)
    np.save(str(ids_path), np.array(user_ids, dtype=object))
    logger.info("Saved user_voice_embeddings.npy (%d users, dim=%d)", len(user_ids), embeddings.shape[1])
    logger.info("Saved user_voice_ids.npy")

    return embeddings, user_ids
