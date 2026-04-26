"""Stage 5 — blends query and user embeddings based on cold-start tier."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _l2_norm(v: np.ndarray) -> np.ndarray:
    """L2-normalise; return as-is for zero vector."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _safe_emb(val: Any, dim: int) -> np.ndarray | None:
    """Extract numpy array from profile value (may be ndarray, list, or None)."""
    if val is None:
        return None
    try:
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == dim:
            return arr
    except (TypeError, ValueError):
        pass
    return None


def blend_query(
    query_emb: np.ndarray,
    user_profile: pd.Series | dict,
    user_voice_emb: np.ndarray | None,
    cfg: dict,
) -> np.ndarray:
    """Blend a BLAIR-encoded query with the user's profile embedding.

    Returns an L2-normalised float32 array of shape (dim,) ready for FAISS search.
    Blending strategy depends on cold_start_tier: 0=pure query, 1=uniform, 2=recency,
    3=three-way (query + combined + voice) using weights from stage5 config.
    """
    query_emb = _l2_norm(np.asarray(query_emb, dtype=np.float32))
    dim = query_emb.shape[0]

    cfg5 = cfg.get("stage5", {})
    # Per-tier fallback weights
    t12_query_w = float(cfg5.get("query_weight", 0.7))
    t12_user_w  = float(cfg5.get("user_weight",  0.3))
    # Tier-3 three-way blend weights
    t3_q_w = float(cfg5.get("tier3_query_weight",    0.5))
    t3_c_w = float(cfg5.get("tier3_combined_weight", 0.3))
    t3_v_w = float(cfg5.get("tier3_voice_weight",    0.2))
    # Tier-3 fallback (no voice)
    t3_q_nv = float(cfg5.get("tier3_query_weight_no_voice",    0.6))
    t3_c_nv = float(cfg5.get("tier3_combined_weight_no_voice", 0.4))

    def _get(key: str):
        return (user_profile.get(key) if hasattr(user_profile, "get")
                else user_profile[key])

    tier = int(_get("cold_start_tier") or 0)
    qw   = float(_get("query_weight") or t12_query_w)
    uw   = float(_get("user_weight")  or t12_user_w)

    if tier == 0:
        return query_emb

    if tier == 1:
        unif = _safe_emb(_get("uniform_embedding"), dim)
        if unif is None:
            return query_emb
        return _l2_norm(qw * query_emb + uw * unif)

    if tier == 2:
        rec = _safe_emb(_get("recency_embedding"), dim)
        if rec is None:
            unif = _safe_emb(_get("uniform_embedding"), dim)
            if unif is None:
                return query_emb
            rec = unif
        return _l2_norm(qw * query_emb + uw * rec)

    # tier 3: warm — three-way blend if voice available
    comb = _safe_emb(_get("combined_embedding"), dim)

    if user_voice_emb is not None:
        voice = _l2_norm(np.asarray(user_voice_emb, dtype=np.float32))
        if comb is not None:
            blend = t3_q_w * query_emb + t3_c_w * comb + t3_v_w * voice
        else:
            # No combined: split combined weight between query and voice
            blend = (t3_q_w + t3_c_w / 2) * query_emb + (t3_v_w + t3_c_w / 2) * voice
    else:
        if comb is not None:
            blend = t3_q_nv * query_emb + t3_c_nv * comb
        else:
            rec = _safe_emb(_get("recency_embedding"), dim)
            if rec is not None:
                blend = qw * query_emb + uw * rec
            else:
                return query_emb

    return _l2_norm(blend)


def get_cold_start_boost_items(
    faiss_retriever: Any,
    query_emb: np.ndarray,
    products_nlp: pd.DataFrame,
    top_k: int = 200,
) -> list[dict]:
    """Retrieve candidates for cold-start (tier 0) users.

    Re-ranks semantic candidates by a composite quality score and injects
    diversity: at least 3 distinct leaf_categories in the top 10.
    """
    fetch_k = min(top_k * 3, faiss_retriever.ntotal)
    raw_candidates = faiss_retriever.retrieve(query_emb, top_k=fetch_k)

    if not raw_candidates:
        return []

    cands_df = pd.DataFrame(raw_candidates)
    cands_df["faiss_score_norm"] = (
        (cands_df["faiss_score"] - cands_df["faiss_score"].min())
        / (cands_df["faiss_score"].max() - cands_df["faiss_score"].min() + 1e-9)
    )

    quality_cols = [c for c in ["hidden_gem_score", "category_rating_percentile", "leaf_category"]
                    if c in products_nlp.columns]
    if quality_cols:
        prod_info = products_nlp[["parent_asin"] + quality_cols].copy()
        cands_df = cands_df.merge(prod_info, on="parent_asin", how="left")

    score = 0.2 * cands_df["faiss_score_norm"]
    if "hidden_gem_score" in cands_df.columns:
        gem = cands_df["hidden_gem_score"].fillna(0.0)
        gem_norm = (gem - gem.min()) / (gem.max() - gem.min() + 1e-9)
        score = score + 0.5 * gem_norm
    if "category_rating_percentile" in cands_df.columns:
        pct = cands_df["category_rating_percentile"].fillna(0.5)
        score = score + 0.3 * pct

    cands_df["quality_score"] = score
    cands_df = cands_df.sort_values("quality_score", ascending=False).reset_index(drop=True)

    if "leaf_category" in cands_df.columns:
        top10 = cands_df.head(10).copy()
        if top10["leaf_category"].nunique() < 3:
            top10_asins = set(top10["parent_asin"])
            top10_cats  = set(top10["leaf_category"].dropna())
            rest = cands_df[~cands_df["parent_asin"].isin(top10_asins)]
            diverse_rows = rest[~rest["leaf_category"].isin(top10_cats)].head(3 - top10["leaf_category"].nunique())
            if not diverse_rows.empty:
                n_replace = len(diverse_rows)
                keep = top10.head(10 - n_replace)
                top10 = pd.concat([keep, diverse_rows], ignore_index=True)
        tail = cands_df[~cands_df["parent_asin"].isin(set(top10["parent_asin"]))].head(top_k - 10)
        cands_df = pd.concat([top10, tail], ignore_index=True)

    cands_df = cands_df.head(top_k).reset_index(drop=True)
    cands_df["rank"] = range(1, len(cands_df) + 1)
    output_cols = [c for c in ["parent_asin", "faiss_score", "rank", "quality_score"] if c in cands_df.columns]
    return cands_df[output_cols].to_dict(orient="records")
