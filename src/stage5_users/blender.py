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


# ---------------------------------------------------------------------------
# Main blend function
# ---------------------------------------------------------------------------

def blend_query(
    query_emb: np.ndarray,
    user_profile: pd.Series | dict,
    user_voice_emb: np.ndarray | None,
    cfg: dict,
) -> np.ndarray:
    """
    Blend a BLAIR-encoded query with the user's profile embedding.

    Parameters
    ----------
    query_emb      : L2-normalised float32 array of shape (dim,).
    user_profile   : Row from user_profiles.parquet (or dict with same keys).
    user_voice_emb : BLAIR-encoded user voice (shape (dim,)), or None.
    cfg            : Full config dict.

    Returns
    -------
    L2-normalised float32 array of shape (dim,) ready for FAISS search.
    """
    query_emb = _l2_norm(np.asarray(query_emb, dtype=np.float32))
    dim = query_emb.shape[0]

    cfg5 = cfg.get("stage5", {})
    # Weights for tier-3 three-way blend
    t3_query_w = float(cfg5.get("query_weight", 0.7))
    t3_comb_w  = float(cfg5.get("user_weight",  0.3))   # split between comb + voice if available

    tier = int(user_profile.get("cold_start_tier", 0) if hasattr(user_profile, "get")
               else user_profile["cold_start_tier"])
    qw   = float(user_profile.get("query_weight",   t3_query_w)
                 if hasattr(user_profile, "get") else user_profile["query_weight"])
    uw   = float(user_profile.get("user_weight",    1 - t3_query_w)
                 if hasattr(user_profile, "get") else user_profile["user_weight"])

    # tier 0: no history, pure semantic search
    if tier == 0:
        return query_emb

    # tier 1: minimal history — uniform embedding
    if tier == 1:
        unif = _safe_emb(user_profile.get("uniform_embedding") if hasattr(user_profile, "get")
                         else user_profile["uniform_embedding"], dim)
        if unif is None:
            return query_emb
        blend = qw * query_emb + uw * unif
        return _l2_norm(blend)

    # tier 2: developing — recency embedding
    if tier == 2:
        rec = _safe_emb(user_profile.get("recency_embedding") if hasattr(user_profile, "get")
                        else user_profile["recency_embedding"], dim)
        if rec is None:
            # Fallback to uniform
            unif = _safe_emb(user_profile.get("uniform_embedding") if hasattr(user_profile, "get")
                             else user_profile["uniform_embedding"], dim)
            if unif is None:
                return query_emb
            blend = qw * query_emb + uw * unif
        else:
            blend = qw * query_emb + uw * rec
        return _l2_norm(blend)

    # tier 3: warm — three-way blend if voice available
    comb = _safe_emb(user_profile.get("combined_embedding") if hasattr(user_profile, "get")
                     else user_profile["combined_embedding"], dim)

    if user_voice_emb is not None:
        voice = _l2_norm(np.asarray(user_voice_emb, dtype=np.float32))
        # Three-way: 50% query + 30% combined profile + 20% user voice
        w_q = 0.50
        w_c = 0.30
        w_v = 0.20
        if comb is not None:
            blend = w_q * query_emb + w_c * comb + w_v * voice
        else:
            blend = (w_q + w_c / 2) * query_emb + (w_v + w_c / 2) * voice
    else:
        # Two-way: use config weights
        if comb is not None:
            blend = t3_query_w * query_emb + t3_comb_w * comb
        else:
            rec = _safe_emb(user_profile.get("recency_embedding") if hasattr(user_profile, "get")
                            else user_profile["recency_embedding"], dim)
            if rec is not None:
                blend = qw * query_emb + uw * rec
            else:
                return query_emb

    return _l2_norm(blend)


def get_cold_start_boost_items(
    faiss_retriever: Any,
    query_emb: np.ndarray,
    products_nlp: pd.DataFrame,
    top_k: int = 100,
) -> list[dict]:
    """
    Retrieve candidates for cold-start (tier 0) users.

    Strategy:
      1. Retrieve top_k * 3 semantic candidates.
      2. Re-rank by a composite quality score:
           0.5 * hidden_gem_score + 0.3 * category_rating_percentile + 0.2 * faiss_score
      3. Inject diversity: ensure at least 3 distinct leaf_categories in top 10.

    Parameters
    ----------
    faiss_retriever : Retriever instance from stage 4.
    query_emb       : L2-normalised query embedding.
    products_nlp    : products_nlp DataFrame (has hidden_gem_score, category columns).
    top_k           : Final number of candidates to return.

    Returns
    -------
    List of dicts: [{parent_asin, faiss_score, rank, quality_score}, ...]
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

    # Join product quality signals if available
    quality_cols = [c for c in ["hidden_gem_score", "category_rating_percentile", "leaf_category"]
                    if c in products_nlp.columns]
    if quality_cols:
        prod_info = products_nlp[["parent_asin"] + quality_cols].copy()
        cands_df = cands_df.merge(prod_info, on="parent_asin", how="left")

    # Build composite quality score
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

    # Diversity injection: ensure top 10 has >= 3 categories
    if "leaf_category" in cands_df.columns:
        top10 = cands_df.head(10).copy()
        n_cats = top10["leaf_category"].nunique()
        if n_cats < 3:
            # Find candidates not already in top 10 with different categories
            top10_asins = set(top10["parent_asin"])
            top10_cats  = set(top10["leaf_category"].dropna())
            rest = cands_df[~cands_df["parent_asin"].isin(top10_asins)]
            diverse_rows = rest[~rest["leaf_category"].isin(top10_cats)].head(3 - n_cats)
            if not diverse_rows.empty:
                # Replace the lowest-quality items in top 10 with diverse ones
                n_replace = len(diverse_rows)
                keep = top10.head(10 - n_replace)
                top10 = pd.concat([keep, diverse_rows], ignore_index=True)
        tail = cands_df[~cands_df["parent_asin"].isin(set(top10["parent_asin"]))].head(top_k - 10)
        cands_df = pd.concat([top10, tail], ignore_index=True)

    # Final slice
    cands_df = cands_df.head(top_k).reset_index(drop=True)
    cands_df["rank"] = range(1, len(cands_df) + 1)

    output_cols = ["parent_asin", "faiss_score", "rank", "quality_score"]
    output_cols = [c for c in output_cols if c in cands_df.columns]
    return cands_df[output_cols].to_dict(orient="records")
