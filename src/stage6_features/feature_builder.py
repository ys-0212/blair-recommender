"""Stage 6 feature builder — 31 features per (user, candidate) pair."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_COLS = [
    "f01_faiss_score",
    "f02_faiss_rank",
    "f03_query_item_cosine",
    "f04_user_uniform_cosine",
    "f05_avg_rating",
    "f06_rating_count_log",
    "f07_review_count_log",
    "f08_mean_sentiment",
    "f09_price_normalized",
    "f10_hidden_gem_score",
    "f11_controversy_index",
    "f12_desc_richness",
    "f13_aspect_gameplay",
    "f14_aspect_graphics",
    "f15_aspect_story",
    "f16_aspect_controls",
    "f17_aspect_value",
    "f18_user_item_voice_cosine",
    "f19_category_match",
    "f20_price_tier_match",
    "f21_user_avg_sentiment_gap",
    "f22_top_aspect_match",
    "f23_interaction_count_log",
    "f24_sentiment_trajectory",
    "f25_verified_ratio",
    "f26_helpfulness_weighted_sentiment",
    "f27_is_forced_positive",
    "f28_user_recency_cosine",
    "f29_user_rating_cosine",
    "f30_user_combined_cosine",
    "f31_bm25_score",
]

OUTPUT_COLS = [
    "user_id",
    "query_parent_asin",
    "candidate_parent_asin",
    "relevance_label",
    "is_forced_positive",
] + FEATURE_COLS

_TRAJECTORY_MAP: dict[str, float] = {
    "rising":        1.0,
    "stable":        0.5,
    "controversial": 0.25,
    "declining":     0.0,
}


@dataclass
class UserArrays:
    """Pre-built numpy arrays for all user profile fields, indexed by user_id_to_idx.

    Replaces per-row dict lookups with O(1) integer-indexed numpy matrix operations,
    turning O(batch_size) Python overhead into pure numpy gather + einsum calls.
    """
    user_id_to_idx:        dict[str, int]   # str(user_id) → row index
    uniform_matrix:        np.ndarray       # (N, dim) float32
    recency_matrix:        np.ndarray       # (N, dim) float32
    rating_matrix:         np.ndarray       # (N, dim) float32
    combined_matrix:       np.ndarray       # (N, dim) float32
    voice_matrix:          np.ndarray       # (N, dim) float32  (zeros where no voice emb)
    interaction_count_log: np.ndarray       # (N,) float32
    avg_sentiment:         np.ndarray       # (N,) float32, NaN for missing
    top_aspect:            np.ndarray       # (N,) object (str)
    preferred_price_tier:  np.ndarray       # (N,) object (str)
    top_cat_sets:          list             # list[set[str]], len N, for O(1) category lookup
    user_aspect_gameplay:  np.ndarray       # (N,) float32, NaN for missing
    user_aspect_graphics:  np.ndarray
    user_aspect_story:     np.ndarray
    user_aspect_controls:  np.ndarray
    user_aspect_value:     np.ndarray


def _safe_emb(val: Any, dim: int) -> np.ndarray | None:
    """Convert list/array profile value to float32 numpy, or None."""
    if val is None:
        return None
    try:
        arr = np.asarray(val, dtype=np.float32)
        if arr.ndim == 1 and arr.shape[0] == dim:
            return arr
    except (TypeError, ValueError):
        pass
    return None


def _parse_price(val: Any) -> float:
    """Parse raw price string → float, NaN on failure."""
    if val is None:
        return float("nan")
    try:
        v = float(str(val).replace("$", "").replace(",", "").strip())
        return v if v >= 0 else float("nan")
    except (ValueError, TypeError):
        return float("nan")


def compute_norm_stats(
    products_df: pd.DataFrame,
    user_profiles_df: pd.DataFrame,
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Compute per-feature min/max/fill stats from training sources only."""
    stats: dict[str, dict[str, float]] = {}

    def _st(col: pd.Series, lo: float | None = None, hi: float | None = None) -> dict:
        vals = col.dropna()
        mn  = float(vals.min())    if len(vals) else 0.0
        mx  = float(vals.max())    if len(vals) else 1.0
        med = float(vals.median()) if len(vals) else (mn + mx) / 2
        return {
            "min":  lo if lo is not None else mn,
            "max":  hi if hi is not None else mx,
            "fill": med,
        }

    # group 1: retrieval
    stats["f01_faiss_score"]        = {"min": -1.0,        "max": 1.0,  "fill": 0.3}
    stats["f02_faiss_rank"]         = {"min": 1.0 / top_k, "max": 1.0,  "fill": 0.5}
    stats["f03_query_item_cosine"]  = {"min": -1.0,         "max": 1.0,  "fill": 0.0}
    stats["f04_user_uniform_cosine"]= {"min": -1.0,         "max": 1.0,  "fill": 0.0}

    # group 2: product
    if "average_rating" in products_df.columns:
        st = _st(products_df["average_rating"], lo=1.0, hi=5.0)
        st["min"] = 1.0; st["max"] = 5.0
        stats["f05_avg_rating"] = st
    else:
        stats["f05_avg_rating"] = {"min": 1.0, "max": 5.0, "fill": 4.0}

    if "rating_number" in products_df.columns:
        rn_log = np.log1p(products_df["rating_number"].fillna(0).clip(lower=0))
        stats["f06_rating_count_log"] = _st(rn_log, lo=0.0)
    else:
        stats["f06_rating_count_log"] = {"min": 0.0, "max": 10.0, "fill": 2.0}

    rc_col = "review_count" if "review_count" in products_df.columns else "rating_number"
    if rc_col in products_df.columns:
        rc_log = np.log1p(products_df[rc_col].fillna(0).clip(lower=0))
        stats["f07_review_count_log"] = _st(rc_log, lo=0.0)
    else:
        stats["f07_review_count_log"] = {"min": 0.0, "max": 10.0, "fill": 2.0}

    if "mean_sentiment" in products_df.columns:
        stats["f08_mean_sentiment"] = _st(products_df["mean_sentiment"], lo=-1.0, hi=1.0)
    else:
        stats["f08_mean_sentiment"] = {"min": -1.0, "max": 1.0, "fill": 0.5}

    # f09 price — divide by p99 cap so range is truly [0, 1]
    if "price" in products_df.columns:
        prices = products_df["price"].apply(_parse_price).dropna()
        if len(prices) > 0:
            p99   = float(np.percentile(prices, 99))
            p_med = float(prices.median())
            stats["f09_price_normalized"] = {
                "min":           0.0,
                "max":           p99,
                "fill":          p_med,
                "price_p99_cap": p99,
            }
        else:
            stats["f09_price_normalized"] = {"min": 0.0, "max": 200.0, "fill": 30.0, "price_p99_cap": 200.0}
    else:
        stats["f09_price_normalized"] = {"min": 0.0, "max": 200.0, "fill": 30.0, "price_p99_cap": 200.0}

    # f10 hidden_gem_score — data-driven min/max for proper [0,1] normalization
    if "hidden_gem_score" in products_df.columns:
        stats["f10_hidden_gem_score"] = _st(products_df["hidden_gem_score"])
    else:
        stats["f10_hidden_gem_score"] = {"min": -100.0, "max": 100.0, "fill": 0.0}

    ci_col = "controversy_index" if "controversy_index" in products_df.columns else "std_sentiment"
    if ci_col in products_df.columns:
        stats["f11_controversy_index"] = _st(products_df[ci_col], lo=0.0)
    else:
        stats["f11_controversy_index"] = {"min": 0.0, "max": 1.0, "fill": 0.3}

    if "desc_richness_score" in products_df.columns:
        stats["f12_desc_richness"] = _st(products_df["desc_richness_score"], lo=0.0, hi=1.0)
    else:
        stats["f12_desc_richness"] = {"min": 0.0, "max": 1.0, "fill": 0.4}

    # group 3: aspect
    for fname, col in [
        ("f13_aspect_gameplay", "mean_aspect_gameplay"),
        ("f14_aspect_graphics", "mean_aspect_graphics"),
        ("f15_aspect_story",    "mean_aspect_story"),
        ("f16_aspect_controls", "mean_aspect_controls"),
        ("f17_aspect_value",    "mean_aspect_value"),
    ]:
        if col in products_df.columns:
            stats[fname] = _st(products_df[col], lo=-1.0, hi=1.0)
        else:
            stats[fname] = {"min": -1.0, "max": 1.0, "fill": 0.3}

    # group 4: personalization
    stats["f18_user_item_voice_cosine"] = {"min": -1.0, "max": 1.0, "fill": 0.0}
    stats["f19_category_match"]         = {"min": 0.0,  "max": 1.0, "fill": 0.0}
    stats["f20_price_tier_match"]       = {"min": 0.0,  "max": 1.0, "fill": 0.0}

    if (
        "user_avg_sentiment" in user_profiles_df.columns
        and "mean_sentiment" in products_df.columns
    ):
        us = user_profiles_df["user_avg_sentiment"].dropna()
        ms = products_df["mean_sentiment"].dropna()
        gap_max = float(
            max(abs(float(us.max()) - float(ms.min())),
                abs(float(us.min()) - float(ms.max())))
        )
        stats["f21_user_avg_sentiment_gap"] = {
            "min":  0.0,
            "max":  gap_max,
            "fill": float(us.std()) if len(us) > 1 else 0.3,
        }
    else:
        stats["f21_user_avg_sentiment_gap"] = {"min": 0.0, "max": 2.0, "fill": 0.3}

    stats["f22_top_aspect_match"] = {"min": 0.0, "max": 1.0, "fill": 0.0}

    if "interaction_count" in user_profiles_df.columns:
        ic_log = np.log1p(user_profiles_df["interaction_count"].fillna(0).clip(lower=0))
        stats["f23_interaction_count_log"] = _st(ic_log, lo=0.0)
    else:
        stats["f23_interaction_count_log"] = {"min": 0.0, "max": 6.0, "fill": 2.0}

    # group 5: temporal/quality
    stats["f24_sentiment_trajectory"] = {"min": 0.0, "max": 1.0, "fill": 0.5}

    if "verified_ratio" in products_df.columns:
        stats["f25_verified_ratio"] = _st(products_df["verified_ratio"], lo=0.0, hi=1.0)
    else:
        stats["f25_verified_ratio"] = {"min": 0.0, "max": 1.0, "fill": 0.5}

    if "helpfulness_weighted_sentiment" in products_df.columns:
        stats["f26_helpfulness_weighted_sentiment"] = _st(
            products_df["helpfulness_weighted_sentiment"], lo=-1.0, hi=1.0
        )
    else:
        stats["f26_helpfulness_weighted_sentiment"] = {"min": -1.0, "max": 1.0, "fill": 0.3}

    stats["f27_is_forced_positive"] = {"min": 0.0, "max": 1.0, "fill": 0.0}

    # group 6: additional user embedding cosines (same range as f04)
    stats["f28_user_recency_cosine"]  = {"min": -1.0, "max": 1.0, "fill": 0.0}
    stats["f29_user_rating_cosine"]   = {"min": -1.0, "max": 1.0, "fill": 0.0}
    stats["f30_user_combined_cosine"] = {"min": -1.0, "max": 1.0, "fill": 0.0}

    # group 7: BM25 — scores already normalized per-query to [0,1] before reaching here
    stats["f31_bm25_score"] = {"min": 0.0, "max": 1.0, "fill": 0.0}

    return stats


def normalize_features(df: pd.DataFrame, norm_stats: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Min-max scale feature columns to [0,1]; NaN → fill value; non-feature cols untouched."""
    df = df.copy()
    for col, st in norm_stats.items():
        if col not in df.columns:
            continue
        mn   = st["min"]
        mx   = st["max"]
        fill = st["fill"]
        rng  = mx - mn
        if rng < 1e-9:
            df[col] = 0.0
        else:
            series = df[col].astype(float)
            series = series.fillna(fill)
            df[col] = ((series - mn) / rng).clip(0.0, 1.0).astype(np.float32)
    return df


def build_features_raw(
    candidates_df: pd.DataFrame,
    query_embs: dict[tuple[str, str], np.ndarray],
    item_lookup: dict[str, np.ndarray],
    dim: int,
    user_arrays: UserArrays,
    voice_dict: dict[str, np.ndarray],
    products_df: pd.DataFrame,
    cfg: dict,
    top_k: int = 200,
    # legacy params kept for call-site compatibility — unused
    profiles_dict: Any = None,
    bm25_index: Any | None = None,
    bm25_asin_to_idx: dict[str, int] | None = None,
    bm25_token_cache: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Compute raw (un-normalised) feature values for all candidate rows.

    candidates_df must have an 'is_forced_positive' column produced by
    generate_candidates_batch(). Returns DataFrame with OUTPUT_COLS; call
    normalize_features() afterwards to scale to [0, 1].

    All user-profile lookups use pre-built numpy matrices in user_arrays for
    O(1) integer indexing instead of per-row dict accesses.
    """
    if candidates_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    df = candidates_df.copy().reset_index(drop=True)
    zero_emb = np.zeros(dim, dtype=np.float32)

    # Build integer index array once — all per-user lookups use this for numpy gather
    uid_to_idx = user_arrays.user_id_to_idx
    user_indices = np.array(
        [uid_to_idx.get(str(uid), 0) for uid in df["user_id"]], dtype=np.int32
    )
    user_found = np.array(
        [str(uid) in uid_to_idx for uid in df["user_id"]], dtype=bool
    )

    cand_embs = np.stack(
        [item_lookup.get(a, zero_emb) for a in df["candidate_parent_asin"]],
        axis=0,
    ).astype(np.float32)

    keys = list(zip(df["user_id"], df["query_parent_asin"]))
    query_embs_arr = np.stack(
        [query_embs.get(k, zero_emb) for k in keys],
        axis=0,
    ).astype(np.float32)

    # -- group 1: retrieval --
    df["f01_faiss_score"] = df["faiss_score"].astype(np.float32)
    df["f02_faiss_rank"]  = (df["faiss_rank"].astype(np.float32) / top_k)
    df["f03_query_item_cosine"] = (
        np.einsum("ij,ij->i", query_embs_arr, cand_embs).astype(np.float32)
    )

    # f04: uniform embedding cosine — single matrix gather, no per-user loop
    uniform_embs = user_arrays.uniform_matrix[user_indices].copy()
    uniform_embs[~user_found] = 0.0
    df["f04_user_uniform_cosine"] = np.einsum("ij,ij->i", uniform_embs, cand_embs).astype(np.float32)

    # join product signals
    prod_cols_needed = [
        "parent_asin", "average_rating", "rating_number", "review_count",
        "mean_sentiment", "price", "hidden_gem_score", "controversy_index",
        "std_sentiment", "desc_richness_score", "leaf_category", "price_tier",
        "mean_aspect_gameplay", "mean_aspect_graphics", "mean_aspect_story",
        "mean_aspect_controls", "mean_aspect_value",
        "top_aspect", "sentiment_trajectory", "verified_ratio",
        "helpfulness_weighted_sentiment",
    ]
    available = [c for c in prod_cols_needed if c in products_df.columns]
    prod_sub  = products_df[available].copy()

    df = df.merge(
        prod_sub.rename(columns={"parent_asin": "candidate_parent_asin"}),
        on="candidate_parent_asin",
        how="left",
    ).reset_index(drop=True)

    # Rebuild cand_embs after merge (row order preserved by left join + reset_index)
    cand_embs = np.stack(
        [item_lookup.get(a, zero_emb) for a in df["candidate_parent_asin"]],
        axis=0,
    ).astype(np.float32)

    # -- group 2: product --
    df["f05_avg_rating"] = pd.to_numeric(df.get("average_rating", np.nan), errors="coerce").astype(np.float32)

    rn = pd.to_numeric(df.get("rating_number", 0), errors="coerce").fillna(0).clip(lower=0)
    df["f06_rating_count_log"] = np.log1p(rn).astype(np.float32)

    rc_col = "review_count" if "review_count" in df.columns else "rating_number"
    rc = pd.to_numeric(df.get(rc_col, 0), errors="coerce").fillna(0).clip(lower=0)
    df["f07_review_count_log"] = np.log1p(rc).astype(np.float32)

    df["f08_mean_sentiment"] = pd.to_numeric(df.get("mean_sentiment", np.nan), errors="coerce").astype(np.float32)

    if "price" in df.columns:
        df["f09_price_normalized"] = df["price"].apply(_parse_price).astype(np.float32)
    else:
        df["f09_price_normalized"] = np.nan

    df["f10_hidden_gem_score"] = pd.to_numeric(df.get("hidden_gem_score", np.nan), errors="coerce").astype(np.float32)

    if "controversy_index" in df.columns:
        df["f11_controversy_index"] = pd.to_numeric(df["controversy_index"], errors="coerce").astype(np.float32)
    elif "std_sentiment" in df.columns:
        df["f11_controversy_index"] = pd.to_numeric(df["std_sentiment"], errors="coerce").astype(np.float32)
    else:
        df["f11_controversy_index"] = np.nan

    df["f12_desc_richness"] = pd.to_numeric(df.get("desc_richness_score", np.nan), errors="coerce").astype(np.float32)

    # -- group 3: aspect — vectorised NaN fill with user's own aspect preference --
    _asp_user_arrs = (
        user_arrays.user_aspect_gameplay,
        user_arrays.user_aspect_graphics,
        user_arrays.user_aspect_story,
        user_arrays.user_aspect_controls,
        user_arrays.user_aspect_value,
    )
    _asp_prod_cols = (
        "mean_aspect_gameplay", "mean_aspect_graphics", "mean_aspect_story",
        "mean_aspect_controls", "mean_aspect_value",
    )
    _asp_feat_names = (
        "f13_aspect_gameplay", "f14_aspect_graphics", "f15_aspect_story",
        "f16_aspect_controls", "f17_aspect_value",
    )
    for fname, prod_col, user_asp_arr in zip(_asp_feat_names, _asp_prod_cols, _asp_user_arrs):
        raw = pd.to_numeric(df.get(prod_col, np.nan), errors="coerce").values.astype(np.float32)
        nan_mask = np.isnan(raw)
        if nan_mask.any():
            user_asp = user_asp_arr[user_indices].copy()
            user_asp[~user_found] = np.nan
            filled = np.where(nan_mask, np.where(np.isnan(user_asp), 0.0, user_asp), raw)
            df[fname] = filled.astype(np.float32)
        else:
            df[fname] = raw

    # -- group 4: personalization — all vectorised, no apply() loops --

    # f18: voice cosine — matrix gather replaces per-row dict lookup
    voice_embs = user_arrays.voice_matrix[user_indices].copy()
    voice_embs[~user_found] = 0.0
    df["f18_user_item_voice_cosine"] = np.einsum("ij,ij->i", voice_embs, cand_embs).astype(np.float32)

    # f19: category match — Python loop but with pre-built sets (O(1) per lookup)
    if "leaf_category" in df.columns:
        n = len(df)
        f19 = np.zeros(n, dtype=np.float32)
        top_cat_sets = user_arrays.top_cat_sets
        leaf_cats = df["leaf_category"].values
        for i in range(n):
            if user_found[i] and leaf_cats[i] in top_cat_sets[user_indices[i]]:
                f19[i] = 1.0
        df["f19_category_match"] = f19
    else:
        df["f19_category_match"] = np.float32(0.0)

    # f20: price tier match — vectorised string comparison
    user_tiers = user_arrays.preferred_price_tier[user_indices].copy()
    user_tiers[~user_found] = ""
    if "price_tier" in df.columns:
        cand_tiers = df["price_tier"].fillna("").astype(str).values
        df["f20_price_tier_match"] = (
            (user_tiers != "") & (user_tiers == cand_tiers)
        ).astype(np.float32)
    else:
        df["f20_price_tier_match"] = np.float32(0.0)

    # f21: sentiment gap — vectorised abs difference
    user_sents = user_arrays.avg_sentiment[user_indices].copy()
    user_sents[~user_found] = np.nan
    if "mean_sentiment" in df.columns:
        prod_sents = pd.to_numeric(df["mean_sentiment"], errors="coerce").values.astype(np.float32)
        df["f21_user_avg_sentiment_gap"] = np.abs(user_sents - prod_sents).astype(np.float32)
    else:
        df["f21_user_avg_sentiment_gap"] = np.full(len(df), np.nan, dtype=np.float32)

    # f22: top aspect match — vectorised string comparison
    user_top_asps = user_arrays.top_aspect[user_indices].copy()
    user_top_asps[~user_found] = ""
    if "top_aspect" in df.columns:
        prod_top_asps = df["top_aspect"].fillna("").astype(str).values
        df["f22_top_aspect_match"] = (
            (user_top_asps != "") & (user_top_asps == prod_top_asps)
        ).astype(np.float32)
    else:
        df["f22_top_aspect_match"] = np.float32(0.0)

    # f23: interaction count log — direct array gather
    ic_log = user_arrays.interaction_count_log[user_indices].copy()
    ic_log[~user_found] = np.nan
    df["f23_interaction_count_log"] = ic_log.astype(np.float32)

    # -- group 5: temporal/quality --
    if "sentiment_trajectory" in df.columns:
        df["f24_sentiment_trajectory"] = (
            df["sentiment_trajectory"].map(_TRAJECTORY_MAP).fillna(0.5).astype(np.float32)
        )
    else:
        df["f24_sentiment_trajectory"] = np.float32(0.5)

    df["f25_verified_ratio"] = pd.to_numeric(df.get("verified_ratio", np.nan), errors="coerce").astype(np.float32)
    df["f26_helpfulness_weighted_sentiment"] = pd.to_numeric(df.get("helpfulness_weighted_sentiment", np.nan), errors="coerce").astype(np.float32)

    # -- group 6: forced positive flag --
    if "is_forced_positive" in df.columns:
        df["f27_is_forced_positive"] = df["is_forced_positive"].astype(np.float32)
    else:
        df["f27_is_forced_positive"] = np.float32(0.0)

    # -- group 7: additional user embedding cosines — vectorised matrix gathers --
    recency_embs  = user_arrays.recency_matrix[user_indices].copy()
    recency_embs[~user_found]  = 0.0
    rating_embs   = user_arrays.rating_matrix[user_indices].copy()
    rating_embs[~user_found]   = 0.0
    combined_embs = user_arrays.combined_matrix[user_indices].copy()
    combined_embs[~user_found] = 0.0
    df["f28_user_recency_cosine"]  = np.einsum("ij,ij->i", recency_embs,  cand_embs).astype(np.float32)
    df["f29_user_rating_cosine"]   = np.einsum("ij,ij->i", rating_embs,   cand_embs).astype(np.float32)
    df["f30_user_combined_cosine"] = np.einsum("ij,ij->i", combined_embs, cand_embs).astype(np.float32)

    # -- group 8: BM25 score (f31) — disabled, set to 0.0 --
    df["f31_bm25_score"] = np.float32(0.0)

    for c in OUTPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[OUTPUT_COLS]
