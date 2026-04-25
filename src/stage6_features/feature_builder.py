"""Stage 6 feature builder — 27 features per (user, candidate) pair."""

from __future__ import annotations

import logging
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
]

# Metadata + label columns prepended to feature columns in the output file.
# is_forced_positive is the raw flag (int); f27 is the float32 feature copy.
OUTPUT_COLS = [
    "user_id",
    "query_parent_asin",
    "candidate_parent_asin",
    "relevance_label",
    "is_forced_positive",
] + FEATURE_COLS

# Trajectory string → numeric
_TRAJECTORY_MAP: dict[str, float] = {
    "rising":        1.0,
    "stable":        0.5,
    "controversial": 0.25,
    "declining":     0.0,
}


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


# normalization stats come from products/profiles only — no candidate rows, no leakage
def compute_norm_stats(
    products_df: pd.DataFrame,
    user_profiles_df: pd.DataFrame,
    top_k: int,
) -> dict[str, dict[str, float]]:
    """Compute per-feature min/max/fill stats from training sources only."""
    stats: dict[str, dict[str, float]] = {}

    def _st(col: pd.Series, lo: float | None = None, hi: float | None = None) -> dict:
        vals = col.dropna()
        mn  = float(vals.min())   if len(vals) else 0.0
        mx  = float(vals.max())   if len(vals) else 1.0
        med = float(vals.median()) if len(vals) else (mn + mx) / 2
        return {
            "min":  lo if lo is not None else mn,
            "max":  hi if hi is not None else mx,
            "fill": med,
        }

    # ── GROUP 1: Retrieval — theoretical ranges for L2-normalised embeddings ──
    stats["f01_faiss_score"]        = {"min": -1.0,        "max": 1.0,  "fill": 0.3}
    stats["f02_faiss_rank"]         = {"min": 1.0 / top_k, "max": 1.0,  "fill": 0.5}
    stats["f03_query_item_cosine"]  = {"min": -1.0,         "max": 1.0,  "fill": 0.0}
    stats["f04_user_uniform_cosine"]= {"min": -1.0,         "max": 1.0,  "fill": 0.0}

    # group 2: product
    # f05 average_rating
    if "average_rating" in products_df.columns:
        st = _st(products_df["average_rating"], lo=1.0, hi=5.0)
        st["min"] = 1.0
        st["max"] = 5.0
        stats["f05_avg_rating"] = st
    else:
        stats["f05_avg_rating"] = {"min": 1.0, "max": 5.0, "fill": 4.0}

    # f06 log(rating_number + 1)
    if "rating_number" in products_df.columns:
        rn_log = np.log1p(products_df["rating_number"].fillna(0).clip(lower=0))
        stats["f06_rating_count_log"] = _st(rn_log, lo=0.0)
    else:
        stats["f06_rating_count_log"] = {"min": 0.0, "max": 10.0, "fill": 2.0}

    # f07 log(review_count + 1)
    rc_col = "review_count" if "review_count" in products_df.columns else "rating_number"
    if rc_col in products_df.columns:
        rc_log = np.log1p(products_df[rc_col].fillna(0).clip(lower=0))
        stats["f07_review_count_log"] = _st(rc_log, lo=0.0)
    else:
        stats["f07_review_count_log"] = {"min": 0.0, "max": 10.0, "fill": 2.0}

    # f08 mean_sentiment
    if "mean_sentiment" in products_df.columns:
        stats["f08_mean_sentiment"] = _st(products_df["mean_sentiment"], lo=-1.0, hi=1.0)
    else:
        stats["f08_mean_sentiment"] = {"min": -1.0, "max": 1.0, "fill": 0.5}

    # f09 price — p99 cap handles extreme outliers (e.g. $3499.99 collector items)
    if "price" in products_df.columns:
        prices = products_df["price"].apply(_parse_price).dropna()
        if len(prices) > 0:
            p99 = float(np.percentile(prices, 99))
            p_med = float(prices.median())
            stats["f09_price_normalized"] = {
                "min":           0.0,
                "max":           p99,   # normalise to [0, p99]; clip handles >p99
                "fill":          p_med,
                "price_p99_cap": p99,   # documented explicitly in feature_stats.json
            }
        else:
            stats["f09_price_normalized"] = {
                "min": 0.0, "max": 200.0, "fill": 30.0, "price_p99_cap": 200.0,
            }
    else:
        stats["f09_price_normalized"] = {
            "min": 0.0, "max": 200.0, "fill": 30.0, "price_p99_cap": 200.0,
        }

    # f10 hidden_gem_score — data-driven min/max from training products
    if "hidden_gem_score" in products_df.columns:
        stats["f10_hidden_gem_score"] = _st(products_df["hidden_gem_score"])
    else:
        stats["f10_hidden_gem_score"] = {"min": -100.0, "max": 100.0, "fill": 0.0}

    # f11 controversy_index (= std_sentiment)
    ci_col = "controversy_index" if "controversy_index" in products_df.columns else "std_sentiment"
    if ci_col in products_df.columns:
        stats["f11_controversy_index"] = _st(products_df[ci_col], lo=0.0)
    else:
        stats["f11_controversy_index"] = {"min": 0.0, "max": 1.0, "fill": 0.3}

    # f12 desc_richness_score
    if "desc_richness_score" in products_df.columns:
        stats["f12_desc_richness"] = _st(
            products_df["desc_richness_score"], lo=0.0, hi=1.0
        )
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
        ic_log = np.log1p(
            user_profiles_df["interaction_count"].fillna(0).clip(lower=0)
        )
        stats["f23_interaction_count_log"] = _st(ic_log, lo=0.0)
    else:
        stats["f23_interaction_count_log"] = {"min": 0.0, "max": 6.0, "fill": 2.0}

    # group 5: temporal/quality
    stats["f24_sentiment_trajectory"] = {"min": 0.0, "max": 1.0, "fill": 0.5}

    if "verified_ratio" in products_df.columns:
        stats["f25_verified_ratio"] = _st(
            products_df["verified_ratio"], lo=0.0, hi=1.0
        )
    else:
        stats["f25_verified_ratio"] = {"min": 0.0, "max": 1.0, "fill": 0.5}

    if "helpfulness_weighted_sentiment" in products_df.columns:
        stats["f26_helpfulness_weighted_sentiment"] = _st(
            products_df["helpfulness_weighted_sentiment"], lo=-1.0, hi=1.0
        )
    else:
        stats["f26_helpfulness_weighted_sentiment"] = {
            "min": -1.0, "max": 1.0, "fill": 0.3,
        }

    # group 6: forced positive flag (already [0,1])
    stats["f27_is_forced_positive"] = {"min": 0.0, "max": 1.0, "fill": 0.0}

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
    profiles_dict: dict[str, Any],
    voice_dict: dict[str, np.ndarray],
    products_df: pd.DataFrame,
    cfg: dict,
    top_k: int = 100,
) -> pd.DataFrame:
    """
    Compute raw (un-normalised) feature values for all candidate rows.

    The candidates_df must have an 'is_forced_positive' column produced by
    generate_candidates_batch().

    Returns
    -------
    DataFrame with OUTPUT_COLS columns; feature columns are raw float32.
    Call normalize_features() afterwards to scale to [0, 1].
    """
    if candidates_df.empty:
        return pd.DataFrame(columns=OUTPUT_COLS)

    df = candidates_df.copy()
    zero_emb = np.zeros(dim, dtype=np.float32)

    # pre-build numpy arrays for vectorised cosine computations
    cand_embs = np.stack(
        [item_lookup.get(a, zero_emb) for a in df["candidate_parent_asin"]],
        axis=0,
    ).astype(np.float32)  # (N, dim)

    keys = list(zip(df["user_id"], df["query_parent_asin"]))
    query_embs_arr = np.stack(
        [query_embs.get(k, zero_emb) for k in keys],
        axis=0,
    ).astype(np.float32)  # (N, dim)

    # -- group 1: retrieval --
    df["f01_faiss_score"] = df["faiss_score"].astype(np.float32)
    # f02: forced positives have rank = top_k+1 → normalises to just above 1.0,
    # which normalize_features clips to 1.0 — correct: they are "worst rank".
    df["f02_faiss_rank"]  = (df["faiss_rank"].astype(np.float32) / top_k)

    df["f03_query_item_cosine"] = (
        np.einsum("ij,ij->i", query_embs_arr, cand_embs).astype(np.float32)
    )

    # f04: user uniform embedding · candidate
    uniform_map: dict[str, np.ndarray] = {}
    for uid in df["user_id"].unique():
        p = profiles_dict.get(uid)
        if p is not None:
            e = _safe_emb(p.get("uniform_embedding"), dim)
            uniform_map[uid] = e if e is not None else zero_emb
        else:
            uniform_map[uid] = zero_emb

    uniform_arr = np.stack(
        [uniform_map.get(uid, zero_emb) for uid in df["user_id"]],
        axis=0,
    ).astype(np.float32)
    df["f04_user_uniform_cosine"] = (
        np.einsum("ij,ij->i", uniform_arr, cand_embs).astype(np.float32)
    )

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
    )

    # -- group 2: product --
    df["f05_avg_rating"] = pd.to_numeric(
        df.get("average_rating", np.nan), errors="coerce"
    ).astype(np.float32)

    rn = pd.to_numeric(df.get("rating_number", 0), errors="coerce").fillna(0).clip(lower=0)
    df["f06_rating_count_log"] = np.log1p(rn).astype(np.float32)

    rc_col = "review_count" if "review_count" in df.columns else "rating_number"
    rc = pd.to_numeric(df.get(rc_col, 0), errors="coerce").fillna(0).clip(lower=0)
    df["f07_review_count_log"] = np.log1p(rc).astype(np.float32)

    df["f08_mean_sentiment"] = pd.to_numeric(
        df.get("mean_sentiment", np.nan), errors="coerce"
    ).astype(np.float32)

    # f09 price — parse raw string; normalize_features will apply p99 cap via max
    if "price" in df.columns:
        df["f09_price_normalized"] = df["price"].apply(_parse_price).astype(np.float32)
    else:
        df["f09_price_normalized"] = np.nan

    df["f10_hidden_gem_score"] = pd.to_numeric(
        df.get("hidden_gem_score", np.nan), errors="coerce"
    ).astype(np.float32)

    if "controversy_index" in df.columns:
        df["f11_controversy_index"] = pd.to_numeric(
            df["controversy_index"], errors="coerce"
        ).astype(np.float32)
    elif "std_sentiment" in df.columns:
        df["f11_controversy_index"] = pd.to_numeric(
            df["std_sentiment"], errors="coerce"
        ).astype(np.float32)
    else:
        df["f11_controversy_index"] = np.nan

    df["f12_desc_richness"] = pd.to_numeric(
        df.get("desc_richness_score", np.nan), errors="coerce"
    ).astype(np.float32)

    # -- group 3: aspect --
    for fname, col in [
        ("f13_aspect_gameplay", "mean_aspect_gameplay"),
        ("f14_aspect_graphics", "mean_aspect_graphics"),
        ("f15_aspect_story",    "mean_aspect_story"),
        ("f16_aspect_controls", "mean_aspect_controls"),
        ("f17_aspect_value",    "mean_aspect_value"),
    ]:
        df[fname] = pd.to_numeric(
            df.get(col, np.nan), errors="coerce"
        ).astype(np.float32)

    # -- group 4: personalization --
    voice_arr = np.stack(
        [voice_dict.get(uid, zero_emb) for uid in df["user_id"]],
        axis=0,
    ).astype(np.float32)
    df["f18_user_item_voice_cosine"] = (
        np.einsum("ij,ij->i", voice_arr, cand_embs).astype(np.float32)
    )

    def _cat_match(row: pd.Series) -> float:
        p = profiles_dict.get(row["user_id"])
        if p is None:
            return 0.0
        top_cats = p.get("top_categories", [])
        if not isinstance(top_cats, list) or not top_cats:
            return 0.0
        return 1.0 if row.get("leaf_category") in top_cats else 0.0

    df["f19_category_match"] = df.apply(_cat_match, axis=1).astype(np.float32)

    def _tier_match(row: pd.Series) -> float:
        p = profiles_dict.get(row["user_id"])
        if p is None:
            return 0.0
        pref = p.get("preferred_price_tier")
        cand = row.get("price_tier")
        return 1.0 if (pref and cand and pref == cand) else 0.0

    df["f20_price_tier_match"] = df.apply(_tier_match, axis=1).astype(np.float32)

    def _sent_gap(row: pd.Series) -> float:
        p = profiles_dict.get(row["user_id"])
        if p is None:
            return float("nan")
        try:
            return float(abs(float(p.get("user_avg_sentiment", float("nan")))
                             - float(row.get("mean_sentiment", float("nan")))))
        except (TypeError, ValueError):
            return float("nan")

    df["f21_user_avg_sentiment_gap"] = df.apply(_sent_gap, axis=1).astype(np.float32)

    def _asp_match(row: pd.Series) -> float:
        p = profiles_dict.get(row["user_id"])
        if p is None:
            return 0.0
        user_top = p.get("user_top_aspect")
        prod_top = row.get("top_aspect")
        return 1.0 if (user_top and prod_top and user_top == prod_top) else 0.0

    df["f22_top_aspect_match"] = df.apply(_asp_match, axis=1).astype(np.float32)

    def _ic_log(uid: str) -> float:
        p = profiles_dict.get(uid)
        if p is None:
            return float("nan")
        cnt = p.get("interaction_count")
        try:
            return float(np.log1p(max(0.0, float(cnt))))
        except (TypeError, ValueError):
            return float("nan")

    df["f23_interaction_count_log"] = df["user_id"].apply(_ic_log).astype(np.float32)

    # -- group 5: temporal/quality --
    if "sentiment_trajectory" in df.columns:
        df["f24_sentiment_trajectory"] = (
            df["sentiment_trajectory"].map(_TRAJECTORY_MAP).fillna(0.5).astype(np.float32)
        )
    else:
        df["f24_sentiment_trajectory"] = np.float32(0.5)

    df["f25_verified_ratio"] = pd.to_numeric(
        df.get("verified_ratio", np.nan), errors="coerce"
    ).astype(np.float32)

    df["f26_helpfulness_weighted_sentiment"] = pd.to_numeric(
        df.get("helpfulness_weighted_sentiment", np.nan), errors="coerce"
    ).astype(np.float32)

    # -- group 6: forced positive flag --
    # is_forced_positive came in from candidates_df; copy to float32 feature.
    if "is_forced_positive" in df.columns:
        df["f27_is_forced_positive"] = df["is_forced_positive"].astype(np.float32)
    else:
        df["f27_is_forced_positive"] = np.float32(0.0)

    # finalise output schema
    for c in OUTPUT_COLS:
        if c not in df.columns:
            df[c] = np.nan
    return df[OUTPUT_COLS]
