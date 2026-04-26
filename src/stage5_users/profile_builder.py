"""Stage 5 — builds per-user feature vectors from interactions, reviews, and embeddings."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

from src.utils.config import get_path, load_config

logger = logging.getLogger(__name__)


def _load_item_lookup(cfg: dict) -> tuple[dict[str, np.ndarray], int]:
    """Return {parent_asin: embedding_vector} and embedding dim."""
    emb_dir = get_path(cfg, "data_embeddings")
    emb = np.load(emb_dir / "item_embeddings.npy").astype(np.float32)
    ids = np.load(emb_dir / "item_ids.npy", allow_pickle=True).tolist()
    lookup = {asin: emb[i] for i, asin in enumerate(ids)}
    dim = emb.shape[1]
    logger.info("Loaded item embedding lookup: %d items, dim=%d", len(lookup), dim)
    return lookup, dim


def _l2_norm(v: np.ndarray) -> np.ndarray:
    """L2-normalise a vector; return zero-vector unchanged."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _weighted_mean_embedding(
    asins: list[str],
    weights: np.ndarray,
    lookup: dict[str, np.ndarray],
    dim: int,
) -> np.ndarray | None:
    """Weighted mean of embeddings for the given asins.

    Skips items not present in the lookup (no embedding available).
    Returns None if no valid items found.
    """
    vecs, ws = [], []
    for asin, w in zip(asins, weights):
        if asin in lookup and w > 0:
            vecs.append(lookup[asin])
            ws.append(w)
    if not vecs:
        return None
    mat = np.stack(vecs, axis=0)           # (N, dim)
    ws_arr = np.array(ws, dtype=np.float32)
    ws_arr /= ws_arr.sum()                 # normalize weights
    result = (mat * ws_arr[:, None]).sum(axis=0)
    return _l2_norm(result)


# ---------------------------------------------------------------------------
# Step 1 — Interaction-based signals
# ---------------------------------------------------------------------------

def _build_interaction_signals(
    train: pd.DataFrame,
    products: pd.DataFrame,
    item_lookup: dict[str, np.ndarray],
    dim: int,
    recency_lambda: float,
) -> pd.DataFrame:
    """Per-user interaction signals: counts, embeddings, category/price prefs."""
    logger.info("Building interaction-based signals for %d train rows ...", len(train))

    agg = (
        train.groupby("user_id")
        .agg(
            interaction_count=("parent_asin", "count"),
            avg_rating_given=("rating", "mean"),
            rating_std=("rating", "std"),
            first_interaction_ts=("timestamp", "min"),
            last_interaction_ts=("timestamp", "max"),
        )
        .reset_index()
    )
    agg["rating_std"] = agg["rating_std"].fillna(0.0)
    agg["active_days"] = (
        (agg["last_interaction_ts"] - agg["first_interaction_ts"]) / 86_400_000
    ).clip(lower=0)

    logger.info("  computing embedding profiles")
    dataset_max_ts = train["timestamp"].max()

    # Collect per-user rows once
    grouped = train.sort_values("timestamp").groupby("user_id", sort=False)

    uniform_embs: dict[str, np.ndarray | None] = {}
    recency_embs: dict[str, np.ndarray | None] = {}
    rating_embs:  dict[str, np.ndarray | None] = {}
    combined_embs: dict[str, np.ndarray | None] = {}

    for uid, grp in grouped:
        asins   = grp["parent_asin"].tolist()
        ts_arr  = grp["timestamp"].to_numpy(dtype=np.float64)
        rat_arr = grp["rating"].to_numpy(dtype=np.float32)
        n       = len(asins)

        # Uniform
        unif_w = np.ones(n, dtype=np.float32)
        uniform_embs[uid] = _weighted_mean_embedding(asins, unif_w, item_lookup, dim)

        # Recency-weighted  w = exp(-lambda * delta_days)
        delta_days = (dataset_max_ts - ts_arr) / 86_400_000
        rec_w = np.exp(-recency_lambda * delta_days).astype(np.float32)
        recency_embs[uid] = _weighted_mean_embedding(asins, rec_w, item_lookup, dim)

        # Rating-weighted  w = rating
        rating_embs[uid] = _weighted_mean_embedding(asins, rat_arr, item_lookup, dim)

        # Combined  w = rating * exp(-lambda * delta_days)
        comb_w = rat_arr * rec_w
        combined_embs[uid] = _weighted_mean_embedding(asins, comb_w, item_lookup, dim)

    agg["uniform_embedding"]  = agg["user_id"].map(uniform_embs)
    agg["recency_embedding"]  = agg["user_id"].map(recency_embs)
    agg["rating_embedding"]   = agg["user_id"].map(rating_embs)
    agg["combined_embedding"] = agg["user_id"].map(combined_embs)

    logger.info("  computing category preferences")
    cat_map = products.set_index("parent_asin")["leaf_category"].to_dict()
    tier_map = products.set_index("parent_asin")["price_tier"].to_dict()

    def _cat_signals(grp: pd.DataFrame) -> pd.Series:
        cats = [cat_map.get(a, "unknown") for a in grp["parent_asin"]]
        cats = [c for c in cats if c and c != "unknown"]
        if not cats:
            return pd.Series({
                "top_categories": [],
                "category_diversity": 0.0,
                "dominant_category": "unknown",
                "category_entropy": 0.0,
            })
        from collections import Counter
        counts = Counter(cats)
        total  = len(cats)
        top3   = [c for c, _ in counts.most_common(3)]
        dom    = counts.most_common(1)[0][0]
        unique = len(counts)
        div    = unique / total
        probs  = np.array(list(counts.values()), dtype=float) / total
        ent    = float(scipy_entropy(probs))
        return pd.Series({
            "top_categories":    top3,
            "category_diversity": div,
            "dominant_category": dom,
            "category_entropy":  ent,
        })

    def _price_signals(grp: pd.DataFrame) -> pd.Series:
        tiers = [tier_map.get(a, "unknown") for a in grp["parent_asin"]]
        tiers = [t for t in tiers if t and t != "unknown"]
        if not tiers:
            return pd.Series({"preferred_price_tier": "unknown", "price_diversity": 0.0})
        from collections import Counter
        counts = Counter(tiers)
        pref   = counts.most_common(1)[0][0]
        div    = min(len(counts) / 4.0, 1.0)   # 4 possible tiers
        return pd.Series({"preferred_price_tier": pref, "price_diversity": div})

    try:
        cat_df   = grouped.apply(_cat_signals, include_groups=False).reset_index()
        price_df = grouped.apply(_price_signals, include_groups=False).reset_index()
    except TypeError:
        # pandas < 2.2 does not have include_groups parameter
        cat_df   = grouped.apply(_cat_signals).reset_index()
        price_df = grouped.apply(_price_signals).reset_index()

    agg = agg.merge(cat_df,   on="user_id", how="left")
    agg = agg.merge(price_df, on="user_id", how="left")

    agg["interaction_velocity"] = (
        agg["interaction_count"] / agg["active_days"].clip(lower=1)
    )
    agg["recency_score"] = (dataset_max_ts - agg["last_interaction_ts"]) / 86_400_000

    logger.info("  Interaction signals done: %d users", len(agg))
    return agg


def _build_review_signals(reviews_nlp: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Per-user signals from written reviews: sentiment, aspect coverage, length."""
    logger.info("Building review-based signals ...")
    aspects = cfg["stage2"]["aspects"]

    # Text length in words
    reviews_nlp = reviews_nlp.copy()
    if "text" in reviews_nlp.columns:
        reviews_nlp["_word_count"] = reviews_nlp["text"].fillna("").str.split().str.len()
    else:
        reviews_nlp["_word_count"] = 0

    # Aspect coverage: 1 if user mentioned any aspect in this review
    aspect_cols = [f"aspect_{a}" for a in aspects]
    existing_aspect_cols = [c for c in aspect_cols if c in reviews_nlp.columns]
    if existing_aspect_cols:
        reviews_nlp["_any_aspect"] = reviews_nlp[existing_aspect_cols].notna().any(axis=1)
    else:
        reviews_nlp["_any_aspect"] = False

    def _safe_mean(s: pd.Series) -> float:
        v = s.dropna()
        return float(v.mean()) if len(v) > 0 else float("nan")

    def _user_aspects(grp: pd.DataFrame) -> dict:
        result = {}
        for col in existing_aspect_cols:
            result[f"user_{col}"] = _safe_mean(grp[col])
        return result

    g = reviews_nlp.groupby("user_id")

    base_agg = g.agg(
        user_avg_sentiment=("sentiment_score", "mean"),
        user_sentiment_std=("sentiment_score", "std"),
        user_review_count=("sentiment_score", "count"),
        user_avg_review_length=("_word_count", "mean"),
        user_helpful_votes_received=("helpful_vote", "sum"),
        user_aspect_coverage=("_any_aspect", "mean"),
    ).reset_index()
    base_agg["user_sentiment_std"] = base_agg["user_sentiment_std"].fillna(0.0)

    # Positive / negative ratios
    if "sentiment_label" in reviews_nlp.columns:
        lab = (
            reviews_nlp.groupby("user_id")["sentiment_label"]
            .value_counts(normalize=True)
            .unstack(fill_value=0.0)
            .reset_index()
        )
        for col in ("positive", "negative", "neutral"):
            if col not in lab.columns:
                lab[col] = 0.0
        lab = lab[["user_id", "positive", "negative"]].rename(
            columns={"positive": "user_pos_ratio", "negative": "user_neg_ratio"}
        )
        base_agg = base_agg.merge(lab, on="user_id", how="left")
    else:
        base_agg["user_pos_ratio"] = float("nan")
        base_agg["user_neg_ratio"] = float("nan")

    # Verified ratio
    if "verified_purchase" in reviews_nlp.columns:
        ver = (
            reviews_nlp.groupby("user_id")["verified_purchase"]
            .mean()
            .rename("user_verified_ratio")
            .reset_index()
        )
        base_agg = base_agg.merge(ver, on="user_id", how="left")
    else:
        base_agg["user_verified_ratio"] = float("nan")

    # Aspect means
    if existing_aspect_cols:
        asp_df = (
            reviews_nlp.groupby("user_id")[existing_aspect_cols]
            .mean()
            .reset_index()
            .rename(columns={c: f"user_{c}" for c in existing_aspect_cols})
        )
        base_agg = base_agg.merge(asp_df, on="user_id", how="left")

    # Top / worst aspect per user (data-driven min coverage threshold)
    user_asp_cols = [f"user_aspect_{a}" for a in aspects
                     if f"user_aspect_{a}" in base_agg.columns]

    if user_asp_cols:
        # Min coverage: 5th percentile of non-NaN coverage values
        asp_mat = base_agg[user_asp_cols]
        non_nan_fracs = (~asp_mat.isna()).mean(axis=0)
        min_cov = float(np.percentile(non_nan_fracs[non_nan_fracs > 0], 5))
        nan_fracs_dict = non_nan_fracs.to_dict()

        def _top_worst(row: pd.Series) -> pd.Series:
            vals = {}
            for col in user_asp_cols:
                v = row[col]
                try:
                    if np.isnan(v):
                        continue
                except (TypeError, ValueError):
                    continue
                vals[col.replace("user_aspect_", "")] = v
            if not vals:
                return pd.Series({"user_top_aspect": None, "user_worst_aspect": None})
            # Only aspects with meaningful signal
            filtered = {k: v for k, v in vals.items()
                        if nan_fracs_dict.get(f"user_aspect_{k}", 0) >= min_cov}
            if not filtered:
                filtered = vals
            top   = max(filtered, key=filtered.get)
            worst = min(filtered, key=filtered.get)
            return pd.Series({"user_top_aspect": top, "user_worst_aspect": worst})

        top_worst = base_agg[user_asp_cols + ["user_id"]].apply(_top_worst, axis=1)
        base_agg["user_top_aspect"]   = top_worst["user_top_aspect"].values
        base_agg["user_worst_aspect"] = top_worst["user_worst_aspect"].values
    else:
        base_agg["user_top_aspect"]   = None
        base_agg["user_worst_aspect"] = None

    logger.info("  Review signals done: %d users", len(base_agg))
    return base_agg


def validate_recency_lambda(
    profiles_df: pd.DataFrame,
    item_emb_lookup: dict[str, np.ndarray],
    cfg: dict,
) -> None:
    """Test 3 lambda values and report which creates most differentiation from uniform embedding.

    Only diagnostic — does not modify profiles or config.
    """
    from src.utils.config import get_path
    lambdas = [0.0001, 0.001, 0.01]
    proc = get_path(cfg, "data_processed")
    try:
        train = pd.read_parquet(proc / "train.parquet")
    except Exception as e:
        logger.warning("Could not load train.parquet for recency validation: %s", e)
        return

    dataset_max_ts = float(train["timestamp"].max())
    dim = next(iter(item_emb_lookup.values())).shape[0]

    logger.info("Recency lambda validation:")
    for lam in lambdas:
        diffs = []
        for uid, grp in train.sort_values("timestamp").groupby("user_id", sort=False):
            asins   = grp["parent_asin"].tolist()
            ts_arr  = grp["timestamp"].to_numpy(dtype=np.float64)
            rat_arr = np.ones(len(asins), dtype=np.float32)

            unif_emb = _weighted_mean_embedding(asins, rat_arr, item_emb_lookup, dim)
            delta    = (dataset_max_ts - ts_arr) / 86_400_000
            rec_w    = np.exp(-lam * delta).astype(np.float32)
            rec_emb  = _weighted_mean_embedding(asins, rec_w, item_emb_lookup, dim)

            if unif_emb is not None and rec_emb is not None:
                cos_sim = float(np.dot(unif_emb, rec_emb))
                diffs.append(1.0 - cos_sim)  # higher = more differentiation

            if len(diffs) >= 5000:  # sample for speed
                break

        mean_diff = float(np.mean(diffs)) if diffs else 0.0
        logger.info("  lambda=%.4f  mean(1-cosine_to_uniform)=%.4f", lam, mean_diff)


def _assign_cold_start_tiers(profiles: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Assign cold_start_tier 0-3 and query/user weights from interaction count percentiles.

    Note: in the training set (5-core, >=5 interactions per user), all users will
    be tier 3 (warm). Cold-start tiers are meaningful at inference time for new users.
    """
    cfg5 = cfg.get("stage5", {})
    std_query_w = float(cfg5.get("query_weight", 0.7))

    counts = profiles["interaction_count"].dropna().values
    t1 = float(np.percentile(counts, 5))
    t2 = float(np.percentile(counts, 25))
    t3 = float(np.percentile(counts, 50))

    logger.info(
        "Cold-start thresholds (data-driven): "
        "tier1_max=%.1f  tier2_max=%.1f  tier3_max=%.1f",
        t1, t2, t3,
    )

    # Query weights scale linearly between tier boundaries
    # tier 1 -> 0.9  tier 2 -> midpoint  tier 3 -> std_query_w
    qw_tier1 = min(0.9, 1.0 - 0.1 * std_query_w)
    qw_tier2 = (qw_tier1 + std_query_w) / 2.0

    def _tier(cnt: int) -> int:
        if cnt is None or np.isnan(cnt):
            return 0
        if cnt < t1:
            return 1
        if cnt < t2:
            return 2
        return 3

    def _qw(tier: int) -> float:
        if tier <= 1:
            return qw_tier1
        if tier == 2:
            return qw_tier2
        return std_query_w

    profiles = profiles.copy()
    profiles["cold_start_tier"] = profiles["interaction_count"].apply(_tier)
    profiles["query_weight"]    = profiles["cold_start_tier"].apply(_qw)
    profiles["user_weight"]     = 1.0 - profiles["query_weight"]

    # Data-driven active threshold: 75th pct of recency_score
    active_threshold = float(np.percentile(profiles["recency_score"].dropna(), 75))
    profiles["is_active"] = profiles["recency_score"] <= active_threshold
    logger.info("Active threshold (75th pct recency): %.1f days", active_threshold)

    tier_dist = profiles["cold_start_tier"].value_counts().sort_index()
    logger.info("Cold-start tier distribution:\n%s", tier_dist.to_string())
    return profiles


def build_user_profiles(cfg: dict | None = None) -> pd.DataFrame:
    """Build per-user profile DataFrame from interactions, reviews, and embeddings."""
    if cfg is None:
        cfg = load_config()
    cfg5 = cfg.get("stage5", {})
    recency_lambda = float(cfg5.get("recency_lambda", 0.001))

    proc = get_path(cfg, "data_processed")
    logger.info("loading train.parquet")
    train = pd.read_parquet(proc / "train.parquet")

    logger.info("loading reviews_nlp.parquet")
    rev_cols = (
        ["user_id", "parent_asin", "sentiment_score", "sentiment_label",
         "helpful_vote", "verified_purchase", "text"]
        + [f"aspect_{a}" for a in cfg["stage2"]["aspects"]]
    )
    existing_rev_cols = None   # load all; filter below
    reviews_nlp = pd.read_parquet(proc / "reviews_nlp.parquet")
    # Keep only needed columns if present
    keep = [c for c in rev_cols if c in reviews_nlp.columns]
    reviews_nlp = reviews_nlp[keep]

    logger.info("loading products_nlp.parquet")
    prod_cols = ["parent_asin", "leaf_category", "price_tier"]
    products = pd.read_parquet(proc / "products_nlp.parquet", columns=prod_cols)

    item_lookup, dim = _load_item_lookup(cfg)

    interaction_df = _build_interaction_signals(
        train, products, item_lookup, dim, recency_lambda
    )
    review_df = _build_review_signals(reviews_nlp, cfg)

    profiles = interaction_df.merge(review_df, on="user_id", how="left")
    profiles = _assign_cold_start_tiers(profiles, cfg)

    logger.info(
        "User profiles built: %d users, %d columns",
        len(profiles), profiles.shape[1],
    )
    return profiles
