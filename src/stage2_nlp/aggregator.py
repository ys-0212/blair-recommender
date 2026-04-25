"""Stage 2 aggregator — per-review NLP signals → product-level signals (~90 cols)."""

from __future__ import annotations

import logging
import re
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)

ASPECTS = ["gameplay", "graphics", "story", "controls", "value"]

def _ts(label: str) -> None:
    logger.info("[AGG] %s", label)


def _safe_float(val: Any) -> float | None:
    """Return float or None for NaN/None/non-numeric."""
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def _extract_category_parts(cat_str: Any) -> list[str]:
    """Parse the '. '-joined categories string back to a list of parts."""
    if not cat_str or (isinstance(cat_str, float) and np.isnan(cat_str)):
        return []
    return [p.strip() for p in str(cat_str).split(".") if p.strip()]


def _build_review_corpus(reviews: pd.DataFrame) -> pd.Series:
    """Concatenate all review texts per product into a single string."""
    return (
        reviews.groupby("parent_asin")["text"]
        .apply(lambda texts: " ".join(str(t) for t in texts if t))
        .rename("review_corpus")
    )


def _agg_base(reviews: pd.DataFrame) -> pd.DataFrame:
    """Base per-product aggregation: sentiment, aspect stats, helpfulness, verified."""
    rv = reviews.copy()

    # Pre-compute row-level helpers
    rv["_word_count"] = rv["text"].fillna("").str.split().str.len()
    rv["_weight"] = pd.to_numeric(rv["helpful_vote"], errors="coerce").fillna(0).clip(lower=1)
    rv["_helpful_flag"] = pd.to_numeric(rv["helpful_vote"], errors="coerce").fillna(0) > 0
    rv["_verified"] = rv["verified_purchase"].astype(str).str.strip().str.lower() == "true"
    rv["_wsent"] = rv["sentiment_score"] * rv["_weight"]

    # Core groupby
    agg = rv.groupby("parent_asin").agg(
        review_count   =("sentiment_score", "count"),
        mean_sentiment =("sentiment_score", "mean"),
        std_sentiment  =("sentiment_score", "std"),
        _sum_w         =("_weight",       "sum"),
        _sum_ws        =("_wsent",        "sum"),
        _helpful_cnt   =("_helpful_flag", "sum"),
        _verified_cnt  =("_verified",     "sum"),
        avg_review_length=("_word_count", "mean"),
        _ts_min        =("timestamp",     "min"),
        _ts_max        =("timestamp",     "max"),
    ).reset_index()

    agg["helpfulness_weighted_sentiment"] = (agg["_sum_ws"] / agg["_sum_w"]).astype(np.float32)
    agg["helpful_ratio"]  = (agg["_helpful_cnt"]  / agg["review_count"]).astype(np.float32)
    agg["verified_ratio"] = (agg["_verified_cnt"] / agg["review_count"]).astype(np.float32)
    agg["days_on_market"] = (
        (agg["_ts_max"] - agg["_ts_min"]) / 86_400_000
    ).clip(lower=0).astype(np.float32)

    agg = agg.drop(columns=["_sum_w", "_sum_ws", "_helpful_cnt",
                              "_verified_cnt", "_ts_min", "_ts_max"])

    # Sentiment label ratios
    lc = (
        rv.groupby(["parent_asin", "sentiment_label"])
        .size().unstack(fill_value=0).reset_index()
    )
    for col in ["positive", "neutral", "negative"]:
        if col not in lc.columns:
            lc[col] = 0
    tot = lc[["positive", "neutral", "negative"]].sum(axis=1).clip(lower=1)
    lc["pos_ratio"] = (lc["positive"] / tot).astype(np.float32)
    lc["neg_ratio"] = (lc["negative"] / tot).astype(np.float32)
    lc["neu_ratio"] = (lc["neutral"]  / tot).astype(np.float32)
    agg = agg.merge(lc[["parent_asin", "pos_ratio", "neg_ratio", "neu_ratio"]],
                    on="parent_asin", how="left")

    # Per-aspect stats
    for a in ASPECTS:
        col = f"aspect_{a}"
        if col not in rv.columns:
            agg[f"mean_aspect_{a}"] = np.nan
            agg[f"coverage_{a}"]    = np.nan
            agg[f"std_aspect_{a}"]  = np.nan
            continue
        grp = rv.groupby("parent_asin")[col]
        asp = pd.DataFrame({
            "parent_asin":        grp.mean().index,
            f"mean_aspect_{a}":   grp.mean().values.astype(np.float32),
            f"coverage_{a}":      grp.apply(lambda s: s.notna().mean()).values.astype(np.float32),
            f"std_aspect_{a}":    grp.std().values.astype(np.float32),
        })
        agg = agg.merge(asp, on="parent_asin", how="left")

    return agg


def _agg_temporal(reviews: pd.DataFrame, cfg_s2: dict) -> pd.DataFrame:
    """Vectorised early/mid/recent window aggregation of sentiment and aspects."""
    window_pct  = float(cfg_s2.get("temporal_window_pct",    0.10))
    mid_start   = float(cfg_s2.get("temporal_mid_start_pct", 0.45))
    mid_end     = float(cfg_s2.get("temporal_mid_end_pct",   0.55))
    min_reviews = int  (cfg_s2.get("temporal_min_reviews",   10))

    rv = reviews.copy()
    rv["timestamp"] = pd.to_numeric(rv["timestamp"], errors="coerce")
    rv = rv.dropna(subset=["timestamp"])
    rv = rv.sort_values(["parent_asin", "timestamp"]).reset_index(drop=True)

    # Cumulative rank (0-based) and total count per product
    rv["_rank"]  = rv.groupby("parent_asin").cumcount()
    rv["_total"] = rv.groupby("parent_asin")["parent_asin"].transform("count")
    rv["_small"] = rv["_total"] < min_reviews

    # Window size k = max(1, floor(total * window_pct))
    k = np.maximum(1, (rv["_total"] * window_pct).astype(int))
    mid_lo = (rv["_total"] * mid_start).astype(int)
    mid_hi = np.maximum(mid_lo + 1, (rv["_total"] * mid_end).astype(int))

    rv["_in_early"]  = rv["_small"] | (rv["_rank"] < k)
    rv["_in_recent"] = rv["_small"] | (rv["_rank"] >= rv["_total"] - k)
    rv["_in_mid"]    = rv["_small"] | (
        (rv["_rank"] >= mid_lo) & (rv["_rank"] < mid_hi)
    )

    # Columns to window-aggregate
    sent_col  = "sentiment_score"
    asp_cols  = [f"aspect_{a}" for a in ASPECTS if f"aspect_{a}" in rv.columns]
    value_cols = [sent_col] + asp_cols

    def _window_means(mask_col: str, prefix: str) -> pd.DataFrame:
        return (
            rv[rv[mask_col]]
            .groupby("parent_asin")[value_cols]
            .mean()
            .add_prefix(f"{prefix}_")
            .reset_index()
        )

    early  = _window_means("_in_early",  "early")
    mid    = _window_means("_in_mid",    "mid")
    recent = _window_means("_in_recent", "recent")

    # Rename sentiment columns to clean names
    for df, prefix in [(early, "early"), (mid, "mid"), (recent, "recent")]:
        df.rename(columns={f"{prefix}_{sent_col}": f"{prefix}_sentiment"}, inplace=True)
        for a in ASPECTS:
            old = f"{prefix}_aspect_{a}"
            new = f"{prefix}_aspect_{a}"   # keep as-is: early_aspect_gameplay etc.
            if old in df.columns:
                df.rename(columns={old: new}, inplace=True)

    # Period strings from timestamps
    early_ts = rv[rv["_in_early"]].groupby("parent_asin")["timestamp"].min()
    recent_ts = rv[rv["_in_recent"]].groupby("parent_asin")["timestamp"].max()

    periods = pd.DataFrame({
        "parent_asin":   early_ts.index,
        "early_period":  pd.to_datetime(early_ts.values, unit="ms").strftime("%Y-%m"),
        "recent_period": pd.to_datetime(recent_ts.values, unit="ms").strftime("%Y-%m"),
    })

    # Merge all temporal frames
    temp = early.merge(mid, on="parent_asin", how="outer")
    temp = temp.merge(recent, on="parent_asin", how="outer")
    temp = temp.merge(periods, on="parent_asin", how="left")
    return temp


def _agg_emotion_vocab(reviews: pd.DataFrame) -> pd.DataFrame:
    """Top-5 emotionally charged words per product using VADER lexicon directly."""
    sia = SentimentIntensityAnalyzer()
    lexicon = sia.lexicon  # word -> float

    corpus = reviews.groupby("parent_asin")["text"].apply(
        lambda texts: " ".join(str(t) for t in texts if t)
    )

    word_re = re.compile(r"\b[a-zA-Z]{3,}\b")  # words with >= 3 chars

    def _top_emo(text: str) -> str:
        words = set(word_re.findall(text.lower()))
        scored = {w: lexicon[w] for w in words if w in lexicon}
        if not scored:
            return ""
        top = sorted(scored, key=lambda w: abs(scored[w]), reverse=True)[:5]
        return ", ".join(top)

    result = corpus.apply(_top_emo).rename("emotion_vocabulary").reset_index()
    return result


def _compute_listing_quality(products: pd.DataFrame) -> pd.DataFrame:
    """Add leaf_category, full_category_path, feature_count, desc_richness_score."""
    prod = products.copy()

    # Feature count — handle both string (". "-joined) and list
    def _count_feats(val: Any) -> int:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0
        if isinstance(val, list):
            return len([v for v in val if v])
        return len([p for p in str(val).split(".") if p.strip()])

    prod["feature_count"] = prod["features"].apply(_count_feats).astype(np.int16)

    # Description richness: unique-word ratio
    def _richness(val: Any) -> float:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0.0
        words = str(val).lower().split()
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    prod["desc_richness_score"] = prod["description"].apply(_richness).astype(np.float32)

    # Category parsing
    cat_parts = prod["categories"].apply(_extract_category_parts)
    prod["leaf_category"]      = cat_parts.apply(lambda p: p[-1] if p else "Unknown")
    prod["full_category_path"] = cat_parts.apply(lambda p: " > ".join(p) if p else "Unknown")

    return prod


def _compute_price_tier(products: pd.DataFrame) -> pd.DataFrame:
    """Data-driven price tier using p25/p75/p90 of valid prices."""
    prod = products.copy()
    valid_prices = pd.to_numeric(prod["price"], errors="coerce").dropna()
    p25 = float(np.percentile(valid_prices, 25))
    p75 = float(np.percentile(valid_prices, 75))
    p90 = float(np.percentile(valid_prices, 90))
    logger.info("[AGG] Price tier thresholds: p25=%.2f, p75=%.2f, p90=%.2f", p25, p75, p90)

    def _tier(val: Any) -> str:
        v = _safe_float(val)
        if v is None:
            return "unknown"
        if v <= p25:
            return "budget"
        if v <= p75:
            return "mid"
        if v <= p90:
            return "premium"
        return "luxury"

    prod["price_tier"] = prod["price"].apply(_tier)
    return prod


def _compute_reality_gap(products: pd.DataFrame, aspect_keywords: dict) -> pd.DataFrame:
    """Official description vs community sentiment gap signals."""
    prod = products.copy()
    sia = SentimentIntensityAnalyzer()

    # Official description sentiment
    desc_texts = (
        prod["description"].fillna("") + " " + prod["features"].fillna("")
    ).str.strip()
    logger.info("[AGG] Running VADER on %d product descriptions ...", len(prod))
    prod["desc_sentiment"] = desc_texts.apply(
        lambda t: float(sia.polarity_scores(t)["compound"]) if t else np.nan
    ).astype(np.float32)

    prod["hype_gap_score"] = (prod["desc_sentiment"] - prod["mean_sentiment"]).astype(np.float32)
    prod["official_vs_community_gap"] = prod["hype_gap_score"].abs().astype(np.float32)

    # Data-driven minimum coverage threshold: 5th pct of non-zero coverages
    cov_cols = [f"coverage_{a}" for a in ASPECTS]
    all_covs = prod[cov_cols].values.flatten()
    nonzero_covs = all_covs[(~np.isnan(all_covs)) & (all_covs > 0)]
    min_cov = float(np.percentile(nonzero_covs, 5)) if len(nonzero_covs) else 0.0
    logger.info("[AGG] min_coverage_threshold (5th pct of non-zero coverages) = %.4f", min_cov)

    # Keyword frequency in features+description per aspect per product
    text_for_kw = (
        prod["features"].fillna("") + " " + prod["description"].fillna("")
    ).str.lower()
    for a in ASPECTS:
        kws = aspect_keywords.get(a, [])
        freq = sum(text_for_kw.str.count(re.escape(kw)) for kw in kws)
        prod[f"_kw_{a}"] = freq.astype(np.float32)

    # Median keyword frequency per aspect (data-driven "high" threshold)
    kw_medians = {a: float(prod[f"_kw_{a}"].median()) for a in ASPECTS}

    def _overhyped(row: pd.Series) -> str:
        eligible = [
            a for a in ASPECTS
            if (not np.isnan(row.get(f"coverage_{a}", np.nan)))
               and row.get(f"coverage_{a}", 0) > min_cov
        ]
        if not eligible:
            return "insufficient data"
        # Score = kw_freq (normalised by median) - community score
        best, best_score = "insufficient data", -np.inf
        for a in eligible:
            score = row.get(f"mean_aspect_{a}", np.nan)
            freq  = row.get(f"_kw_{a}", 0.0)
            if np.isnan(score):
                continue
            kw_med = kw_medians.get(a, 1.0) or 1.0
            s = (freq / kw_med) - score   # high freq + low score = overhyped
            if s > best_score:
                best_score, best = s, a
        return best

    def _underrated(row: pd.Series) -> str:
        eligible = [
            a for a in ASPECTS
            if (not np.isnan(row.get(f"coverage_{a}", np.nan)))
               and row.get(f"coverage_{a}", 0) > min_cov
        ]
        if not eligible:
            return "insufficient data"
        # Highest community score / lowest coverage → underrated
        best, best_val = "insufficient data", -np.inf
        for a in eligible:
            score = row.get(f"mean_aspect_{a}", np.nan)
            cov   = row.get(f"coverage_{a}",   np.nan)
            if np.isnan(score) or np.isnan(cov):
                continue
            # Normalise both, maximise score / coverage
            val = score - cov   # high score but low coverage = underrated
            if val > best_val:
                best_val, best = val, a
        return best

    logger.info("[AGG] Computing overhyped / underrated aspects ...")
    prod["overhyped_aspect"]  = prod.apply(_overhyped,  axis=1)
    prod["underrated_aspect"] = prod.apply(_underrated, axis=1)

    # Drop temp keyword frequency columns
    prod = prod.drop(columns=[f"_kw_{a}" for a in ASPECTS])
    return prod


def _compute_consensus_and_trajectory(products: pd.DataFrame) -> pd.DataFrame:
    """Consensus, controversy, velocity, trajectory, and aspect trend signals."""
    prod = products.copy()

    # --- Consensus score (1 - normalised std_sentiment) ---
    max_std = float(prod["std_sentiment"].max())
    if max_std > 0:
        prod["consensus_score"] = (
            1.0 - prod["std_sentiment"] / max_std
        ).clip(0, 1).astype(np.float32)
    else:
        prod["consensus_score"] = 1.0

    prod["controversy_index"] = prod["std_sentiment"].astype(np.float32)

    # Per-aspect consensus
    for a in ASPECTS:
        std_col = f"std_aspect_{a}"
        if std_col in prod.columns:
            max_asp_std = float(prod[std_col].max())
            if max_asp_std > 0:
                prod[f"{a}_consensus"] = (
                    1.0 - prod[std_col] / max_asp_std
                ).clip(0, 1).astype(np.float32)
            else:
                prod[f"{a}_consensus"] = 1.0
        else:
            prod[f"{a}_consensus"] = np.nan

    # --- Sentiment velocity & trajectory ---
    prod["sentiment_velocity"] = (
        prod["recent_sentiment"] - prod["early_sentiment"]
    ).astype(np.float32)

    vel   = prod["sentiment_velocity"].dropna()
    v_mu  = float(vel.mean())
    v_std = float(vel.std())
    s_mu  = float(prod["std_sentiment"].mean())
    s_std = float(prod["std_sentiment"].std())
    logger.info(
        "[AGG] Velocity: mean=%.4f, std=%.4f | "
        "Controversy std: mean=%.4f, std=%.4f",
        v_mu, v_std, s_mu, s_std
    )

    def _trajectory(row: pd.Series) -> str:
        v   = row.get("sentiment_velocity", np.nan)
        std = row.get("std_sentiment",      np.nan)
        if np.isnan(v) or np.isnan(std):
            return "stable"
        if v > v_mu + v_std:
            return "rising"
        if v < v_mu - v_std:
            return "declining"
        if std > s_mu + s_std:
            return "controversial"
        return "stable"

    prod["sentiment_trajectory"] = prod.apply(_trajectory, axis=1)

    # --- Per-aspect trend & trend summary ---
    for a in ASPECTS:
        early_col  = f"early_aspect_{a}"
        recent_col = f"recent_aspect_{a}"
        if early_col in prod.columns and recent_col in prod.columns:
            prod[f"_delta_{a}"] = (prod[recent_col] - prod[early_col]).astype(np.float32)
        else:
            prod[f"_delta_{a}"] = np.nan

    # Data-driven delta threshold: mean of absolute deltas across all products & aspects
    all_deltas = prod[[f"_delta_{a}" for a in ASPECTS]].values.flatten()
    valid_deltas = all_deltas[~np.isnan(all_deltas)]
    delta_threshold = float(np.mean(np.abs(valid_deltas))) if len(valid_deltas) else 0.05
    logger.info("[AGG] Aspect delta threshold (mean |delta|) = %.4f", delta_threshold)

    for a in ASPECTS:
        col = f"_delta_{a}"
        trend = pd.Series("stable", index=prod.index)
        trend[prod[col] > delta_threshold]  = "improving"
        trend[prod[col] < -delta_threshold] = "declining"
        trend[prod[col].isna()] = "N/A"
        prod[f"{a}_trend"] = trend

    def _trend_summary(row: pd.Series) -> str:
        parts = []
        for a in ASPECTS:
            t = row.get(f"{a}_trend", "N/A")
            d = row.get(f"_delta_{a}", np.nan)
            if t == "improving" and not np.isnan(d):
                parts.append(f"{a.capitalize()} improved (+{d:.2f})")
            elif t == "declining" and not np.isnan(d):
                parts.append(f"{a.capitalize()} declined ({d:.2f})")
        return "; ".join(parts) if parts else "stable"

    prod["aspect_trend_summary"] = prod.apply(_trend_summary, axis=1)
    prod = prod.drop(columns=[f"_delta_{a}" for a in ASPECTS])
    return prod


def _compute_category_signals(products: pd.DataFrame) -> pd.DataFrame:
    """Category percentile ranks, distinctive score, and hidden gem score."""
    prod = products.copy()
    asp_cols = [f"mean_aspect_{a}" for a in ASPECTS]

    grp = prod.groupby("leaf_category")

    # Basic category stats
    prod["category_size"]         = grp["parent_asin"].transform("count").astype(np.int32)
    prod["category_avg_rating"]   = grp["average_rating"].transform("mean").astype(np.float32)
    prod["category_avg_sentiment"]= grp["mean_sentiment"].transform("mean").astype(np.float32)

    # Percentile ranks within category (pandas rank pct=True gives [0,1])
    prod["category_rating_percentile"] = (
        prod.groupby("leaf_category")["average_rating"]
        .rank(pct=True) * 100
    ).astype(np.float32)
    prod["category_sentiment_percentile"] = (
        prod.groupby("leaf_category")["mean_sentiment"]
        .rank(pct=True) * 100
    ).astype(np.float32)

    # Category mean per aspect (used for distinctive score and outperforms)
    for a in ASPECTS:
        col = f"mean_aspect_{a}"
        if col in prod.columns:
            prod[f"category_avg_{a}"] = grp[col].transform("mean").astype(np.float32)
            prod[f"_cat_std_{a}"]     = grp[col].transform("std").fillna(0).astype(np.float32)
        else:
            prod[f"category_avg_{a}"] = np.nan
            prod[f"_cat_std_{a}"]     = np.nan

    # Distinctive score: Euclidean distance of product aspect vector from category mean
    # Fill NaN aspects with the category mean before distance computation
    prod_asp = prod[asp_cols].copy()
    cat_asp  = prod[[f"category_avg_{a}" for a in ASPECTS]].copy()
    cat_asp.columns = asp_cols

    for col in asp_cols:
        nan_mask = prod_asp[col].isna()
        prod_asp.loc[nan_mask, col] = cat_asp.loc[nan_mask, col]

    diff = prod_asp.values - cat_asp.values
    prod["distinctive_score"] = np.sqrt(
        np.nansum(diff ** 2, axis=1)
    ).astype(np.float32)

    # Outperforms / underperforms: product > cat_mean + 0.5 * cat_std (spec-given 0.5)
    def _aspect_list(row: pd.Series, direction: str) -> str:
        out = []
        for a in ASPECTS:
            v    = row.get(f"mean_aspect_{a}", np.nan)
            mu   = row.get(f"category_avg_{a}", np.nan)
            std  = row.get(f"_cat_std_{a}",    0.0)
            if np.isnan(v) or np.isnan(mu):
                continue
            if direction == "over"  and v > mu + 0.5 * std:
                out.append(a)
            if direction == "under" and v < mu - 0.5 * std:
                out.append(a)
        return ", ".join(out) if out else "none"

    prod["outperforms_aspects"]   = prod.apply(lambda r: _aspect_list(r, "over"),  axis=1)
    prod["underperforms_aspects"] = prod.apply(lambda r: _aspect_list(r, "under"), axis=1)

    # Hidden gem score: sentiment_rank_pct - rating_number_rank_pct (within category)
    prod["_sent_rank"]   = prod.groupby("leaf_category")["mean_sentiment"].rank(pct=True)
    prod["_rn_rank"]     = prod.groupby("leaf_category")["rating_number"].rank(pct=True)
    prod["hidden_gem_score"] = (
        (prod["_sent_rank"] - prod["_rn_rank"]) * 100
    ).astype(np.float32)

    # Drop helper columns
    prod = prod.drop(columns=[f"_cat_std_{a}" for a in ASPECTS]
                               + ["_sent_rank", "_rn_rank"])
    return prod


def _compute_tfidf(
    products: pd.DataFrame,
    review_corpus: pd.Series,
    cfg_s2: dict,
) -> pd.DataFrame:
    """Per-category TF-IDF: top phrases and distinctive terms per product."""
    ngram_range  = (int(cfg_s2.get("tfidf_ngram_min",   1)),
                    int(cfg_s2.get("tfidf_ngram_max",    2)))
    max_features = int(cfg_s2.get("tfidf_max_features", 50_000))
    min_df       = int(cfg_s2.get("tfidf_min_df",       2))

    results: dict[str, tuple[str, str]] = {}   # parent_asin -> (phrases, terms)

    # Build a lookup: parent_asin -> corpus text
    corpus_map = review_corpus.to_dict()

    categories = products["leaf_category"].unique()
    logger.info("[AGG] TF-IDF: processing %d categories ...", len(categories))

    for i, cat in enumerate(categories, 1):
        cat_mask = products["leaf_category"] == cat
        cat_asins = products.loc[cat_mask, "parent_asin"].tolist()

        # Products with reviews in this category
        docs  = [corpus_map.get(a, "") for a in cat_asins]
        # Keep only products that have at least some review text
        valid_pairs = [(a, d) for a, d in zip(cat_asins, docs) if d.strip()]

        if not valid_pairs:
            for a in cat_asins:
                results[a] = ("", "")
            continue

        valid_asins, valid_docs = zip(*valid_pairs)

        # Fall back to min_df=1 for very small categories
        eff_min_df = min_df if len(valid_docs) >= min_df else 1

        try:
            vect = TfidfVectorizer(
                ngram_range=ngram_range,
                max_features=max_features,
                min_df=eff_min_df,
                stop_words="english",
                sublinear_tf=True,
            )
            tfidf_mat = vect.fit_transform(valid_docs)   # sparse [n_docs, n_feats]
            feature_names = np.array(vect.get_feature_names_out())

            # Category centroid (mean TF-IDF vector)
            centroid = np.asarray(tfidf_mat.mean(axis=0)).flatten()

            for j, asin in enumerate(valid_asins):
                row_vec = np.asarray(tfidf_mat[j].todense()).flatten()

                # top 10 by raw TF-IDF score
                top10_idx = row_vec.argsort()[-10:][::-1]
                top_phrases = ", ".join(feature_names[top10_idx[row_vec[top10_idx] > 0]])

                # top 5 most distinctive vs category centroid
                diff = row_vec - centroid
                top5_idx = diff.argsort()[-5:][::-1]
                dist_terms = ", ".join(feature_names[top5_idx[diff[top5_idx] > 0]])

                results[asin] = (top_phrases, dist_terms)

        except Exception as exc:
            logger.warning("[AGG] TF-IDF failed for category '%s': %s", cat, exc)
            for a in valid_asins:
                results[a] = ("", "")

        # Products without reviews
        for a in cat_asins:
            if a not in results:
                results[a] = ("", "")

        if i % 100 == 0 or i == len(categories):
            logger.info("[AGG] TF-IDF: %d / %d categories done", i, len(categories))

    tfidf_df = pd.DataFrame([
        {"parent_asin": a, "top_tfidf_phrases": v[0], "distinctive_terms": v[1]}
        for a, v in results.items()
    ])
    return tfidf_df


def _compute_aspect_best_worst(products: pd.DataFrame) -> pd.DataFrame:
    """top_aspect and worst_aspect using data-driven coverage threshold."""
    prod = products.copy()

    cov_cols = [f"coverage_{a}" for a in ASPECTS]
    all_covs = prod[cov_cols].values.flatten()
    nonzero  = all_covs[(~np.isnan(all_covs)) & (all_covs > 0)]
    min_cov  = float(np.percentile(nonzero, 5)) if len(nonzero) else 0.0
    logger.info("[AGG] Aspect best/worst min_coverage_threshold = %.4f", min_cov)

    def _best_worst(row: pd.Series) -> tuple[str, str]:
        eligible = {
            a: row.get(f"mean_aspect_{a}", np.nan)
            for a in ASPECTS
            if (not np.isnan(row.get(f"coverage_{a}", np.nan)))
               and row.get(f"coverage_{a}", 0) > min_cov
        }
        eligible = {a: v for a, v in eligible.items() if not np.isnan(v)}
        if not eligible:
            return "insufficient data", "insufficient data"
        best  = max(eligible, key=lambda a: eligible[a])
        worst = min(eligible, key=lambda a: eligible[a])
        return best, worst

    bw = prod.apply(_best_worst, axis=1, result_type="expand")
    bw.columns = ["top_aspect", "worst_aspect"]
    prod["top_aspect"]   = bw["top_aspect"]
    prod["worst_aspect"] = bw["worst_aspect"]
    return prod


def _compute_dominant_emotion(products: pd.DataFrame) -> pd.DataFrame:
    """dominant_emotion label using data-driven pos/neg separation threshold."""
    prod = products.copy()
    diffs = (prod["pos_ratio"] - prod["neg_ratio"]).abs().dropna()
    threshold = float(diffs.mean()) if len(diffs) else 0.05
    logger.info("[AGG] Dominant emotion threshold (mean |pos-neg|) = %.4f", threshold)

    def _emotion(row: pd.Series) -> str:
        p = row.get("pos_ratio", np.nan)
        n = row.get("neg_ratio", np.nan)
        if np.isnan(p) or np.isnan(n):
            return "mixed"
        if p > n + threshold:
            return "positive"
        if n > p + threshold:
            return "negative"
        return "mixed"

    prod["dominant_emotion"] = prod.apply(_emotion, axis=1)
    return prod


def aggregate_to_products(
    reviews_nlp: pd.DataFrame,
    meta_clean: pd.DataFrame,
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Full product-level NLP aggregation; returns ~90-column product DataFrame."""
    cfg_s2 = (cfg or {}).get("stage2", {})
    aspect_keywords: dict = cfg_s2.get("aspect_keywords", {})
    t0 = time.time()

    # -- phase 1: review-level aggregations --
    _ts("Phase 1a — build review corpus")
    review_corpus = _build_review_corpus(reviews_nlp)

    _ts("Phase 1b — base aggregation (sentiment + helpfulness + verified)")
    base = _agg_base(reviews_nlp)

    _ts("Phase 1c — temporal windows")
    temp = _agg_temporal(reviews_nlp, cfg_s2)

    _ts("Phase 1d — emotion vocabulary")
    emo = _agg_emotion_vocab(reviews_nlp)

    # Merge phase-1 frames
    agg = base
    for frame in [temp, emo]:
        agg = agg.merge(frame, on="parent_asin", how="left")

    # -- phase 2: join with meta --
    _ts("Phase 2 — join with meta_clean")
    products = meta_clean.merge(agg, on="parent_asin", how="left")
    products["review_count"] = products["review_count"].fillna(0).astype(np.int32)

    # -- phase 3: product-level computations --
    _ts("Phase 3a — listing quality (leaf_category, feature_count, …)")
    products = _compute_listing_quality(products)

    _ts("Phase 3b — price tier (data-driven percentiles)")
    products = _compute_price_tier(products)

    _ts("Phase 3c — reality gap signals (desc_sentiment, hype_gap, …)")
    products = _compute_reality_gap(products, aspect_keywords)

    _ts("Phase 3d — consensus + trajectory (data-driven thresholds)")
    products = _compute_consensus_and_trajectory(products)

    _ts("Phase 3e — category comparative signals")
    products = _compute_category_signals(products)

    _ts("Phase 3f — TF-IDF collective vocabulary (per-category)")
    tfidf_df = _compute_tfidf(products, review_corpus, cfg_s2)
    products = products.merge(tfidf_df, on="parent_asin", how="left")

    _ts("Phase 3g — aspect best / worst")
    products = _compute_aspect_best_worst(products)

    _ts("Phase 3h — dominant emotion (data-driven threshold)")
    products = _compute_dominant_emotion(products)

    # ── Drop std_aspect_* (kept for consensus computation, not needed in output)
    std_asp_cols = [c for c in products.columns if c.startswith("std_aspect_")]
    products = products.drop(columns=std_asp_cols, errors="ignore")

    elapsed = time.time() - t0
    logger.info("[AGG] aggregate_to_products complete in %.1f s  (%d products)",
                elapsed, len(products))
    return products.reset_index(drop=True)
