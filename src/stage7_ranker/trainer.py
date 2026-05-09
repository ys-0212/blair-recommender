"""Stage 7 — LightGBM LambdaRank trainer on Stage 6 feature files."""

from __future__ import annotations

import json
import logging
import random as _random
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on Windows without display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

_FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "f01_faiss_score":                     "FAISS Score",
    "f02_faiss_rank":                      "FAISS Rank",
    "f03_query_item_cosine":               "Query-Item Cosine",
    "f04_user_uniform_cosine":             "User-Item Cosine (Uniform)",
    "f05_avg_rating":                      "Avg Rating",
    "f06_rating_count_log":                "Rating Count (log)",
    "f07_review_count_log":                "Review Count (log)",
    "f08_mean_sentiment":                  "Mean Sentiment",
    "f09_price_normalized":                "Price",
    "f10_hidden_gem_score":                "Hidden Gem Score",
    "f11_controversy_index":               "Controversy Index",
    "f12_desc_richness":                   "Description Richness",
    "f13_aspect_gameplay":                 "Aspect: Gameplay",
    "f14_aspect_graphics":                 "Aspect: Graphics",
    "f15_aspect_story":                    "Aspect: Story",
    "f16_aspect_controls":                 "Aspect: Controls",
    "f17_aspect_value":                    "Aspect: Value",
    "f18_user_item_voice_cosine":          "User Voice Cosine",
    "f19_category_match":                  "Category Match",
    "f20_price_tier_match":                "Price Tier Match",
    "f21_user_avg_sentiment_gap":          "Sentiment Gap",
    "f22_top_aspect_match":                "Top Aspect Match",
    "f23_interaction_count_log":           "Interaction Count (log)",
    "f24_sentiment_trajectory":            "Sentiment Trajectory",
    "f25_verified_ratio":                  "Verified Purchase Ratio",
    "f26_helpfulness_weighted_sentiment":  "Helpfulness-Weighted Sentiment",
    "f28_user_recency_cosine":             "User-Item Cosine (Recency)",
    "f29_user_rating_cosine":              "User-Item Cosine (Rating)",
    "f30_user_combined_cosine":            "User-Item Cosine (Combined)",
    "f31_bm25_score":                      "BM25 Score",
}

_GROUP_COLORS: dict[str, str] = {
    "retrieval":        "#2196F3",   # blue
    "product":          "#4CAF50",   # green
    "aspect":           "#FF9800",   # orange
    "personalization":  "#F44336",   # red
    "temporal_quality": "#9C27B0",   # purple
}

def _feature_group(fname: str) -> str:
    n = int(fname[1:3])
    if n <= 4:              return "retrieval"
    if n <= 12:             return "product"
    if n <= 17:             return "aspect"
    if n <= 23:             return "personalization"
    if n <= 26:             return "temporal_quality"
    if fname == "f31_bm25_score": return "retrieval"
    return "personalization"   # f28-f30 are user embedding cosines


def load_split_chunked(
    path: Path,
    feature_cols: list[str],
    label_col: str = "relevance_label",
    query_col: str = "query_parent_asin",
    user_col:  str = "user_id",
    sample_ratio: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Chunked parquet loader with optional row-group sampling to cap peak RAM.

    sample_ratio=1.0 loads all row groups (default).
    sample_ratio=0.20 loads ~20% of row groups, reducing peak RAM from ~14 GB
    to ~2.8 GB for 125M-row training files. Row group order is preserved so
    the carry-over group-count logic remains correct.
    """
    t0 = time.time()
    pf = pq.ParquetFile(str(path))
    n_rg       = pf.metadata.num_row_groups
    total_rows = pf.metadata.num_rows

    # Decide which row groups to load, preserving order for carry-over logic
    rng = _random.Random(seed)
    if sample_ratio < 1.0:
        selected_rgs = [rg for rg in range(n_rg) if rng.random() < sample_ratio]
        if not selected_rgs:
            selected_rgs = [0]  # always load at least one group
        logger.info(
            "Loading %s: sampling %.0f%% -> %d / %d row groups (~%.0fM / %.0fM rows)",
            path.name, sample_ratio * 100, len(selected_rgs), n_rg,
            total_rows * len(selected_rgs) / max(n_rg, 1) / 1e6, total_rows / 1e6,
        )
    else:
        selected_rgs = list(range(n_rg))
        logger.info(
            "Loading %s: %d row groups, %d rows total",
            path.name, n_rg, total_rows,
        )

    read_cols = [user_col, query_col, label_col] + feature_cols

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_counts: list[int]   = []

    # Carry-over: last query key + partial count across row group boundaries
    prev_key:   str | None = None
    prev_count: int        = 0

    for load_idx, rg_idx in enumerate(selected_rgs):
        table = pf.read_row_group(rg_idx, columns=read_cols)
        df    = table.to_pandas()

        df["_qkey"] = df[user_col].astype(str) + "_" + df[query_col].astype(str)

        X_parts.append(df[feature_cols].values.astype(np.float32))
        y_parts.append(df[label_col].values.astype(np.float32))

        # Count consecutive query group sizes
        keys = df["_qkey"].values
        pos  = 0
        n    = len(keys)
        while pos < n:
            k = keys[pos]
            if k == prev_key:
                # Continue counting the carry-over group
                j = pos
                while j < n and keys[j] == k:
                    j += 1
                prev_count += j - pos
                pos = j
            else:
                # Flush completed group, start new one
                if prev_key is not None:
                    group_counts.append(prev_count)
                j = pos
                while j < n and keys[j] == k:
                    j += 1
                prev_key   = k
                prev_count = j - pos
                pos = j

        if (load_idx + 1) % 20 == 0 or (load_idx + 1) == len(selected_rgs):
            logger.info(
                "  loaded %d / %d row groups", load_idx + 1, len(selected_rgs)
            )

    # Flush the last group
    if prev_key is not None:
        group_counts.append(prev_count)

    X      = np.concatenate(X_parts, axis=0)
    y      = np.concatenate(y_parts, axis=0)
    groups = np.array(group_counts, dtype=np.int32)

    logger.info(
        "Loaded %s: X=%s  y=%s  groups=%d  sum(groups)=%d  %.0fs",
        path.name, X.shape, y.shape, len(groups), groups.sum(), time.time() - t0,
    )
    assert groups.sum() == len(X), (
        f"Group sum {groups.sum()} != row count {len(X)} in {path.name}"
    )
    return X, y, groups


def train_lambdarank(
    train_path: Path,
    valid_path: Path,
    cfg: dict,
    output_dir: Path,
) -> lgb.Booster:
    """Train LambdaRank on chunked train features and validate on valid features."""
    cfg7         = cfg.get("stage7", {})
    feature_cols = list(cfg7.get("feature_cols", []))
    if not feature_cols:
        raise ValueError("stage7.feature_cols is empty in config.yaml")

    sample_ratio = float(cfg7.get("train_sample_ratio", 1.0))
    seed         = int(cfg.get("project", {}).get("seed", 42))

    logger.info(
        "Loading training data (chunked, sample_ratio=%.0f%%) ...",
        sample_ratio * 100,
    )
    X_train, y_train, g_train = load_split_chunked(
        train_path, feature_cols, sample_ratio=sample_ratio, seed=seed
    )

    logger.info("Loading validation data (full) ...")
    X_valid, y_valid, g_valid = load_split_chunked(
        valid_path, feature_cols, sample_ratio=1.0, seed=seed
    )

    logger.info("Building LightGBM datasets ...")
    train_ds = lgb.Dataset(
        X_train, label=y_train, group=g_train,
        feature_name=feature_cols, free_raw_data=False,
    )
    valid_ds = lgb.Dataset(
        X_valid, label=y_valid, group=g_valid,
        feature_name=feature_cols, reference=train_ds, free_raw_data=False,
    )

    ndcg_at = cfg7.get("ndcg_eval_at", [1, 5, 10])
    params  = {
        "objective":                    cfg7.get("objective",          "lambdarank"),
        "metric":                       cfg7.get("metric",             "ndcg"),
        "ndcg_eval_at":                 ndcg_at,
        "learning_rate":                float(cfg7.get("learning_rate",    0.05)),
        "max_depth":                    int(cfg7.get("max_depth",          7)),
        "num_leaves":                   int(cfg7.get("num_leaves",         63)),
        "min_child_samples":            int(cfg7.get("min_child_samples",  10)),
        "subsample":                    float(cfg7.get("subsample",         0.8)),
        "colsample_bytree":             float(cfg7.get("colsample_bytree",  0.8)),
        "lambdarank_truncation_level":  int(cfg7.get("lambdarank_truncation_level", 20)),
        "n_jobs":                       int(cfg7.get("n_jobs",             -1)),
        "importance_type":              "gain",
        "verbose":                      -1,
    }
    # Graded relevance: labels 0-5 → gains 0, 1, 3, 7, 15, 31
    if cfg.get("stage6", {}).get("graded_relevance", False):
        params["label_gain"] = [0, 1, 3, 7, 15, 31]
        logger.info("Graded relevance enabled: label_gain=%s", params["label_gain"])

    n_estimators        = int(cfg7.get("n_estimators",        1000))
    early_stopping_rds  = int(cfg7.get("early_stopping_rounds", 50))

    logger.info("Training LambdaRank: n_estimators=%d  lr=%.3f  leaves=%d",
                n_estimators, params["learning_rate"], params["num_leaves"])
    logger.info(
        "Note: ~95.9%% of positives are forced (FAISS natural recall ~4%%). "
        "LambdaRank still learns valid re-ranking signal from feature patterns."
    )

    evals_result: dict = {}
    callbacks = [
        lgb.early_stopping(stopping_rounds=early_stopping_rds, verbose=True),
        lgb.log_evaluation(period=10),
        lgb.record_evaluation(evals_result),
    ]

    t_train = time.time()
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=n_estimators,
        valid_sets=[train_ds, valid_ds],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    elapsed = time.time() - t_train
    logger.info("Training complete in %.0f s (%d rounds)", elapsed, model.num_trees())

    model_path = output_dir / cfg7.get("model_path", "outputs/results/lambdarank_model.lgb").split("/")[-1]
    # Resolve full path from config
    from src.utils.config import PROJECT_ROOT
    model_full = PROJECT_ROOT / cfg7.get("model_path", "outputs/results/lambdarank_model.lgb")
    model_full.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_full))
    logger.info("Saved model: %s", model_full)

    history_full = PROJECT_ROOT / cfg7.get(
        "training_history", "outputs/results/training_history.json"
    )
    history_full.parent.mkdir(parents=True, exist_ok=True)

    # Build readable history dict
    history: dict[str, Any] = {
        "best_iteration": model.best_iteration,
        "n_trees":        model.num_trees(),
        "training_time_s": round(elapsed, 1),
        "params":          params,
        "evals_result":   evals_result,
    }
    # Add final scores summary for quick reference
    summary: dict[str, dict[str, float]] = {}
    for split_name, split_metrics in evals_result.items():
        summary[split_name] = {}
        for metric_key, values in split_metrics.items():
            summary[split_name][metric_key] = {
                "best":  float(max(values)),
                "final": float(values[-1]),
                "best_iter": int(values.index(max(values)) + 1),
            }
    history["summary"] = summary

    history_full.write_text(
        json.dumps(history, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved training history: %s", history_full)

    # Log best validation metrics
    for metric_key in (evals_result.get("valid") or {}).keys():
        vals  = evals_result["valid"][metric_key]
        best  = max(vals)
        bi    = vals.index(best) + 1
        logger.info("  Best valid %s = %.4f at round %d", metric_key, best, bi)

    return model


def plot_feature_importance(
    model: lgb.Booster,
    feature_cols: list[str],
    output_path: Path,
) -> None:
    """Horizontal bar chart of feature importances (gain), coloured by feature group."""
    importances = model.feature_importance(importance_type="gain")
    names       = model.feature_name()

    # Build display-name and colour arrays in importance order
    idx_sorted  = np.argsort(importances)  # ascending so horizontal bars go large→top
    sorted_imp  = importances[idx_sorted]
    sorted_names = [names[i] for i in idx_sorted]

    display_names = [
        _FEATURE_DISPLAY_NAMES.get(n, n) for n in sorted_names
    ]
    colors = [
        _GROUP_COLORS[_feature_group(n)] for n in sorted_names
    ]

    fig, ax = plt.subplots(figsize=(12, 10))
    bars = ax.barh(display_names, sorted_imp, color=colors, edgecolor="white", height=0.7)

    # Legend
    legend_patches = [
        mpatches.Patch(color=_GROUP_COLORS["retrieval"],        label="Retrieval (f01-f04, f31)"),
        mpatches.Patch(color=_GROUP_COLORS["product"],          label="Product (f05-f12)"),
        mpatches.Patch(color=_GROUP_COLORS["aspect"],           label="Aspect (f13-f17)"),
        mpatches.Patch(color=_GROUP_COLORS["personalization"],  label="Personalization (f18-f23, f28-f30)"),
        mpatches.Patch(color=_GROUP_COLORS["temporal_quality"], label="Temporal/Quality (f24-f26)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=9)

    ax.set_title("LambdaRank Feature Importance (Gain)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance (Gain)", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=9)

    # Value labels on bars
    for bar, val in zip(bars, sorted_imp):
        if val > 0:
            ax.text(
                val + sorted_imp.max() * 0.005,
                bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f}",
                va="center", ha="left", fontsize=7, color="#333333",
            )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved feature importance chart: %s", output_path)
