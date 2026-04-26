"""Stage 9 — ablation study: zeros one feature group at a time and measures NDCG@10 drop."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT, ensure_dirs, get_path, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

ABLATION_CONFIGS: list[dict[str, Any]] = [
    {
        "id":   "full",
        "name": "Full System",
        "zero": [],
    },
    {
        "id":   "no_retrieval",
        "name": "w/o Retrieval Features",
        "zero": ["f01_faiss_score", "f02_faiss_rank",
                 "f03_query_item_cosine", "f04_user_uniform_cosine"],
    },
    {
        "id":   "no_product",
        "name": "w/o Product Features",
        "zero": ["f05_avg_rating", "f06_rating_count_log", "f07_review_count_log",
                 "f08_mean_sentiment", "f09_price_normalized",
                 "f10_hidden_gem_score", "f11_controversy_index", "f12_desc_richness"],
    },
    {
        "id":   "no_aspect",
        "name": "w/o Aspect Features",
        "zero": ["f13_aspect_gameplay", "f14_aspect_graphics", "f15_aspect_story",
                 "f16_aspect_controls", "f17_aspect_value"],
    },
    {
        "id":   "no_personalization",
        "name": "w/o Personalization",
        "zero": ["f18_user_item_voice_cosine", "f19_category_match",
                 "f20_price_tier_match", "f21_user_avg_sentiment_gap",
                 "f22_top_aspect_match", "f23_interaction_count_log"],
    },
    {
        "id":   "no_temporal",
        "name": "w/o Temporal/Quality",
        "zero": ["f24_sentiment_trajectory", "f25_verified_ratio",
                 "f26_helpfulness_weighted_sentiment"],
    },
    {
        "id":   "no_nlp",
        "name": "w/o ALL NLP Features",
        "zero": [
            "f08_mean_sentiment",
            "f10_hidden_gem_score", "f11_controversy_index", "f12_desc_richness",
            "f13_aspect_gameplay", "f14_aspect_graphics", "f15_aspect_story",
            "f16_aspect_controls", "f17_aspect_value",
            "f24_sentiment_trajectory", "f25_verified_ratio",
            "f26_helpfulness_weighted_sentiment",
        ],
    },
    {
        "id":   "no_voice",
        "name": "w/o User Voice Cosine",
        "zero": ["f18_user_item_voice_cosine"],
    },
    {
        "id":   "no_bm25",
        "name": "w/o BM25 Score",
        "zero": ["f31_bm25_score"],
    },
    {
        "id":   "no_user_embeddings",
        "name": "w/o All User Embedding Cosines",
        "zero": ["f04_user_uniform_cosine", "f28_user_recency_cosine",
                 "f29_user_rating_cosine", "f30_user_combined_cosine"],
    },
]


def _ndcg_at_k(sorted_labels: np.ndarray, k: int) -> float:
    rels  = sorted_labels[:k].astype(float)
    dcg   = float(np.sum((2.0 ** rels - 1.0) / np.log2(np.arange(2, len(rels) + 2))))
    ideal = np.sort(sorted_labels.astype(float))[::-1][:k]
    idcg  = float(np.sum((2.0 ** ideal - 1.0) / np.log2(np.arange(2, len(ideal) + 2))))
    return dcg / idcg if idcg > 0 else 0.0


def _evaluate_config(
    labels: np.ndarray,
    scores: np.ndarray,
    group_indices: dict[Any, np.ndarray],
    k_values: list[int],
) -> dict[str, float]:
    """Vectorised per-query NDCG/HR/MRR/Recall over pre-computed group indices."""
    ndcg_sums  = {k: 0.0 for k in k_values}
    hr_sums    = {k: 0.0 for k in k_values}
    mrr_total  = 0.0
    rec10      = 0.0
    evaluated  = 0

    for idx in group_indices.values():
        grp_labels = labels[idx]
        if grp_labels.sum() == 0:
            continue

        grp_scores = scores[idx]
        order      = np.argsort(grp_scores)[::-1]
        sl         = grp_labels[order]

        for k in k_values:
            ndcg_sums[k] += _ndcg_at_k(sl, k)
            hr_sums[k]   += 1.0 if sl[:k].sum() > 0 else 0.0

        pos_ranks = np.where(sl > 0)[0]
        mrr_total += 1.0 / (pos_ranks[0] + 1) if len(pos_ranks) > 0 else 0.0
        rec10     += float(sl[:10].sum()) / max(int(grp_labels.sum()), 1)
        evaluated += 1

    n = max(evaluated, 1)
    result: dict[str, float] = {}
    for k in k_values:
        result[f"ndcg@{k}"] = ndcg_sums[k] / n
        result[f"hr@{k}"]   = hr_sums[k]   / n
    result["mrr"]        = mrr_total / n
    result["recall@10"]  = rec10     / n
    result["evaluated_queries"] = evaluated
    return result


def run_ablations(
    X: np.ndarray,
    labels: np.ndarray,
    group_indices: dict[Any, np.ndarray],
    model: lgb.Booster,
    feature_cols: list[str],
    k_values: list[int],
) -> list[dict[str, Any]]:
    """Zeros each config's feature columns, predicts, computes metrics. group_indices pre-computed once."""
    feat_idx = {f: i for i, f in enumerate(feature_cols)}
    results  = []

    for cfg_item in ABLATION_CONFIGS:
        name      = cfg_item["name"]
        zero_cols = cfg_item["zero"]

        X_abl = X.copy()
        for col in zero_cols:
            if col in feat_idx:
                X_abl[:, feat_idx[col]] = 0.0

        t0     = time.time()
        scores = model.predict(X_abl).astype(np.float32)
        m      = _evaluate_config(labels, scores, group_indices, k_values)
        elapsed = time.time() - t0

        logger.info("%-35s NDCG@10=%.4f  MRR=%.4f  (%.0fs)", name, m.get("ndcg@10", 0), m["mrr"], elapsed)
        results.append({
            "id":           cfg_item["id"],
            "name":         name,
            "zeroed":       zero_cols,
            "metrics":      m,
            "elapsed_s":    round(elapsed, 1),
        })

    return results


def _print_ablation_table(results: list[dict], full_ndcg10: float) -> None:
    col_w = 10
    name_w = 36

    def _sep(l: str, m: str, r: str) -> str:
        return l + "═" * name_w + m + "═" * col_w + m + "═" * col_w + r

    def _row(name: str, ndcg: float, drop: str) -> str:
        return "║" + name.ljust(name_w) + "║" + f"{ndcg:.4f}".center(col_w) + "║" + drop.center(col_w) + "║"

    print()
    print(_sep("╔", "╦", "╗"))
    print("║" + " Configuration".ljust(name_w) + "║" + "NDCG@10".center(col_w) + "║" + "Drop".center(col_w) + "║")
    print(_sep("╠", "╬", "╣"))

    for r in results:
        ndcg10 = r["metrics"].get("ndcg@10", 0.0)
        if r["id"] == "full":
            drop_str = "   —  "
        else:
            pct = (ndcg10 - full_ndcg10) / max(full_ndcg10, 1e-9) * 100.0
            drop_str = f"{pct:+.2f}%"
        print(_row(" " + r["name"], ndcg10, drop_str))

    print(_sep("╚", "╩", "╝"))
    print()


def _plot_ablation(results: list[dict], full_ndcg10: float, output_path: Path) -> None:
    names  = [r["name"] for r in results]
    ndcgs  = [r["metrics"].get("ndcg@10", 0.0) for r in results]
    colors = ["#4CAF50" if r["id"] == "full" else "#F44336" for r in results]

    fig, ax = plt.subplots(figsize=(12, 6))
    y_pos   = np.arange(len(names))
    bars    = ax.barh(y_pos, ndcgs, color=colors, edgecolor="white", height=0.6)

    # Annotate bars with value + drop
    for bar, r, ndcg in zip(bars, results, ndcgs):
        if r["id"] == "full":
            label = f"{ndcg:.4f}"
        else:
            pct   = (ndcg - full_ndcg10) / max(full_ndcg10, 1e-9) * 100.0
            label = f"{ndcg:.4f}  ({pct:+.2f}%)"
        ax.text(
            max(ndcg + 0.002, 0.002),
            bar.get_y() + bar.get_height() / 2,
            label, va="center", ha="left", fontsize=8,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("NDCG@10 (validation set)", fontsize=11)
    ax.set_title("Ablation Study — Feature Group Contribution", fontsize=13, fontweight="bold")
    ax.axvline(x=full_ndcg10, color="#4CAF50", linestyle="--", linewidth=1.2, alpha=0.7, label="Full System")

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#4CAF50", label="Full System"),
        Patch(color="#F44336", label="Ablated"),
    ], loc="lower right", fontsize=9)

    ax.set_xlim(0, min(1.1, max(ndcgs) * 1.15))
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ablation chart: %s", output_path)


def _update_progress() -> None:
    prog_path = PROJECT_ROOT / "PROGRESS.md"
    if not prog_path.exists():
        return
    text = prog_path.read_text(encoding="utf-8")
    for old in [
        "| 9 | Ablation Study | ⬜ Not started |",
        "| 9 | Ablation Study | 🟡 Code complete — ready to run |",
    ]:
        if old in text:
            text = text.replace(old, "| 9 | Ablation Study | ✅ Complete |")
            break
    prog_path.write_text(text, encoding="utf-8")
    logger.info("Updated PROGRESS.md - Stage 9 marked complete")


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    cfg7         = cfg.get("stage7", {})
    feature_cols = list(cfg7.get("feature_cols", []))
    k_values     = [1, 5, 10]
    proc         = get_path(cfg, "data_processed")
    results_dir  = get_path(cfg, "outputs_results")
    charts_dir   = get_path(cfg, "outputs_charts")

    model_path = PROJECT_ROOT / cfg7.get("model_path", "outputs/results/lambdarank_model.lgb")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}  Run Stage 7 first.")

    valid_path = proc / "features_valid.parquet"
    if not valid_path.exists():
        raise FileNotFoundError(f"features_valid.parquet not found. Run Stage 6 first.")

    model = lgb.Booster(model_file=str(model_path))
    logger.info("Loaded model: %d trees", model.num_trees())

    t0 = time.time()
    df = pd.read_parquet(valid_path)
    logger.info("Loaded features_valid: %d rows x %d cols in %.0fs", len(df), df.shape[1], time.time() - t0)

    X      = df[feature_cols].values.astype(np.float32)
    labels = df["relevance_label"].values.astype(np.float32)

    # Build query key + pre-compute group indices ONCE (most expensive step)
    logger.info("Pre-computing query groups ...")
    df["_qkey"] = df["user_id"].astype(str) + "_" + df["query_parent_asin"].astype(str)
    group_indices = {k: np.asarray(v) for k, v in df.groupby("_qkey", sort=False).indices.items()}
    logger.info("Pre-computed %d query groups in %.0fs", len(group_indices), time.time() - t0)

    logger.info("Running %d ablation configurations ...", len(ABLATION_CONFIGS))
    results = run_ablations(X, labels, group_indices, model, feature_cols, k_values)

    full_ndcg10 = results[0]["metrics"].get("ndcg@10", 0.0)  # first config is always "full"

    print("stage9 ablation results (valid set, NDCG@10):")
    _print_ablation_table(results, full_ndcg10)
    print("Larger negative drop = more important feature group.")
    print()

    # Save JSON
    out_path = results_dir / "ablation_results.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved ablation_results.json: %s", out_path)

    # Plot
    chart_path = charts_dir / "ablation_chart.png"
    _plot_ablation(results, full_ndcg10, chart_path)

    _update_progress()
    print("stage9 complete:")
    print(f"  results: {out_path}")
    print(f"  chart:   {chart_path}")


if __name__ == "__main__":
    run()
