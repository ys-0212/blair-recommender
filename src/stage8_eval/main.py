"""Stage 8 — compares Random Baseline, BM25 Baseline, FAISS Baseline, and LambdaRank on the test set."""

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
from src.stage7_ranker.predictor import compute_ndcg_at_k

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _evaluate_df(
    df: pd.DataFrame,
    score_col: str,
    k_values: list[int],
    system_name: str,
    bootstrap_iterations: int = 0,
    bootstrap_sample_size: int = 10000,
    seed: int = 42,
) -> dict[str, Any]:
    """Macro-averaged NDCG@k/HR@k/MRR/Recall@10 over queries with at least one positive.

    If bootstrap_iterations > 0, also computes bootstrap std for NDCG@10.
    """
    t0 = time.time()
    query_metrics: list[dict[str, float]] = []
    total = skipped = 0

    for _qk, grp in df.groupby("_qkey", sort=False):
        total += 1
        if grp["relevance_label"].sum() == 0:
            skipped += 1
            continue

        ranked = grp.sort_values(score_col, ascending=False).reset_index(drop=True)
        m: dict[str, float] = {}

        for k in k_values:
            m[f"ndcg@{k}"] = compute_ndcg_at_k(ranked, k)
            m[f"hr@{k}"]   = 1.0 if ranked["relevance_label"].values[:k].sum() > 0 else 0.0

        pos_idx = np.where(ranked["relevance_label"].values > 0)[0]
        m["mrr"] = 1.0 / (pos_idx[0] + 1) if len(pos_idx) > 0 else 0.0

        n_rel = int(ranked["relevance_label"].sum())
        m["recall@10"] = float(ranked["relevance_label"].values[:10].sum()) / max(n_rel, 1)

        query_metrics.append(m)

    evaluated = len(query_metrics)
    elapsed   = time.time() - t0

    result: dict[str, Any] = {}
    if evaluated > 0:
        for key in query_metrics[0]:
            result[key] = float(np.mean([m[key] for m in query_metrics]))
    else:
        for k in k_values:
            result[f"ndcg@{k}"] = 0.0
            result[f"hr@{k}"]   = 0.0
        result["mrr"] = result["recall@10"] = 0.0

    # Bootstrap confidence interval for NDCG@10
    if bootstrap_iterations > 0 and evaluated > 0:
        ndcg10_vals = np.array([m["ndcg@10"] for m in query_metrics])
        rng = np.random.default_rng(seed)
        n_sample = min(bootstrap_sample_size, evaluated)
        bs_means = []
        for _ in range(bootstrap_iterations):
            sample = rng.choice(ndcg10_vals, size=n_sample, replace=True)
            bs_means.append(float(np.mean(sample)))
        result["ndcg@10_std"] = float(np.std(bs_means))
    else:
        result["ndcg@10_std"] = 0.0

    result.update({
        "total_queries":     total,
        "evaluated_queries": evaluated,
        "skipped_queries":   skipped,
        "elapsed_s":         round(elapsed, 1),
    })

    logger.info(
        "%s: %d/%d evaluated | NDCG@10=%.4f (+/-%.4f)  MRR=%.4f  HR@10=%.4f  (%.0fs)",
        system_name, evaluated, total,
        result.get("ndcg@10", 0.0), result.get("ndcg@10_std", 0.0),
        result["mrr"], result.get("hr@10", 0.0), elapsed,
    )
    return result


def _print_results_table(results: dict[str, dict], k_values: list[int]) -> None:
    col_w = 10
    sys_w = 22

    metric_keys   = [f"ndcg@{k}" for k in k_values] + ["mrr", f"hr@{max(k_values)}"]
    metric_labels = [f"NDCG@{k}" for k in k_values] + ["MRR",  f"HR@{max(k_values)}"]
    n = len(metric_keys)

    def _row(name: str, vals: list[str]) -> str:
        return "║" + name.ljust(sys_w) + "║" + "║".join(v.center(col_w) for v in vals) + "║"

    def _sep(l: str, m: str, r: str) -> str:
        return l + "═" * sys_w + m + m.join("═" * col_w for _ in range(n)) + r

    system_order = [
        ("random_baseline", " Random Baseline"),
        ("bm25_baseline",   " BM25 Baseline"),
        ("faiss_baseline",  " FAISS Baseline"),
        ("lambdarank",      " LambdaRank (Ours)"),
    ]

    print()
    print(_sep("╔", "╦", "╗"))
    print(_row(" System", [lbl.center(col_w) for lbl in metric_labels]))
    print(_sep("╠", "╬", "╣"))
    for key, label in system_order:
        if key in results:
            vals = [f"{results[key].get(mk, 0.0):.4f}" for mk in metric_keys]
            # Append CI for NDCG@10
            if key in results and results[key].get("ndcg@10_std", 0.0) > 0:
                ci = f" +/-{results[key]['ndcg@10_std']:.4f}"
                vals[0] = vals[0] + ci[:col_w - len(vals[0])]
            print(_row(label, vals))
    print(_sep("╚", "╩", "╝"))
    print()


def _plot_comparison(
    results: dict[str, dict],
    k_values: list[int],
    output_path: Path,
) -> None:
    metric_keys   = [f"ndcg@{k}" for k in k_values]
    metric_labels = [f"NDCG@{k}" for k in k_values]

    system_info = [
        ("random_baseline", "Random Baseline",   "#9E9E9E"),
        ("bm25_baseline",   "BM25 Baseline",     "#FF9800"),
        ("faiss_baseline",  "FAISS Semantic",    "#2196F3"),
        ("lambdarank",      "LambdaRank (Ours)", "#F44336"),
    ]

    x     = np.arange(len(metric_labels))
    width = 0.20
    fig, ax = plt.subplots(figsize=(11, 6))

    for i, (key, label, color) in enumerate(system_info):
        if key not in results:
            continue
        vals = [results[key].get(mk, 0.0) for mk in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=label, color=color,
                      alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xlabel("Metric", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("System Comparison — Test Set (NDCG@k)", fontsize=13, fontweight="bold")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison chart: %s", output_path)


def _update_progress() -> None:
    prog_path = PROJECT_ROOT / "PROGRESS.md"
    if not prog_path.exists():
        return
    text = prog_path.read_text(encoding="utf-8")
    for old in [
        "| 8 | Evaluation | ⬜ Not started |",
        "| 8 | Evaluation | 🟡 Code complete — ready to run |",
    ]:
        if old in text:
            text = text.replace(old, "| 8 | Evaluation | ✅ Complete |")
            break
    prog_path.write_text(text, encoding="utf-8")
    logger.info("Updated PROGRESS.md - Stage 8 marked complete")


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    cfg7         = cfg.get("stage7", {})
    cfg8         = cfg.get("stage8", {})
    feature_cols = list(cfg7.get("feature_cols", []))
    k_values     = list(cfg8.get("k_values", [1, 5, 10]))
    bs_iters     = int(cfg8.get("bootstrap_iterations", 1000))
    bs_size      = int(cfg8.get("bootstrap_sample_size", 10000))
    seed         = int(cfg.get("project", {}).get("seed", 42))
    proc         = get_path(cfg, "data_processed")
    results_dir  = get_path(cfg, "outputs_results")
    charts_dir   = get_path(cfg, "outputs_charts")

    model_path = PROJECT_ROOT / cfg7.get("model_path", "outputs/results/lambdarank_model.lgb")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}  Run Stage 7 first.")

    test_path = proc / "features_test.parquet"
    if not test_path.exists():
        raise FileNotFoundError(f"features_test.parquet not found. Run Stage 6 first.")

    model = lgb.Booster(model_file=str(model_path))
    logger.info("Loaded model: %s  (%d trees)", model_path.name, model.num_trees())

    t0 = time.time()
    df = pd.read_parquet(test_path)
    logger.info("Loaded features_test: %d rows x %d cols in %.0fs", len(df), df.shape[1], time.time() - t0)

    df["_qkey"] = df["user_id"].astype(str) + "_" + df["query_parent_asin"].astype(str)

    # Random baseline (reproducible)
    np.random.seed(seed)
    df["random_score"] = np.random.rand(len(df)).astype(np.float32)

    # LambdaRank scores
    logger.info("Computing LambdaRank scores on %d rows ...", len(df))
    X = df[feature_cols].values.astype(np.float32)
    df["lambdarank_score"] = model.predict(X).astype(np.float32)
    del X

    eval_kwargs = dict(
        k_values             = k_values,
        bootstrap_iterations = bs_iters,
        bootstrap_sample_size= bs_size,
        seed                 = seed,
    )

    all_results: dict[str, dict] = {}
    all_results["random_baseline"] = _evaluate_df(df, "random_score",      system_name="Random Baseline",  **eval_kwargs)
    # BM25 baseline (f31_bm25_score — may be zeros if BM25 not computed in stage6)
    if "f31_bm25_score" in df.columns:
        all_results["bm25_baseline"] = _evaluate_df(df, "f31_bm25_score",  system_name="BM25 Baseline",    **eval_kwargs)
    all_results["faiss_baseline"]  = _evaluate_df(df, "f01_faiss_score",   system_name="FAISS Baseline",   **eval_kwargs)
    all_results["lambdarank"]      = _evaluate_df(df, "lambdarank_score",  system_name="LambdaRank (Ours)",**eval_kwargs)

    print("stage8 test set results:")
    _print_results_table(all_results, k_values)

    eval_q  = all_results["faiss_baseline"].get("evaluated_queries", 0)
    total_q = all_results["faiss_baseline"].get("total_queries", 0)
    skipped = all_results["faiss_baseline"].get("skipped_queries", 0)
    print(f"Queries: {eval_q:,} / {total_q:,} evaluated ({skipped:,} skipped — no positive in candidate set)")
    print()

    out_path = results_dir / "test_results.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved test_results.json: %s", out_path)

    v1_backup = PROJECT_ROOT / "blair-v1-backup" / "test_results.json"
    if v1_backup.exists():
        v1 = json.loads(v1_backup.read_text(encoding="utf-8"))
        v2_lr = all_results.get("lambdarank", {})
        v1_lr = v1.get("lambdarank", {})

        print()
        print("=" * 70)
        print("V1 (Pretrained BLAIR) vs V2 (Custom BLAIR) COMPARISON")
        print("=" * 70)
        print(f"{'Metric':<20} {'V1':>12} {'V2':>12} {'Change':>12}")
        print("-" * 70)
        for metric in ["ndcg@1", "ndcg@5", "ndcg@10", "mrr", "hr@10"]:
            v1_val = v1_lr.get(metric, 0)
            v2_val = v2_lr.get(metric, 0)
            if v1_val > 0:
                change = (v2_val - v1_val) / v1_val * 100
                arrow = "^" if change >= 0 else "v"
                print(f"{metric:<20} {v1_val:>12.4f} {v2_val:>12.4f} {arrow}{abs(change):>10.2f}%")
        print("=" * 70)

        comparison = {
            "v1_model": "hyp1231/blair-roberta-large",
            "v2_model": "blair-videogames-multiaspect",
            "v1": v1_lr,
            "v2": v2_lr,
            "improvement_pct": {
                m: round(
                    (v2_lr.get(m, 0) - v1_lr.get(m, 0))
                    / max(v1_lr.get(m, 1e-9), 1e-9) * 100,
                    4,
                )
                for m in ["ndcg@1", "ndcg@5", "ndcg@10", "mrr", "hr@10"]
            },
        }
        comp_path = results_dir / "v1_v2_comparison.json"
        comp_path.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        logger.info("Saved V1 vs V2 comparison: %s", comp_path)

    chart_path = charts_dir / "system_comparison.png"
    _plot_comparison(all_results, k_values, chart_path)

    _update_progress()
    print("stage8 complete:")
    print(f"  results: {out_path}")
    print(f"  chart:   {chart_path}")


if __name__ == "__main__":
    run()
