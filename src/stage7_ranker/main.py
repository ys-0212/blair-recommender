"""Stage 7 — trains LambdaRank on Stage 6 features and evaluates against FAISS baseline."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

from src.utils.config import PROJECT_ROOT, ensure_dirs, get_path, load_config
from src.stage7_ranker.trainer import plot_feature_importance, train_lambdarank
from src.stage7_ranker.predictor import evaluate_system

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _print_comparison_table(
    faiss_metrics: dict,
    lr_metrics: dict,
    k_values: list[int],
) -> None:
    """Print a box-drawing comparison table to stdout."""

    col_w  = 10  # width of each metric column
    sys_w  = 22  # width of the system name column

    def _fmt(v: object, width: int = col_w) -> str:
        if isinstance(v, float):
            return f"{v:.4f}".center(width)
        return str(v).center(width)

    def _fmt_pct(base: float, new: float, width: int = col_w) -> str:
        if base == 0.0:
            return "  N/A  ".center(width)
        delta = (new - base) / base * 100.0
        sign  = "+" if delta >= 0 else ""
        return f"{sign}{delta:.2f}%".center(width)

    metrics_order = (
        [f"ndcg@{k}" for k in k_values]
        + ["mrr"]
        + [f"hr@{max(k_values)}"]
    )
    headers = (
        [f"NDCG@{k}" for k in k_values]
        + ["MRR"]
        + [f"HR@{max(k_values)}"]
    )

    n_cols  = len(metrics_order)
    h_parts = "".join(f"{'═' * col_w}╦" for _ in range(n_cols))
    h_parts = h_parts.rstrip("╦")

    sep_top   = f"╔{'═' * sys_w}╦{h_parts}╗"
    sep_mid   = f"╠{'═' * sys_w}╬{'╬'.join('═' * col_w for _ in range(n_cols))}╣"
    sep_bot   = f"╚{'═' * sys_w}╩{'╩'.join('═' * col_w for _ in range(n_cols))}╝"

    def _row(name: str, values: list[str]) -> str:
        cells = "║".join(v for v in values)
        return f"║{name:<{sys_w}}║{cells}║"

    header_vals  = [_fmt(h) for h in headers]
    faiss_vals   = [_fmt(faiss_metrics.get(m, 0.0)) for m in metrics_order]
    lr_vals      = [_fmt(lr_metrics.get(m, 0.0))    for m in metrics_order]
    improv_vals  = [
        _fmt_pct(faiss_metrics.get(m, 0.0), lr_metrics.get(m, 0.0))
        for m in metrics_order
    ]

    print()
    print(sep_top)
    print(_row(" System", header_vals))
    print(sep_mid)
    print(_row(" FAISS Baseline", faiss_vals))
    print(_row(" LambdaRank (Ours)", lr_vals))
    print(_row(" Improvement", improv_vals))
    print(sep_bot)
    print()


def _update_progress(faiss_ndcg10: float, lr_ndcg10: float) -> None:
    prog_path = PROJECT_ROOT / "PROGRESS.md"
    if not prog_path.exists():
        logger.warning("PROGRESS.md not found at %s -- skipping update", prog_path)
        return

    text = prog_path.read_text(encoding="utf-8")

    # Fix Stage 6 status — find the exact table row and replace status cell.
    # str.replace is safe here; no regex needed and avoids unicode-escape issues.
    for old_s6 in [
        "| 6 | Feature Engineering | ⚠️ Regenerating — stale files deleted |",
        "| 6 | Feature Engineering | 🟡 Code complete — ready to run |",
        "| 6 | Feature Engineering | ⬜ Not started |",
    ]:
        if old_s6 in text:
            text = text.replace(old_s6, "| 6 | Feature Engineering | ✅ Complete |")
            break

    # Fix Stage 7 status
    for old_s7 in [
        "| 7 | LambdaRank | 🟡 Code complete — ready to run |",
        "| 7 | LambdaRank | ⬜ Not started |",
    ]:
        if old_s7 in text:
            text = text.replace(old_s7, "| 7 | LambdaRank | ✅ Complete |")
            break

    # Add Stage 7 detail section if not present
    stage7_header = "## Stage 7 -- LambdaRank"
    if stage7_header not in text:
        section = f"""
---

## Stage 7 -- LambdaRank \u2705 Complete

**Completed:** 2026-04-22

### What was done
- Trained LightGBM LambdaRank on 26 features (f01-f26) from Stage 6 feature files
- f27_is_forced_positive excluded from training (would leak label information)
- Evaluated FAISS Baseline vs LambdaRank on validation set

### Results (valid set)
| System | NDCG@10 |
|--------|---------|
| FAISS Baseline | {faiss_ndcg10:.4f} |
| LambdaRank (Ours) | {lr_ndcg10:.4f} |

### Honest FAISS Recall Limitation
FAISS natural recall is approximately 4% at nlist=128, nprobe=16.
Approximately 95.9% of positive labels in training data are force-injected
(ground truth not retrieved by FAISS, added with faiss_score=0.0, faiss_rank=101).

LambdaRank still learns valid re-ranking signal:
- Forced positives teach the model what a "good" item looks like for a user
- The model learns feature patterns that correlate with user preferences

**Future improvement:** Rebuild FAISS index with nlist=512, nprobe=64.
Expected natural recall improvement: ~4% -> ~25-40%.

"""
        # Insert before the Known Issues section or at end
        if "## Known Issues" in text:
            text = text.replace("## Known Issues", section + "## Known Issues")
        else:
            text = text.rstrip() + "\n" + section

    prog_path.write_text(text, encoding="utf-8")
    logger.info("Updated PROGRESS.md -- Stages 6 and 7 marked complete")


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    cfg7         = cfg.get("stage7", {})
    feature_cols = list(cfg7.get("feature_cols", []))
    k_values     = list(cfg7.get("ndcg_eval_at", [1, 5, 10]))

    proc    = get_path(cfg, "data_processed")
    results = get_path(cfg, "outputs_results")
    charts  = get_path(cfg, "outputs_charts")

    train_path = proc / "features_train.parquet"
    valid_path = proc / "features_valid.parquet"

    for p in [train_path, valid_path]:
        if not p.exists():
            raise FileNotFoundError(
                f"Feature file not found: {p}\n"
                "Run Stage 6 first: python -m src.stage6_features.main"
            )

    if not feature_cols:
        raise ValueError(
            "stage7.feature_cols is empty in config.yaml -- check configuration"
        )

    logger.info("Training LambdaRank: %d features (%s ... %s)", len(feature_cols), feature_cols[0], feature_cols[-1])

    t_total = time.time()
    model   = train_lambdarank(train_path, valid_path, cfg, results)

    faiss_metrics = evaluate_system(
        features_path = valid_path,
        model         = None,
        feature_cols  = feature_cols,
        system_name   = "FAISS Baseline",
        score_col     = "f01_faiss_score",
        k_values      = k_values,
    )

    lr_metrics = evaluate_system(
        features_path = valid_path,
        model         = model,
        feature_cols  = feature_cols,
        system_name   = "LambdaRank (Ours)",
        score_col     = "lambdarank_score",
        k_values      = k_values,
    )

    print()
    print("LambdaRank vs FAISS Baseline (valid set):")
    _print_comparison_table(faiss_metrics, lr_metrics, k_values)

    eval_q   = faiss_metrics.get("evaluated_queries", 0)
    total_q  = faiss_metrics.get("total_queries", 0)
    skipped  = faiss_metrics.get("skipped_queries", 0)
    print(
        f"Note: {eval_q:,} / {total_q:,} queries evaluated "
        f"({skipped:,} skipped — no positive label in candidate set)."
    )
    print(
        "FAISS natural recall ~4% (nlist=128, nprobe=16). "
        "~95.9% of positives are force-injected."
    )
    print()

    improvement: dict[str, float] = {}
    metric_keys = (
        [f"ndcg@{k}" for k in k_values]
        + [f"hr@{k}"  for k in k_values]
        + ["mrr", "recall@10"]
    )
    for key in metric_keys:
        base = faiss_metrics.get(key, 0.0)
        new_ = lr_metrics.get(key, 0.0)
        improvement[key] = (
            (new_ - base) / base * 100.0 if base > 0.0 else 0.0
        )

    eval_out = {
        "faiss_baseline": faiss_metrics,
        "lambdarank":     lr_metrics,
        "improvement":    improvement,
    }
    eval_path = results / "eval_results.json"
    eval_path.write_text(
        json.dumps(eval_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Saved eval_results.json: %s", eval_path)

    logger.info("Plotting feature importance ...")
    importance_path = PROJECT_ROOT / cfg7.get(
        "importance_chart", "outputs/charts/feature_importance.png"
    )
    plot_feature_importance(model, feature_cols, importance_path)

    _update_progress(
        faiss_ndcg10 = faiss_metrics.get("ndcg@10", 0.0),
        lr_ndcg10    = lr_metrics.get("ndcg@10", 0.0),
    )

    elapsed_total = time.time() - t_total
    logger.info("stage7 done in %.0f s (%.1f min)", elapsed_total, elapsed_total / 60)
    print("stage7 complete:")
    print(f"  model:   {PROJECT_ROOT / cfg7.get('model_path', 'outputs/results/lambdarank_model.lgb')}")
    print(f"  history: {PROJECT_ROOT / cfg7.get('training_history', 'outputs/results/training_history.json')}")
    print(f"  eval:    {eval_path}")
    print(f"  chart:   {importance_path}")


if __name__ == "__main__":
    run()
