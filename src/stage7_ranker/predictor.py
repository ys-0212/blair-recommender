"""Stage 7 — inference utilities and offline evaluation for the trained LambdaRank model."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

def predict_scores(
    model: lgb.Booster,
    X: np.ndarray,
    feature_cols: list[str],
) -> np.ndarray:
    """Return raw LambdaRank scores for a feature matrix; higher = ranked higher."""
    assert X.shape[1] == len(feature_cols), (
        f"Feature count mismatch: X has {X.shape[1]} cols, "
        f"feature_cols has {len(feature_cols)}"
    )
    return model.predict(X).astype(np.float32)


def rerank(
    candidates_df: pd.DataFrame,
    model: lgb.Booster,
    feature_cols: list[str],
    top_k: int = 10,
) -> pd.DataFrame:
    """Adds lambdarank_score, sorts descending, returns top_k rows."""
    X = candidates_df[feature_cols].values.astype(np.float32)
    scores = predict_scores(model, X, feature_cols)
    out = candidates_df.copy()
    out["lambdarank_score"] = scores
    out = out.sort_values("lambdarank_score", ascending=False)
    return out.head(top_k).reset_index(drop=True)


def compute_ndcg_at_k(
    ranked_df: pd.DataFrame,
    k: int,
    label_col: str = "relevance_label",
) -> float:
    """NDCG@k for an already-ranked candidate list (sorted descending by score)."""
    rels = ranked_df[label_col].values[:k].astype(float)

    dcg = float(np.sum(rels / np.log2(np.arange(2, len(rels) + 2))))

    # Ideal: sort positives first
    ideal_rels = np.sort(ranked_df[label_col].values.astype(float))[::-1][:k]
    idcg = float(np.sum(ideal_rels / np.log2(np.arange(2, len(ideal_rels) + 2))))

    return dcg / idcg if idcg > 0.0 else 0.0


def evaluate_system(
    features_path: Path,
    model: lgb.Booster | None,
    feature_cols: list[str],
    system_name: str,
    score_col: str,
    k_values: list[int] | None = None,
    row_group_batch: int = 10,
) -> dict[str, Any]:
    """
    Chunked evaluation of a ranking system on a feature parquet file.

    Skips queries with no positive label (standard LTR protocol). Pass model=None
    for baselines (score_col already in file); pass model for LambdaRank.
    """
    if k_values is None:
        k_values = [1, 5, 10]

    t0 = time.time()
    logger.info("Evaluating %s on %s ...", system_name, features_path.name)

    pf   = pq.ParquetFile(str(features_path))
    n_rg = pf.metadata.num_row_groups

    # Columns to read from disk
    read_cols = ["user_id", "query_parent_asin", "relevance_label"] + feature_cols

    # Accumulators
    query_metrics: list[dict[str, float]] = []
    total_queries    = 0
    skipped_queries  = 0

    # Carry-over buffer: rows for the last query in the previous chunk
    carry_df: pd.DataFrame | None = None

    def _score_and_eval(group_df: pd.DataFrame) -> dict[str, float] | None:
        """Score a single-query DataFrame and return per-query metrics."""
        if group_df["relevance_label"].sum() == 0:
            return None  # no positive → skip

        if model is not None:
            X = group_df[feature_cols].values.astype(np.float32)
            group_df = group_df.copy()
            group_df["lambdarank_score"] = predict_scores(model, X, feature_cols)

        ranked = group_df.sort_values(score_col, ascending=False).reset_index(drop=True)

        m: dict[str, float] = {}
        for k in k_values:
            m[f"ndcg@{k}"] = compute_ndcg_at_k(ranked, k)
            top_k_labels    = ranked["relevance_label"].values[:k]
            m[f"hr@{k}"]   = 1.0 if top_k_labels.sum() > 0 else 0.0

        # MRR — rank of the first relevant item (1-indexed)
        pos_ranks = np.where(ranked["relevance_label"].values > 0)[0]
        m["mrr"] = 1.0 / (pos_ranks[0] + 1) if len(pos_ranks) > 0 else 0.0

        # Recall@10
        n_relevant   = ranked["relevance_label"].sum()
        top10_labels = ranked["relevance_label"].values[:10]
        m["recall@10"] = float(top10_labels.sum()) / max(n_relevant, 1)

        return m

    for rg_start in range(0, n_rg, row_group_batch):
        rg_end  = min(rg_start + row_group_batch, n_rg)
        table   = pf.read_row_groups(list(range(rg_start, rg_end)), columns=read_cols)
        chunk   = table.to_pandas()

        # Prepend carry-over from previous chunk
        if carry_df is not None:
            chunk = pd.concat([carry_df, chunk], ignore_index=True)
            carry_df = None

        # Build query key using string concatenation (avoids np.diff on strings)
        chunk["_qkey"] = chunk["user_id"].astype(str) + "_" + chunk["query_parent_asin"].astype(str)

        # The last qkey in the chunk may be incomplete (continues in next chunk).
        # Hold it back as carry; process everything else via groupby.
        is_last_batch = (rg_end >= n_rg)
        if not is_last_batch:
            last_qkey = chunk["_qkey"].iloc[-1]
            complete  = chunk[chunk["_qkey"] != last_qkey]
            carry_df  = chunk[chunk["_qkey"] == last_qkey].copy()
        else:
            complete = chunk

        for _qkey, grp in complete.groupby("_qkey", sort=False):
            total_queries += 1
            metrics = _score_and_eval(grp)
            if metrics is None:
                skipped_queries += 1
            else:
                query_metrics.append(metrics)

        if (rg_start // row_group_batch) % 20 == 0:
            logger.info(
                "  %s | row groups %d-%d / %d | queries so far: %d",
                system_name, rg_start, rg_end - 1, n_rg, total_queries,
            )

    # Handle leftover carry (last batch already flushed via complete=chunk above,
    # but if the very last row-group batch had a trailing carry, process it now)
    if carry_df is not None:
        if "_qkey" not in carry_df.columns:
            carry_df["_qkey"] = carry_df["user_id"].astype(str) + "_" + carry_df["query_parent_asin"].astype(str)
        for _qkey, grp in carry_df.groupby("_qkey", sort=False):
            total_queries += 1
            metrics = _score_and_eval(grp)
            if metrics is None:
                skipped_queries += 1
            else:
                query_metrics.append(metrics)

    evaluated_queries = len(query_metrics)
    elapsed = time.time() - t0

    if evaluated_queries == 0:
        logger.warning("%s: no evaluable queries found!", system_name)
        result: dict[str, Any] = {k_: 0.0 for k_ in
                                  [f"ndcg@{k}" for k in k_values] +
                                  [f"hr@{k}"   for k in k_values] +
                                  ["mrr", "recall@10"]}
        result.update({
            "total_queries":     total_queries,
            "evaluated_queries": 0,
            "skipped_queries":   total_queries,
            "elapsed_s":         round(elapsed, 1),
        })
        return result

    # Macro average
    result = {}
    all_keys = list(query_metrics[0].keys())
    for key in all_keys:
        result[key] = float(np.mean([m[key] for m in query_metrics]))

    result["total_queries"]     = total_queries
    result["evaluated_queries"] = evaluated_queries
    result["skipped_queries"]   = skipped_queries
    result["elapsed_s"]         = round(elapsed, 1)

    logger.info(
        "%s: %d queries evaluated (%d skipped) in %.0f s",
        system_name, evaluated_queries, skipped_queries, elapsed,
    )
    for k in k_values:
        logger.info(
            "  NDCG@%-3d = %.4f  HR@%-3d = %.4f",
            k, result[f"ndcg@{k}"], k, result[f"hr@{k}"],
        )
    logger.info(
        "  MRR = %.4f  Recall@10 = %.4f",
        result["mrr"], result["recall@10"],
    )

    return result
