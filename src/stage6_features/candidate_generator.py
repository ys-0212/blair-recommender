"""Stage 6 candidate generator — blended BLAIR query + FAISS retrieval + forced positive injection."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.stage5_users.blender import blend_query

logger = logging.getLogger(__name__)


def _l2_norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _build_history_query_emb(
    history_str: Any,
    item_lookup: dict[str, np.ndarray],
    dim: int,
) -> np.ndarray:
    """Uniform-average L2-normalised embedding from space-separated history ASINs.

    Returns a zero vector if the history is empty or no ASINs are in the index.
    """
    if not history_str or not isinstance(history_str, str):
        return np.zeros(dim, dtype=np.float32)
    asins = history_str.strip().split()
    vecs = [item_lookup[a] for a in asins if a in item_lookup]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    avg = np.mean(np.stack(vecs, axis=0), axis=0)
    return _l2_norm(avg.astype(np.float32))


def _build_history_text(
    history_str: Any,
    title_lookup: dict[str, str],
) -> str:
    """Concatenate titles of history ASINs for BM25 query text."""
    if not history_str or not isinstance(history_str, str):
        return ""
    asins = history_str.strip().split()
    titles = [title_lookup.get(a, "") for a in asins if a in title_lookup]
    return " ".join(t for t in titles if t)


def generate_candidates_batch(
    batch_df: pd.DataFrame,
    item_lookup: dict[str, np.ndarray],
    dim: int,
    retriever: Any,
    profiles_dict: dict[str, Any],
    voice_dict: dict[str, np.ndarray],
    cfg: dict,
    top_k: int = 200,
    title_lookup: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """Generate top-k candidates for a batch of split rows, with forced positives.

    Returns (candidates_df, query_embs). candidates_df has columns:
      user_id, query_parent_asin, candidate_parent_asin,
      faiss_score, faiss_rank, relevance_label, is_forced_positive,
      query_history_text (if title_lookup provided).
    Each query produces top_k rows, plus 1 extra if the ground truth was
    not in the FAISS results (is_forced_positive=1).
    """
    graded = cfg.get("stage6", {}).get("graded_relevance", False)

    query_embs: dict[tuple[str, str], np.ndarray] = {}
    blended_list: list[np.ndarray] = []
    row_keys: list[tuple[str, str]] = []
    gt_ratings: dict[tuple[str, str], int] = {}
    query_texts: dict[tuple[str, str], str] = {}

    zero_emb = np.zeros(dim, dtype=np.float32)

    for _, row in batch_df.iterrows():
        uid      = str(row["user_id"])
        gt_asin  = str(row["parent_asin"])
        hist_str = row.get("history", "")
        rating   = row.get("rating", 1)

        raw_query = _build_history_query_emb(hist_str, item_lookup, dim)

        profile   = profiles_dict.get(uid)
        voice_emb = voice_dict.get(uid)

        if profile is not None:
            blended = blend_query(raw_query, profile, voice_emb, cfg)
        else:
            blended = _l2_norm(raw_query) if np.linalg.norm(raw_query) > 1e-9 else zero_emb

        key = (uid, gt_asin)
        query_embs[key]  = blended
        blended_list.append(blended)
        row_keys.append(key)

        if graded:
            try:
                gt_ratings[key] = int(round(float(rating)))
            except (TypeError, ValueError):
                gt_ratings[key] = 1
        else:
            gt_ratings[key] = 1

        if title_lookup is not None:
            query_texts[key] = _build_history_text(hist_str, title_lookup)

    if not blended_list:
        return pd.DataFrame(), {}

    embs_matrix = np.stack(blended_list, axis=0).astype(np.float32)
    all_results = retriever.batch_retrieve(embs_matrix, top_k=top_k)

    rows: list[dict] = []
    n_forced = 0

    for (uid, gt_asin), results in zip(row_keys, all_results):
        key = (uid, gt_asin)
        retrieved_asins = {r["parent_asin"] for r in results}
        gt_in_results   = gt_asin in retrieved_asins
        gt_relevance    = gt_ratings[key]
        hist_text       = query_texts.get(key, "")

        for result in results:
            cand_asin = result["parent_asin"]
            is_gt     = (cand_asin == gt_asin)
            rows.append({
                "user_id":               uid,
                "query_parent_asin":     gt_asin,
                "candidate_parent_asin": cand_asin,
                "faiss_score":           result["faiss_score"],
                "faiss_rank":            result["rank"],
                "relevance_label":       gt_relevance if is_gt else 0,
                "is_forced_positive":    0,
                "query_history_text":    hist_text,
            })

        if not gt_in_results:
            rows.append({
                "user_id":               uid,
                "query_parent_asin":     gt_asin,
                "candidate_parent_asin": gt_asin,
                "faiss_score":           0.0,
                "faiss_rank":            top_k + 1,
                "relevance_label":       gt_relevance,
                "is_forced_positive":    1,
                "query_history_text":    hist_text,
            })
            n_forced += 1

    if n_forced > 0:
        logger.debug(
            "Forced %d positives into candidate set (ground truths outside top-%d)",
            n_forced, top_k,
        )

    candidates_df = pd.DataFrame(rows)
    return candidates_df, query_embs
