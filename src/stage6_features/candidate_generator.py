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
    """
    Uniform-average L2-normalised embedding from space-separated history ASINs.

    Returns a zero vector if the history is empty or no ASINs are in the index.
    Zero vectors are handled gracefully downstream (blend_query returns raw query).
    """
    if not history_str or not isinstance(history_str, str):
        return np.zeros(dim, dtype=np.float32)
    asins = history_str.strip().split()
    vecs = [item_lookup[a] for a in asins if a in item_lookup]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    avg = np.mean(np.stack(vecs, axis=0), axis=0)
    return _l2_norm(avg.astype(np.float32))


def generate_candidates_batch(
    batch_df: pd.DataFrame,
    item_lookup: dict[str, np.ndarray],
    dim: int,
    retriever: Any,
    profiles_dict: dict[str, Any],
    voice_dict: dict[str, np.ndarray],
    cfg: dict,
    top_k: int = 100,
) -> tuple[pd.DataFrame, dict[tuple[str, str], np.ndarray]]:
    """
    Generate top-k candidates for a batch of split rows, with forced positives.

    Parameters
    ----------
    batch_df       : Slice of train/valid/test DataFrame. Expected columns:
                     user_id, parent_asin, history (space-sep ASINs).
    item_lookup    : {parent_asin: L2-normalised float32 (dim,)} from item_embeddings.npy.
    dim            : Embedding dimension (1024 for BLAIR).
    retriever      : Stage 4 Retriever instance (uses batch_retrieve).
    profiles_dict  : {user_id: dict-of-profile-columns} from user_profiles.parquet.
    voice_dict     : {user_id: L2-normalised voice embedding} or empty dict.
    cfg            : Full config dict.
    top_k          : Number of FAISS candidates per query.

    Returns
    -------
    candidates_df : DataFrame with columns:
                      user_id, query_parent_asin, candidate_parent_asin,
                      faiss_score, faiss_rank, relevance_label, is_forced_positive
                    Each query produces top_k rows, plus 1 extra if the ground
                    truth was not in the FAISS results (is_forced_positive=1).
    query_embs    : {(user_id, query_parent_asin): blended_query_emb (dim,)}
    """
    query_embs: dict[tuple[str, str], np.ndarray] = {}
    blended_list: list[np.ndarray] = []
    row_keys: list[tuple[str, str]] = []

    zero_emb = np.zeros(dim, dtype=np.float32)

    for _, row in batch_df.iterrows():
        uid      = str(row["user_id"])
        gt_asin  = str(row["parent_asin"])   # ground-truth item for this interaction
        hist_str = row.get("history", "")

        # Build raw query from user's interaction history (uniform average)
        raw_query = _build_history_query_emb(hist_str, item_lookup, dim)

        # Blend with user profile embeddings
        profile   = profiles_dict.get(uid)
        voice_emb = voice_dict.get(uid)

        if profile is not None:
            blended = blend_query(raw_query, profile, voice_emb, cfg)
        else:
            blended = _l2_norm(raw_query) if np.linalg.norm(raw_query) > 1e-9 else zero_emb

        key = (uid, gt_asin)
        query_embs[key] = blended
        blended_list.append(blended)
        row_keys.append(key)

    if not blended_list:
        return pd.DataFrame(), {}

    # Batch FAISS retrieval — one call for the entire batch
    embs_matrix = np.stack(blended_list, axis=0).astype(np.float32)  # (B, dim)
    all_results = retriever.batch_retrieve(embs_matrix, top_k=top_k)

    # Build candidate rows, forcing in the ground truth when absent
    rows: list[dict] = []
    n_forced = 0

    for (uid, gt_asin), results in zip(row_keys, all_results):
        retrieved_asins = {r["parent_asin"] for r in results}
        gt_in_results   = gt_asin in retrieved_asins

        for result in results:
            cand_asin = result["parent_asin"]
            rows.append({
                "user_id":               uid,
                "query_parent_asin":     gt_asin,
                "candidate_parent_asin": cand_asin,
                "faiss_score":           result["faiss_score"],
                "faiss_rank":            result["rank"],
                "relevance_label":       int(cand_asin == gt_asin),
                "is_forced_positive":    0,
            })

        if not gt_in_results:
            # Force-add the ground truth so every query has exactly 1 positive
            rows.append({
                "user_id":               uid,
                "query_parent_asin":     gt_asin,
                "candidate_parent_asin": gt_asin,
                "faiss_score":           0.0,        # not a real retrieval score
                "faiss_rank":            top_k + 1,  # beyond top-k (e.g. 101)
                "relevance_label":       1,
                "is_forced_positive":    1,
            })
            n_forced += 1

    if n_forced > 0:
        logger.debug(
            "Forced %d positives into candidate set (ground truths outside top-%d)",
            n_forced, top_k,
        )

    candidates_df = pd.DataFrame(rows)
    return candidates_df, query_embs
