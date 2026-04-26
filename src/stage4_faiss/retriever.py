"""Stage 4 — FAISS query interface over the index built by index_builder.py."""

from __future__ import annotations

import logging
from typing import Any

import faiss
import numpy as np

from src.utils.config import load_config
from src.stage4_faiss.index_builder import load_index

logger = logging.getLogger(__name__)


class Retriever:
    """Thin wrapper around a FAISS index; loads once and reuses across calls."""

    def __init__(self, cfg: dict | None = None) -> None:
        if cfg is None:
            cfg = load_config()
        self._cfg   = cfg
        self._cfg4  = cfg["stage4"]
        self._top_k = int(self._cfg4.get("top_k_candidates", 200))
        self._index, self._id_map = load_index(cfg)

        # Set search parameters depending on index type
        if hasattr(self._index, "hnsw"):
            ef_search = int(self._cfg4.get("hnsw_ef_search", 128))
            self._index.hnsw.efSearch = ef_search
            logger.info(
                "Retriever ready (HNSW): ntotal=%d  efSearch=%d  default_top_k=%d",
                self._index.ntotal, ef_search, self._top_k,
            )
        elif hasattr(self._index, "nprobe"):
            nprobe = int(self._cfg4.get("faiss_nprobe", 64))
            self._index.nprobe = nprobe
            logger.info(
                "Retriever ready (IVFFlat): ntotal=%d  nprobe=%d  default_top_k=%d",
                self._index.ntotal, nprobe, self._top_k,
            )
        else:
            logger.info(
                "Retriever ready: ntotal=%d  default_top_k=%d",
                self._index.ntotal, self._top_k,
            )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k nearest neighbours for one L2-normalised query embedding."""
        k = top_k if top_k is not None else self._top_k
        emb = np.asarray(query_embedding, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb[np.newaxis, :]        # (1, dim)

        scores, indices = self._index.search(emb, k)
        scores   = scores[0]
        indices  = indices[0]

        results = []
        for rank, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx == -1:               # FAISS returns -1 for unfilled slots
                continue
            results.append({
                "parent_asin": self._id_map[idx],
                "faiss_score": float(score),
                "rank":        rank,
            })
        return results

    def batch_retrieve(
        self,
        query_embeddings: np.ndarray,
        top_k: int | None = None,
    ) -> list[list[dict[str, Any]]]:
        """Batch version of retrieve; returns N result-lists for N query embeddings."""
        k = top_k if top_k is not None else self._top_k
        embs = np.asarray(query_embeddings, dtype=np.float32)
        if embs.ndim == 1:
            embs = embs[np.newaxis, :]

        scores_mat, indices_mat = self._index.search(embs, k)

        all_results = []
        for q_scores, q_indices in zip(scores_mat, indices_mat):
            results = []
            for rank, (idx, score) in enumerate(zip(q_indices, q_scores), start=1):
                if idx == -1:
                    continue
                results.append({
                    "parent_asin": self._id_map[idx],
                    "faiss_score": float(score),
                    "rank":        rank,
                })
            all_results.append(results)
        return all_results

    @property
    def ntotal(self) -> int:
        return self._index.ntotal

    def embedding_for(self, parent_asin: str) -> np.ndarray | None:
        """Return the stored embedding for a given parent_asin, or None."""
        try:
            pos = self._id_map.index(parent_asin)
        except ValueError:
            return None
        vec = np.zeros((1, self._index.d), dtype=np.float32)
        self._index.reconstruct(pos, vec[0])
        return vec[0]
