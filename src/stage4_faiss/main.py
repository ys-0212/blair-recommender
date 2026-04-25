"""Stage 4 — build FAISS index and run smoke-test retrieval."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import ensure_dirs, get_path, load_config
from src.stage4_faiss.index_builder import build_index, load_index
from src.stage4_faiss.retriever import Retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _load_title_map(cfg: dict) -> dict[str, str]:
    """Return a dict parent_asin -> title from meta_clean.parquet."""
    meta_path = get_path(cfg, "meta_clean")
    meta = pd.read_parquet(meta_path, columns=["parent_asin", "title"])
    return dict(zip(meta["parent_asin"], meta["title"]))


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    emb_dir  = get_path(cfg, "data_embeddings")
    idx_path = emb_dir / "faiss_index.bin"

    if idx_path.exists():
        logger.info("FAISS index already exists, loading")
        index, id_map = load_index(cfg)
    else:
        index, id_map = build_index(cfg)

    logger.info("smoke test: 5 random queries, top-10 each")

    emb_path = emb_dir / "item_embeddings.npy"
    embeddings = np.load(emb_path).astype(np.float32)

    retriever = Retriever(cfg)
    title_map = _load_title_map(cfg)

    rng = np.random.default_rng(cfg["project"]["seed"])
    sample_positions = rng.choice(len(id_map), size=5, replace=False)

    for pos in sample_positions:
        query_asin  = id_map[pos]
        query_title = title_map.get(query_asin, "Unknown")[:60]
        query_emb   = embeddings[pos]

        t0 = time.perf_counter()
        results = retriever.retrieve(query_emb, top_k=10)
        latency_ms = (time.perf_counter() - t0) * 1000

        print()
        print(f"QUERY [{pos}]: {query_asin}")
        print(f"  Title : {query_title}")
        print(f"  Latency: {latency_ms:.2f} ms")
        print(f"  Top-5 retrieved:")
        for r in results[:5]:
            asin  = r["parent_asin"]
            score = r["faiss_score"]
            title = title_map.get(asin, "Unknown")[:55]
            marker = " <-- SELF" if asin == query_asin else ""
            print(f"    rank {r['rank']:2d}  score={score:.4f}  {asin}  {title}{marker}")

    logger.info("stage4 done:")
    logger.info("  index ntotal  : %d", index.ntotal)
    logger.info("  Index nlist   : %d", index.nlist)
    logger.info("  nprobe        : %d", cfg["stage4"]["faiss_nprobe"])
    logger.info("  top_k default : %d", cfg["stage4"]["top_k_candidates"])
    logger.info("  Index file    : %s", idx_path)


if __name__ == "__main__":
    run()
