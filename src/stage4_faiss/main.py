"""Stage 4 — build FAISS index and run smoke-test retrieval."""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd

from src.utils.config import ensure_dirs, get_embedding_dir, get_embedding_path, get_path, load_config
from src.stage4_faiss.index_builder import build_index, load_index, measure_self_recall
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

    logger.info("Active version: %s", cfg["pipeline"]["active_version"])
    logger.info("Embedding dir: %s", get_embedding_dir(cfg))

    idx_path = get_embedding_path(cfg, "faiss_index")
    cfg4     = cfg["stage4"]
    top_k    = int(cfg4.get("top_k_candidates", 200))
    seed     = int(cfg.get("project", {}).get("seed", 42))

    if idx_path.exists():
        logger.info("FAISS index already exists — loading (delete to rebuild)")
        index, id_map = load_index(cfg)
        # Still run recall test on existing index
        emb_path   = get_embedding_path(cfg, "item_embeddings")
        embeddings = np.load(emb_path).astype(np.float32)
        measure_self_recall(index, embeddings, id_map, top_k=top_k, seed=seed)
    else:
        index, id_map = build_index(cfg)
        emb_path   = get_embedding_path(cfg, "item_embeddings")
        embeddings = np.load(emb_path).astype(np.float32)

    logger.info("smoke test: 5 random queries, top-10 each")
    retriever = Retriever(cfg)
    title_map = _load_title_map(cfg)

    rng = np.random.default_rng(seed)
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

    index_type = cfg4.get("index_type", "HNSW").upper()
    logger.info("stage4 done:")
    logger.info("  index type    : %s", index_type)
    logger.info("  index ntotal  : %d", index.ntotal)
    logger.info("  top_k default : %d", top_k)
    logger.info("  Index file    : %s", idx_path)
    if index_type == "HNSW":
        logger.info("  efSearch      : %d", cfg4.get("hnsw_ef_search", 128))
    else:
        logger.info("  nprobe        : %d", cfg4.get("faiss_nprobe", 64))


if __name__ == "__main__":
    run()
