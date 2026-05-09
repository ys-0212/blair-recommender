"""Stage 4 — builds FAISS index over L2-normalised BLAIR embeddings (HNSW or IVFFlat)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np

from src.utils.config import get_embedding_dir, get_embedding_path, load_config

logger = logging.getLogger(__name__)


def _build_hnsw(
    embeddings: np.ndarray,
    m: int,
    ef_construction: int,
    ef_search: int,
    metric_name: str,
) -> faiss.Index:
    """Build HNSW index — no training required."""
    n_items, dim = embeddings.shape
    metric = faiss.METRIC_INNER_PRODUCT if metric_name == "INNER_PRODUCT" else faiss.METRIC_L2
    index = faiss.IndexHNSWFlat(dim, m, metric)
    index.hnsw.efConstruction = ef_construction
    index.hnsw.efSearch = ef_search
    logger.info(
        "Building HNSW index (M=%d, efConstruction=%d, efSearch=%d, metric=%s) on %d vectors ...",
        m, ef_construction, ef_search, metric_name, n_items,
    )
    t0 = time.time()
    index.add(embeddings)
    logger.info("HNSW add done in %.1f s  |  ntotal=%d", time.time() - t0, index.ntotal)
    return index


def _build_ivfflat(
    embeddings: np.ndarray,
    nlist: int,
    metric_name: str,
) -> faiss.Index:
    """Build IVFFlat index — requires training."""
    n_items, dim = embeddings.shape
    metric = faiss.METRIC_INNER_PRODUCT if metric_name == "INNER_PRODUCT" else faiss.METRIC_L2
    quantiser = (faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT
                 else faiss.IndexFlatL2(dim))
    index = faiss.IndexIVFFlat(quantiser, dim, nlist, metric)
    logger.info(
        "Training IVFFlat index (nlist=%d, metric=%s) on %d vectors ...",
        nlist, metric_name, n_items,
    )
    t0 = time.time()
    index.train(embeddings)
    logger.info("Training done in %.1f s", time.time() - t0)
    logger.info("Adding %d vectors to index ...", n_items)
    t1 = time.time()
    index.add(embeddings)
    logger.info("Add done in %.1f s  |  ntotal=%d", time.time() - t1, index.ntotal)
    return index


def measure_self_recall(
    index: faiss.Index,
    embeddings: np.ndarray,
    id_map: list[str],
    top_k: int,
    n_samples: int = 2000,
    seed: int = 42,
) -> float:
    """Sample n_samples items and check if each appears in its own top-k results."""
    rng = np.random.default_rng(seed)
    n_items = len(id_map)
    sample_pos = rng.choice(n_items, size=min(n_samples, n_items), replace=False)

    queries = embeddings[sample_pos].astype(np.float32)
    _, indices_mat = index.search(queries, top_k)

    hits = 0
    for i, orig_pos in enumerate(sample_pos):
        if orig_pos in indices_mat[i]:
            hits += 1

    recall = hits / len(sample_pos)
    logger.info("Self-recall@%d: %.1f%% (%d / %d sampled items)", top_k, recall * 100, hits, len(sample_pos))
    return recall


def build_index(cfg: dict | None = None) -> tuple[faiss.Index, list[str]]:
    """Build HNSW or IVFFlat index, add all items, save to disk; return (index, id_map)."""
    if cfg is None:
        cfg = load_config()

    cfg4        = cfg["stage4"]
    index_type  = cfg4.get("index_type", "HNSW").upper()
    metric_name = cfg4.get("metric", "INNER_PRODUCT").upper()
    top_k       = int(cfg4.get("top_k_candidates", 200))

    logger.info("embedding dir: %s", get_embedding_dir(cfg))
    emb_path = get_embedding_path(cfg, "item_embeddings")
    ids_path = get_embedding_path(cfg, "item_ids")
    idx_path = get_embedding_path(cfg, "faiss_index")
    map_path = get_embedding_path(cfg, "faiss_id_map")

    if not emb_path.exists():
        raise FileNotFoundError(
            f"Item embeddings not found: {emb_path}  "
            "Run Stage 3 (Colab) and place outputs in the active embedding dir."
        )

    logger.info("Loading embeddings from %s", emb_path)
    t0 = time.time()
    embeddings = np.load(emb_path).astype(np.float32)
    item_ids   = np.load(ids_path, allow_pickle=True).tolist()
    n_items, dim = embeddings.shape
    logger.info("Loaded %d embeddings of dim %d in %.1f s", n_items, dim, time.time() - t0)

    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        logger.warning(
            "Embeddings not perfectly L2-normalised (min=%.6f, max=%.6f). Normalising ...",
            norms.min(), norms.max(),
        )
        embeddings = embeddings / norms[:, None]

    if index_type == "HNSW":
        m                = int(cfg4.get("hnsw_m", 32))
        ef_construction  = int(cfg4.get("hnsw_ef_construction", 200))
        ef_search        = int(cfg4.get("hnsw_ef_search", 128))
        index = _build_hnsw(embeddings, m, ef_construction, ef_search, metric_name)
    else:
        nlist = int(cfg4.get("faiss_nlist", 512))
        index = _build_ivfflat(embeddings, nlist, metric_name)

    idx_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    logger.info("Saved FAISS index -> %s  (%.1f MB)",
                idx_path.name, idx_path.stat().st_size / 1_048_576)

    map_path.write_text(json.dumps(item_ids), encoding="utf-8")
    logger.info("Saved id map -> %s  (%d entries)", map_path.name, len(item_ids))

    seed = int(cfg.get("project", {}).get("seed", 42))
    measure_self_recall(index, embeddings, item_ids, top_k=top_k, seed=seed)

    return index, item_ids


def load_index(cfg: dict | None = None) -> tuple[faiss.Index, list[str]]:
    """Load a previously saved FAISS index and id map from disk."""
    if cfg is None:
        cfg = load_config()

    idx_path = get_embedding_path(cfg, "faiss_index")
    map_path = get_embedding_path(cfg, "faiss_id_map")

    if not idx_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {idx_path}  "
            "Run Stage 4 first: python -m src.stage4_faiss.main"
        )
    if not map_path.exists():
        raise FileNotFoundError(
            f"FAISS id map not found: {map_path}  "
            "Run Stage 4 first: python -m src.stage4_faiss.main"
        )

    logger.info("Loading FAISS index from %s", idx_path)
    index  = faiss.read_index(str(idx_path))
    id_map = json.loads(map_path.read_text(encoding="utf-8"))

    try:
        logger.info("Loaded index: ntotal=%d  nlist=%d", index.ntotal, index.nlist)
    except AttributeError:
        logger.info("Loaded index: ntotal=%d  (HNSW)", index.ntotal)

    return index, id_map
