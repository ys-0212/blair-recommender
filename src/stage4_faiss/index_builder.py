"""Stage 4 — builds IVFFlat FAISS index over L2-normalised BLAIR embeddings."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np

from src.utils.config import get_path, load_config

logger = logging.getLogger(__name__)


def build_index(cfg: dict | None = None) -> tuple[faiss.Index, list[str]]:
    """Train IVFFlat index, add all items, save to disk; return (index, id_map)."""
    if cfg is None:
        cfg = load_config()

    cfg4    = cfg["stage4"]
    nlist   = int(cfg4["faiss_nlist"])
    metric_name = cfg4.get("metric", "INNER_PRODUCT").upper()

    emb_dir  = get_path(cfg, "data_embeddings")
    emb_path = emb_dir / "item_embeddings.npy"
    ids_path = emb_dir / "item_ids.npy"
    idx_path = emb_dir / "faiss_index.bin"
    map_path = emb_dir / "faiss_id_map.json"

    logger.info("loading embeddings from %s", emb_path)
    t0 = time.time()
    embeddings = np.load(emb_path).astype(np.float32)   # ensure float32
    item_ids   = np.load(ids_path, allow_pickle=True).tolist()
    n_items, dim = embeddings.shape
    logger.info("Loaded %d embeddings of dim %d in %.1f s", n_items, dim, time.time() - t0)

    # Verify L2 normalisation
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        logger.warning(
            "Embeddings are not perfectly L2-normalised "
            "(min=%.6f, max=%.6f). Normalising now ...",
            norms.min(), norms.max(),
        )
        embeddings = embeddings / norms[:, None]

    metric = (faiss.METRIC_INNER_PRODUCT
              if metric_name == "INNER_PRODUCT"
              else faiss.METRIC_L2)

    quantiser = faiss.IndexFlatIP(dim) if metric == faiss.METRIC_INNER_PRODUCT \
                else faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantiser, dim, nlist, metric)

    logger.info("Training IVFFlat index (nlist=%d, metric=%s) on %d vectors ...",
                nlist, metric_name, n_items)
    t1 = time.time()
    index.train(embeddings)
    logger.info("Training done in %.1f s", time.time() - t1)

    logger.info("Adding %d vectors to index ...", n_items)
    t2 = time.time()
    index.add(embeddings)
    logger.info("Add done in %.1f s  |  index.ntotal=%d", time.time() - t2, index.ntotal)

    emb_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(idx_path))
    logger.info("Saved FAISS index -> %s  (%.1f MB)",
                idx_path.name, idx_path.stat().st_size / 1_048_576)

    map_path.write_text(json.dumps(item_ids), encoding="utf-8")
    logger.info("Saved id map -> %s  (%d entries)", map_path.name, len(item_ids))

    # Stats
    logger.info(
        "Index stats: ntotal=%d  nlist=%d  is_trained=%s  dim=%d",
        index.ntotal, index.nlist, index.is_trained, dim,
    )
    return index, item_ids


def load_index(cfg: dict | None = None) -> tuple[faiss.Index, list[str]]:
    """Load a previously saved FAISS index and id map from disk."""
    if cfg is None:
        cfg = load_config()

    emb_dir  = get_path(cfg, "data_embeddings")
    idx_path = emb_dir / "faiss_index.bin"
    map_path = emb_dir / "faiss_id_map.json"

    if not idx_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {idx_path}. "
            "Run Stage 4 first: python -m src.stage4_faiss.main"
        )

    logger.info("loading FAISS index from %s", idx_path)
    index   = faiss.read_index(str(idx_path))
    id_map  = json.loads(map_path.read_text(encoding="utf-8"))
    logger.info("Loaded index: ntotal=%d  nlist=%d", index.ntotal, index.nlist)
    return index, id_map
