"""Stage 3 — BLAIR item embeddings with checkpointing (GPU/CPU)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from src.utils.config import ensure_dirs, get_path, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _load_model(model_name: str, device: torch.device) -> tuple:
    logger.info("Loading tokenizer and model: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()
    logger.info("Model loaded on %s", device)
    return tokenizer, model


def _embed_batch(
    texts: list[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    max_length: int,
) -> np.ndarray:
    """Encode a batch of texts → L2-normalised CLS embeddings (float32)."""
    enc = tokenizer(
        texts,
        max_length=max_length,
        padding=True,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model(**enc)

    cls = out.last_hidden_state[:, 0, :]           # [B, hidden_dim]
    cls = cls / cls.norm(dim=1, keepdim=True)       # L2 normalise
    return cls.cpu().float().numpy()


def _checkpoint_path(emb_dir: Path) -> tuple[Path, Path]:
    return emb_dir / "item_embeddings_partial.npy", emb_dir / "last_index.txt"


def _save_checkpoint(
    partial: np.ndarray, last_idx: int, emb_dir: Path
) -> None:
    partial_path, idx_path = _checkpoint_path(emb_dir)
    np.save(partial_path, partial)
    idx_path.write_text(str(last_idx))


def _load_checkpoint(emb_dir: Path, n_items: int, emb_dim: int) -> tuple[np.ndarray, int]:
    """Return (partial_array, start_index); start_index=0 if no checkpoint exists."""
    partial_path, idx_path = _checkpoint_path(emb_dir)
    if partial_path.exists() and idx_path.exists():
        last_idx = int(idx_path.read_text().strip())
        partial  = np.load(partial_path)
        if partial.shape == (last_idx, emb_dim):
            logger.info("Resuming from checkpoint: %d / %d items done", last_idx, n_items)
            return partial, last_idx
        logger.warning("Checkpoint shape mismatch — starting from scratch")
    arr = np.zeros((n_items, emb_dim), dtype=np.float32)
    return arr, 0


def _clear_checkpoint(emb_dir: Path) -> None:
    for p in _checkpoint_path(emb_dir):
        if p.exists():
            p.unlink()


def run(
    rich_parquet: Path | None = None,
    emb_dir: Path | None = None,
    cfg: dict | None = None,
) -> None:
    if cfg is None:
        cfg = load_config()
    ensure_dirs(cfg)

    cfg3 = cfg.get("stage3", {})
    model_name       = cfg3.get("model_name",       "hyp1231/blair-roberta-large")
    batch_size       = int(cfg3.get("batch_size",    32))
    max_length       = int(cfg3.get("max_seq_length", 512))
    emb_dim          = int(cfg3.get("embedding_dim", 1024))
    checkpoint_every = int(cfg3.get("checkpoint_every", 5000))

    # Paths
    if rich_parquet is None:
        rich_parquet = get_path(cfg, "data_processed") / "products_rich.parquet"
    if emb_dir is None:
        emb_dir = get_path(cfg, "data_embeddings")
    emb_dir.mkdir(parents=True, exist_ok=True)

    out_emb  = emb_dir / "item_embeddings.npy"
    out_ids  = emb_dir / "item_ids.npy"

    # Load input
    logger.info("Loading products_rich.parquet from %s", rich_parquet)
    df = pd.read_parquet(rich_parquet)
    if "rich_text" not in df.columns or "parent_asin" not in df.columns:
        raise ValueError("products_rich.parquet must have 'parent_asin' and 'rich_text' columns")

    texts  = df["rich_text"].fillna("").tolist()
    asins  = df["parent_asin"].tolist()
    n_items = len(texts)
    logger.info("Embedding %d items  |  batch_size=%d  max_length=%d",
                n_items, batch_size, max_length)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(0))

    # Load model
    tokenizer, model = _load_model(model_name, device)

    # Checkpoint resume
    embeddings, start_idx = _load_checkpoint(emb_dir, n_items, emb_dim)

    t0 = time.time()
    items_done = start_idx

    batches = range(start_idx, n_items, batch_size)
    pbar = tqdm(batches, desc="Embedding", unit="batch",
                initial=start_idx // batch_size,
                total=(n_items + batch_size - 1) // batch_size)

    for batch_start in batches:
        batch_end  = min(batch_start + batch_size, n_items)
        batch_texts = texts[batch_start:batch_end]

        batch_emb = _embed_batch(batch_texts, tokenizer, model, device, max_length)
        embeddings[batch_start:batch_end] = batch_emb
        items_done = batch_end
        pbar.update(1)

        # Checkpoint
        if items_done % checkpoint_every < batch_size:
            _save_checkpoint(embeddings[:items_done], items_done, emb_dir)
            elapsed = time.time() - t0
            rate = items_done / elapsed
            logger.info(
                "Checkpoint: %d / %d  (%.1f items/s, %.1f min elapsed)",
                items_done, n_items, rate, elapsed / 60,
            )

    pbar.close()

    elapsed = time.time() - t0
    rate    = n_items / elapsed

    np.save(out_emb, embeddings)
    np.save(out_ids, np.array(asins, dtype=object))
    _clear_checkpoint(emb_dir)

    logger.info("Saved item_embeddings.npy  shape=%s", embeddings.shape)
    logger.info("Saved item_ids.npy         shape=%s", np.array(asins).shape)
    logger.info("Total time: %.1f min  |  throughput: %.1f items/s", elapsed / 60, rate)

    assert embeddings.shape == (n_items, emb_dim), (
        f"Shape mismatch: expected ({n_items}, {emb_dim}), got {embeddings.shape}"
    )
    norms = np.linalg.norm(embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), \
        f"Embeddings not unit-normalised: min={norms.min():.6f}, max={norms.max():.6f}"
    logger.info("verification passed: shape OK, L2-normalised")

    rng = np.random.default_rng(42)
    idx = rng.choice(n_items, size=6, replace=False)
    logger.info("3 random pair cosine similarities:")
    for i in range(3):
        a, b = idx[i * 2], idx[i * 2 + 1]
        sim  = float(np.dot(embeddings[a], embeddings[b]))
        logger.info("  %s  <->  %s  :  %.4f", asins[a][:10], asins[b][:10], sim)


if __name__ == "__main__":
    run()
