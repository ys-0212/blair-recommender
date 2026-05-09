"""Stage 6 — feature engineering for LambdaRank training data."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config import PROJECT_ROOT, ensure_dirs, get_embedding_dir, get_embedding_path, get_path, load_config
from src.stage4_faiss.retriever import Retriever
from src.stage6_features.candidate_generator import generate_candidates_batch
from src.stage6_features.feature_builder import (
    FEATURE_COLS,
    OUTPUT_COLS,
    UserArrays,
    build_features_raw,
    compute_norm_stats,
    normalize_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

BATCH_SIZE  = 1000   # split rows per processing batch
FLUSH_EVERY = 10     # batches per parquet write


def _load_item_lookup(cfg: dict) -> tuple[dict[str, np.ndarray], int]:
    """Return {parent_asin: L2-normalised embedding} and embedding dim."""
    emb_path = get_embedding_path(cfg, "item_embeddings")
    ids_path = get_embedding_path(cfg, "item_ids")
    if not emb_path.exists():
        raise FileNotFoundError(
            f"Item embeddings not found: {emb_path}  "
            "Run Stage 3 (Colab) and place outputs in the active embedding dir."
        )
    emb = np.load(str(emb_path)).astype(np.float32)
    ids = np.load(str(ids_path), allow_pickle=True).tolist()
    lookup = {asin: emb[i] for i, asin in enumerate(ids)}
    dim = emb.shape[1]
    logger.info("Loaded item lookup: %d items, dim=%d", len(lookup), dim)
    return lookup, dim


def _load_voice_dict(cfg: dict) -> dict[str, np.ndarray]:
    """Return {user_id: voice embedding}. Empty dict if files missing."""
    emb_path = get_embedding_path(cfg, "user_voice_embeddings")
    ids_path = get_embedding_path(cfg, "user_voice_ids")
    if not emb_path.exists() or not ids_path.exists():
        logger.warning(
            "User voice embeddings not found at %s — voice features will be zero.",
            emb_path,
        )
        return {}
    emb = np.load(str(emb_path)).astype(np.float32)
    ids = np.load(str(ids_path), allow_pickle=True).tolist()
    voice_dict = {uid: emb[i] for i, uid in enumerate(ids)}
    logger.info("Loaded voice embeddings: %d users", len(voice_dict))
    return voice_dict


_EMBEDDING_COLS = ("uniform_embedding", "recency_embedding", "rating_embedding", "combined_embedding")


def _build_profiles_dict(profiles_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Convert user_profiles DataFrame to O(1)-lookup dict (used by blender for candidate generation)."""
    result = profiles_df.set_index("user_id").to_dict(orient="index")
    for prof in result.values():
        for col in _EMBEDDING_COLS:
            v = prof.get(col)
            if v is not None and not isinstance(v, np.ndarray):
                try:
                    prof[col] = np.asarray(v, dtype=np.float32)
                except (TypeError, ValueError):
                    prof[col] = None
    return result


def _build_user_arrays(
    profiles_df: pd.DataFrame,
    voice_dict: dict[str, np.ndarray],
    dim: int,
) -> UserArrays:
    """Pre-build numpy matrices for all user profile fields.

    Converts the profiles DataFrame into contiguous numpy arrays indexed by an
    integer user index, enabling O(1) batch gather operations in build_features_raw
    instead of per-row Python dict lookups.
    """
    t0 = time.time()
    uid_list   = profiles_df["user_id"].astype(str).tolist()
    uid_to_idx = {uid: i for i, uid in enumerate(uid_list)}
    n          = len(uid_list)

    def _to_matrix(col: str) -> np.ndarray:
        mat = np.zeros((n, dim), dtype=np.float32)
        if col not in profiles_df.columns:
            return mat
        for i, val in enumerate(profiles_df[col]):
            if val is None:
                continue
            try:
                arr = np.asarray(val, dtype=np.float32)
                if arr.shape == (dim,):
                    mat[i] = arr
            except (TypeError, ValueError):
                pass
        return mat

    def _to_float_arr(col: str) -> np.ndarray:
        if col not in profiles_df.columns:
            return np.full(n, np.nan, dtype=np.float32)
        return pd.to_numeric(profiles_df[col], errors="coerce").values.astype(np.float32)

    def _to_str_arr(col: str) -> np.ndarray:
        if col not in profiles_df.columns:
            return np.full(n, "", dtype=object)
        return profiles_df[col].fillna("").astype(str).values

    # Interaction count log
    if "interaction_count" in profiles_df.columns:
        ic_log = np.log1p(
            profiles_df["interaction_count"].fillna(0).clip(lower=0).values
        ).astype(np.float32)
    else:
        ic_log = np.zeros(n, dtype=np.float32)

    # Top categories as pre-built sets for O(1) lookup
    top_cat_sets: list[set] = []
    if "top_categories" in profiles_df.columns:
        for v in profiles_df["top_categories"]:
            top_cat_sets.append(set(v) if isinstance(v, list) else set())
    else:
        top_cat_sets = [set() for _ in range(n)]

    # Voice matrix: align voice_dict (keyed by str user_id) into user_arrays order
    voice_matrix = np.zeros((n, dim), dtype=np.float32)
    for uid, idx in uid_to_idx.items():
        v = voice_dict.get(uid)
        if v is not None:
            try:
                arr = np.asarray(v, dtype=np.float32)
                if arr.shape == (dim,):
                    voice_matrix[idx] = arr
            except (TypeError, ValueError):
                pass

    user_arrays = UserArrays(
        user_id_to_idx        = uid_to_idx,
        uniform_matrix        = _to_matrix("uniform_embedding"),
        recency_matrix        = _to_matrix("recency_embedding"),
        rating_matrix         = _to_matrix("rating_embedding"),
        combined_matrix       = _to_matrix("combined_embedding"),
        voice_matrix          = voice_matrix,
        interaction_count_log = ic_log,
        avg_sentiment         = _to_float_arr("user_avg_sentiment"),
        top_aspect            = _to_str_arr("user_top_aspect"),
        preferred_price_tier  = _to_str_arr("preferred_price_tier"),
        top_cat_sets          = top_cat_sets,
        user_aspect_gameplay  = _to_float_arr("user_aspect_gameplay"),
        user_aspect_graphics  = _to_float_arr("user_aspect_graphics"),
        user_aspect_story     = _to_float_arr("user_aspect_story"),
        user_aspect_controls  = _to_float_arr("user_aspect_controls"),
        user_aspect_value     = _to_float_arr("user_aspect_value"),
    )
    logger.info(
        "Built UserArrays: %d users, dim=%d, matrices=%s in %.1f s",
        n, dim,
        f"{user_arrays.uniform_matrix.nbytes * 4 / 1e6:.0f} MB total",
        time.time() - t0,
    )
    return user_arrays


def _load_products_df(cfg: dict) -> pd.DataFrame:
    """Load products_nlp.parquet (includes meta_clean fields: price, avg_rating, etc.)."""
    path = get_path(cfg, "data_processed") / "products_nlp.parquet"
    if not path.exists():
        raise FileNotFoundError(f"products_nlp.parquet not found: {path}")
    df = pd.read_parquet(path)
    logger.info("Loaded products_nlp: %d rows x %d cols", len(df), df.shape[1])
    return df



def _process_split(
    split_name: str,
    split_df: pd.DataFrame,
    output_path: Path,
    item_lookup: dict[str, np.ndarray],
    dim: int,
    retriever: Retriever,
    profiles_dict: dict[str, Any],
    voice_dict: dict[str, np.ndarray],
    user_arrays: UserArrays,
    products_df: pd.DataFrame,
    cfg: dict,
    norm_stats: dict[str, dict[str, float]],
    top_k: int,
) -> dict[str, Any]:
    """Candidates + features + normalize + streaming parquet write for one split."""
    t_start    = time.time()
    total_rows = len(split_df)
    n_batches  = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE

    logger.info("processing %s: %d interactions -> %d batches",
                split_name, total_rows, n_batches)

    writer:   pq.ParquetWriter | None = None
    pa_schema: pa.Schema | None       = None
    flush_buf: list[pd.DataFrame]     = []

    total_written = 0
    n_positives   = 0
    n_forced      = 0
    nan_counts: dict[str, int] = {f: 0 for f in FEATURE_COLS}

    for batch_idx in range(n_batches):
        b_start = batch_idx * BATCH_SIZE
        batch   = split_df.iloc[b_start: b_start + BATCH_SIZE]

        candidates_df, query_embs = generate_candidates_batch(
            batch_df      = batch,
            item_lookup   = item_lookup,
            dim           = dim,
            retriever     = retriever,
            profiles_dict = profiles_dict,
            voice_dict    = voice_dict,
            cfg           = cfg,
            top_k         = top_k,
        )

        if candidates_df.empty:
            logger.warning("Batch %d/%d produced no candidates — skipping.", batch_idx + 1, n_batches)
            continue

        n_forced += int(candidates_df["is_forced_positive"].sum())

        raw_df  = build_features_raw(
            candidates_df = candidates_df,
            query_embs    = query_embs,
            item_lookup   = item_lookup,
            dim           = dim,
            user_arrays   = user_arrays,
            voice_dict    = voice_dict,
            products_df   = products_df,
            cfg           = cfg,
            top_k         = top_k,
        )
        norm_df = normalize_features(raw_df, norm_stats)

        for f in FEATURE_COLS:
            if f in raw_df.columns:
                nan_counts[f] += int(raw_df[f].isna().sum())

        flush_buf.append(norm_df)
        n_positives += int(norm_df["relevance_label"].sum())

        del candidates_df, query_embs, raw_df

        is_last = (batch_idx == n_batches - 1)
        if len(flush_buf) >= FLUSH_EVERY or is_last:
            chunk = pd.concat(flush_buf, ignore_index=True)
            total_written += len(chunk)

            if writer is None:
                table     = pa.Table.from_pandas(chunk, preserve_index=False)
                pa_schema = table.schema
                writer    = pq.ParquetWriter(str(output_path), pa_schema, compression="snappy")
            else:
                table = pa.Table.from_pandas(chunk, schema=pa_schema, preserve_index=False)
            writer.write_table(table)

            flush_buf = []
            del chunk

            elapsed = time.time() - t_start
            logger.info(
                "%s | batch %d/%d (%.1f%%) | written %d rows | %.0fs",
                split_name, batch_idx + 1, n_batches,
                100 * (batch_idx + 1) / n_batches,
                total_written, elapsed,
            )

    if writer is not None:
        writer.close()

    elapsed    = time.time() - t_start
    pos_pct    = 100 * n_positives / max(total_written, 1)
    forced_pct = 100 * n_forced / max(n_positives, 1)
    logger.info(
        "%s DONE: %d rows | %d positives (%.3f%%) | %d forced (%.1f%% of pos) | %.0fs",
        split_name, total_written, n_positives, pos_pct, n_forced, forced_pct, elapsed,
    )

    return {
        "split":         split_name,
        "rows":          total_written,
        "positives":     n_positives,
        "negatives":     total_written - n_positives,
        "positive_rate": pos_pct,
        "n_forced":      n_forced,
        "forced_pct":    forced_pct,
        "elapsed_s":     round(elapsed, 1),
        "nan_counts":    nan_counts,
    }


def _update_progress() -> None:
    prog_path = PROJECT_ROOT / "PROGRESS.md"
    if not prog_path.exists():
        logger.warning("PROGRESS.md not found at %s — skipping update", prog_path)
        return
    text = prog_path.read_text(encoding="utf-8")
    updated = text
    for old_status in [
        "| 6 | Feature Engineering | ⚠️ Regenerating — stale files deleted |",
        "| 6 | Feature Engineering | 🟡 Code complete — ready to run |",
        "| 6 | Feature Engineering | ⬜ Not started |",
    ]:
        if old_status in updated:
            updated = updated.replace(old_status, "| 6 | Feature Engineering | ✅ Complete |")
            break
    if updated != text:
        prog_path.write_text(updated, encoding="utf-8")
        logger.info("Updated PROGRESS.md - Stage 6 marked complete")


def _print_summary(summaries: list[dict], norm_stats: dict[str, dict[str, float]]) -> None:
    print()
    print("stage6 complete:")
    print("\nOutput sizes:")
    for s in summaries:
        print(
            f"  {s['split']:6s}: {s['rows']:>12,} rows | "
            f"{s['positives']:>8,} pos ({s['positive_rate']:.3f}%) | "
            f"{s['n_forced']:>7,} forced | "
            f"{s['negatives']:>12,} neg | {s['elapsed_s']:.0f}s"
        )

    print(f"\n{len(FEATURE_COLS)} features:")
    for i, f in enumerate(FEATURE_COLS, 1):
        st = norm_stats.get(f, {})
        print(
            f"  {i:>2}. {f:<42} "
            f"[{st.get('min', 0):.3f}, {st.get('max', 1):.3f}]  "
            f"fill={st.get('fill', 0):.3f}"
        )

    print("\nPre-normalisation NaN counts (train split):")
    train_s = next((s for s in summaries if s["split"] == "train"), None)
    if train_s:
        any_nan = False
        for f, cnt in train_s["nan_counts"].items():
            if cnt > 0:
                print(f"  {f:<42} {cnt:>12,}")
                any_nan = True
        if not any_nan:
            print("  (none)")


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    logger.info("Active version: %s", cfg.get("pipeline", {}).get("active_version", "v3"))
    logger.info("Embedding dir: %s", get_embedding_dir(cfg))

    proc    = get_path(cfg, "data_processed")
    results = get_path(cfg, "outputs_results")
    top_k   = int(cfg.get("stage4", {}).get("top_k_candidates", 200))

    t0 = time.time()

    item_lookup, dim = _load_item_lookup(cfg)
    retriever = Retriever(cfg)

    user_profiles_df = pd.read_parquet(get_path(cfg, "user_profiles"))
    profiles_dict    = _build_profiles_dict(user_profiles_df)
    logger.info("Profiles dict: %d users", len(profiles_dict))

    voice_dict  = _load_voice_dict(cfg)
    products_df = _load_products_df(cfg)

    logger.info("Building UserArrays (pre-indexed numpy matrices) ...")
    user_arrays = _build_user_arrays(user_profiles_df, voice_dict, dim)

    train_df = pd.read_parquet(get_path(cfg, "train"))
    valid_df = pd.read_parquet(get_path(cfg, "valid"))
    test_df  = pd.read_parquet(get_path(cfg, "test"))
    logger.info(
        "Split sizes -- train: %d  valid: %d  test: %d",
        len(train_df), len(valid_df), len(test_df),
    )
    logger.info("Data loading complete in %.1f s", time.time() - t0)

    norm_stats = compute_norm_stats(
        products_df      = products_df,
        user_profiles_df = user_profiles_df,
        top_k            = top_k,
    )

    stats_path = results / "feature_stats.json"
    stats_path.write_text(json.dumps(norm_stats, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved feature_stats.json: %s", stats_path)

    shared = dict(
        item_lookup   = item_lookup,
        dim           = dim,
        retriever     = retriever,
        profiles_dict = profiles_dict,
        voice_dict    = voice_dict,
        user_arrays   = user_arrays,
        products_df   = products_df,
        cfg           = cfg,
        norm_stats    = norm_stats,
        top_k         = top_k,
    )

    summaries: list[dict] = []

    summaries.append(_process_split(
        split_name  = "train",
        split_df    = train_df,
        output_path = proc / "features_train.parquet",
        **shared,
    ))

    summaries.append(_process_split(
        split_name  = "valid",
        split_df    = valid_df,
        output_path = proc / "features_valid.parquet",
        **shared,
    ))

    summaries.append(_process_split(
        split_name  = "test",
        split_df    = test_df,
        output_path = proc / "features_test.parquet",
        **shared,
    ))

    _print_summary(summaries, norm_stats)
    _update_progress()


if __name__ == "__main__":
    run()
