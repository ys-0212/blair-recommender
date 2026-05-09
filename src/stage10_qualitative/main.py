"""Stage 10 — generates qualitative recommendation examples for 5 diverse users."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.config import PROJECT_ROOT, ensure_dirs, get_path, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _select_diverse_users(
    valid_df: pd.DataFrame,
    profiles_df: pd.DataFrame,
    seed: int = 42,
) -> list[str]:
    """Pick 5 diverse user_ids: power user, cold-ish, RPG fan, FPS fan, random."""
    selected: list[str] = []
    seen: set[str] = set()

    def _add(uid: str) -> None:
        s = str(uid)
        if s not in seen:
            seen.add(s)
            selected.append(s)

    # Merge interaction counts into valid users
    valid_users = set(valid_df["user_id"].astype(str).unique())
    prof = profiles_df[profiles_df["user_id"].astype(str).isin(valid_users)].copy()
    prof["user_id"] = prof["user_id"].astype(str)

    # 1. Power user (most interactions)
    if "interaction_count" in prof.columns:
        _add(prof.sort_values("interaction_count", ascending=False).iloc[0]["user_id"])

    # 2. Cold-ish user (few but not zero interactions — avoid tier-0)
    if "interaction_count" in prof.columns:
        mid = prof[prof["interaction_count"] >= 5].sort_values("interaction_count").iloc[0]["user_id"]
        _add(mid)

    # 3. RPG fan
    if "top_categories" in prof.columns:
        rpg_mask = prof["top_categories"].astype(str).str.contains("Role|RPG|rpg", case=False, na=False)
        rpg_users = prof[rpg_mask & ~prof["user_id"].isin(seen)]
        if not rpg_users.empty:
            _add(rpg_users.iloc[0]["user_id"])

    # 4. FPS/Shooter fan
    if "top_categories" in prof.columns:
        fps_mask = prof["top_categories"].astype(str).str.contains("Shoot|Action|FPS|fps", case=False, na=False)
        fps_users = prof[fps_mask & ~prof["user_id"].isin(seen)]
        if not fps_users.empty:
            _add(fps_users.iloc[0]["user_id"])

    # 5. Random diverse user to fill remaining slots
    rng = np.random.default_rng(seed)
    remaining = [u for u in valid_users if u not in seen]
    while len(selected) < 5 and remaining:
        idx = int(rng.integers(0, len(remaining)))
        _add(remaining[idx])
        remaining.pop(idx)

    return selected[:5]


def _load_features_for_users(
    features_path: Path,
    user_ids: list[str],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Streams features parquet row-group by row-group, filtering to specified user_ids."""
    user_set  = set(str(u) for u in user_ids)
    read_cols = ["user_id", "query_parent_asin", "candidate_parent_asin",
                 "relevance_label", "f01_faiss_score"] + feature_cols
    pf     = pq.ParquetFile(str(features_path))
    chunks = []

    for rg in range(pf.metadata.num_row_groups):
        chunk = pf.read_row_group(rg, columns=read_cols).to_pandas()
        mask  = chunk["user_id"].astype(str).isin(user_set)
        if mask.any():
            chunks.append(chunk[mask])

    if not chunks:
        return pd.DataFrame(columns=read_cols)
    return pd.concat(chunks, ignore_index=True)


def _top_features(row: pd.Series, feature_cols: list[str], n: int = 3) -> list[str]:
    """Return names of the top-n highest-value features for a single candidate row."""
    vals = [(f, float(row.get(f, 0.0) or 0.0)) for f in feature_cols]
    vals.sort(key=lambda x: x[1], reverse=True)
    return [f for f, _ in vals[:n]]


def _build_user_report(
    user_id: str,
    user_feat_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    meta_lookup: dict[str, str],
    nlp_lookup: dict[str, dict],
    model: lgb.Booster,
    feature_cols: list[str],
    profiles_dict: dict[str, Any],
    top_k: int = 5,
) -> dict[str, Any]:
    """Build the recommendation report for one user."""
    user_rows = user_feat_df[user_feat_df["user_id"].astype(str) == str(user_id)]
    if user_rows.empty:
        return {"user_id": user_id, "error": "No feature rows found"}

    # Ground truth item (first row; all rows for same user have same query_parent_asin)
    gt_asin = str(user_rows.iloc[0]["query_parent_asin"])

    # History from valid_df
    valid_row = valid_df[valid_df["user_id"].astype(str) == str(user_id)]
    history_asins: list[str] = []
    if not valid_row.empty and "history" in valid_row.columns:
        hist_str = str(valid_row.iloc[0].get("history", ""))
        history_asins = hist_str.split()[-5:]  # last 5

    history_items = [
        {"asin": a, "title": meta_lookup.get(a, a)[:80]}
        for a in history_asins
    ]

    gt_title = meta_lookup.get(gt_asin, gt_asin)

    # User profile summary
    profile = profiles_dict.get(str(user_id), {})
    user_top_aspect   = str(profile.get("user_top_aspect", "—") or "—")
    interaction_count = int(profile.get("interaction_count", 0) or 0)
    top_categories    = profile.get("top_categories", [])
    if not isinstance(top_categories, list):
        top_categories = []

    # Aspect scores (may be absent)
    aspect_scores: dict[str, float] = {}
    for asp in ("gameplay", "graphics", "story", "controls", "value"):
        v = profile.get(f"user_aspect_{asp}")
        try:
            aspect_scores[asp] = round(float(v), 3)
        except (TypeError, ValueError):
            pass

    # Compute LambdaRank scores
    X      = user_rows[feature_cols].values.astype(np.float32)
    scores = model.predict(X).astype(np.float32)
    user_rows = user_rows.copy()
    user_rows["lambdarank_score"] = scores

    # Build a per-ASIN FAISS score map for the comparison view
    faiss_score_map = dict(zip(
        user_rows["candidate_parent_asin"].astype(str),
        user_rows["f01_faiss_score"].astype(float),
    ))

    def _top_k_recs(score_col: str, k: int, add_why: bool = False) -> list[dict]:
        ranked = user_rows.sort_values(score_col, ascending=False).head(k)
        recs = []
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            asin  = str(row["candidate_parent_asin"])
            nlp   = nlp_lookup.get(asin, {})
            rec: dict[str, Any] = {
                "rank":           rank,
                "asin":           asin,
                "title":          meta_lookup.get(asin, asin)[:80],
                "score":          round(float(row[score_col]), 4),
                "faiss_score":    round(faiss_score_map.get(asin, 0.0), 4),
                "sentiment":      round(float(nlp.get("mean_sentiment", 0.0)), 3),
                "top_aspect":     str(nlp.get("top_aspect", "—")),
                "is_gt":          asin == gt_asin,
            }
            if add_why:
                rec["why_top_features"] = _top_features(row, feature_cols)
            recs.append(rec)
        return recs

    faiss_recs = _top_k_recs("f01_faiss_score", top_k)
    lr_recs    = _top_k_recs("lambdarank_score", top_k, add_why=True)

    # Check if GT appears in top-10
    faiss_top10_asins = set(user_rows.sort_values("f01_faiss_score", ascending=False).head(10)["candidate_parent_asin"].astype(str))
    lr_top10_asins    = set(user_rows.sort_values("lambdarank_score", ascending=False).head(10)["candidate_parent_asin"].astype(str))

    return {
        "user_id":             user_id,
        "n_candidates":        len(user_rows),
        "interaction_count":   interaction_count,
        "user_top_aspect":     user_top_aspect,
        "aspect_scores":       aspect_scores,
        "top_categories":      top_categories[:3],
        "history":             history_items,
        "ground_truth_asin":   gt_asin,
        "ground_truth_title":  gt_title[:80],
        "faiss_top5":          faiss_recs,
        "lambdarank_top5":     lr_recs,
        "gt_in_faiss_top10":   gt_asin in faiss_top10_asins,
        "gt_in_lr_top10":      gt_asin in lr_top10_asins,
    }


def _format_txt_report(reports: list[dict], user_labels: list[str]) -> str:
    lines: list[str] = [
        "=" * 70,
        "BLAIR HYBRID RECOMMENDER — QUALITATIVE ANALYSIS",
        "=" * 70,
        "",
    ]

    for label, rep in zip(user_labels, reports):
        uid = rep["user_id"]
        lines += [
            "-" * 70,
            f"USER: {uid}  [{label}]",
            "-" * 70,
        ]

        if "error" in rep:
            lines += [f"  ERROR: {rep['error']}", ""]
            continue

        # User profile summary
        asp_str = rep.get("user_top_aspect", "—")
        asp_scores = rep.get("aspect_scores", {})
        asp_detail = "  ".join(f"{a}={v:.2f}" for a, v in asp_scores.items()) if asp_scores else "—"
        cats_str = ", ".join(rep.get("top_categories", [])) or "—"
        lines += [
            f"PROFILE: interactions={rep.get('interaction_count', 0)}  top_categories={cats_str}",
            f"CARES MOST ABOUT: {asp_str}",
            f"  Aspect scores: {asp_detail}",
            "",
        ]

        lines.append("HISTORY (last 5 items):")
        for item in rep["history"]:
            lines.append(f"  {item['asin']:15s}  {item['title']}")
        if not rep["history"]:
            lines.append("  (no history available)")

        lines += [
            "",
            f"GROUND TRUTH ITEM: {rep['ground_truth_asin']}",
            f"  Title: {rep['ground_truth_title']}",
            "",
            "FAISS BASELINE TOP-5:",
        ]
        for r in rep["faiss_top5"]:
            gt_flag = " <-- GROUND TRUTH" if r["is_gt"] else ""
            lines.append(f"  {r['rank']}. [{r['score']:.4f}] {r['title']}{gt_flag}")

        lines += [
            f"  GT in FAISS top-10: {'YES' if rep['gt_in_faiss_top10'] else 'NO'}",
            "",
            "LAMBDARANK TOP-5  (score | faiss_score | why):",
        ]
        for r in rep["lambdarank_top5"]:
            gt_flag = " <-- GROUND TRUTH" if r["is_gt"] else ""
            why = ", ".join(r.get("why_top_features", []))
            lines.append(
                f"  {r['rank']}. LR={r['score']:.4f} | FAISS={r['faiss_score']:.4f} | "
                f"{r['title']}  (top={r['top_aspect']}, why: {why}){gt_flag}"
            )
        lines += [
            f"  GT in LambdaRank top-10: {'YES' if rep['gt_in_lr_top10'] else 'NO'}",
            "",
        ]

    lines += ["=" * 70, "END OF REPORT", "=" * 70]
    return "\n".join(lines)


def _update_progress() -> None:
    prog_path = PROJECT_ROOT / "PROGRESS.md"
    if not prog_path.exists():
        return
    text = prog_path.read_text(encoding="utf-8")
    for old in [
        "| 10 | Qualitative Analysis | ⬜ Not started |",
        "| 10 | Qualitative Analysis | 🟡 Code complete — ready to run |",
    ]:
        if old in text:
            text = text.replace(old, "| 10 | Qualitative Analysis | ✅ Complete |")
            break
    prog_path.write_text(text, encoding="utf-8")
    logger.info("Updated PROGRESS.md - Stage 10 marked complete")


def run() -> None:
    cfg = load_config()
    ensure_dirs(cfg)

    cfg7         = cfg.get("stage7", {})
    feature_cols = list(cfg7.get("feature_cols", []))
    cfg10        = cfg.get("stage10", {})
    top_k        = int(cfg10.get("top_k_display", 5))
    seed         = int(cfg.get("project", {}).get("seed", 42))

    proc        = get_path(cfg, "data_processed")
    results_dir = get_path(cfg, "outputs_results")

    model_path = PROJECT_ROOT / cfg7.get("model_path", "outputs/results/lambdarank_model.lgb")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}  Run Stage 7 first.")

    valid_path = proc / "features_valid.parquet"
    if not valid_path.exists():
        raise FileNotFoundError(f"features_valid.parquet not found. Run Stage 6 first.")

    model = lgb.Booster(model_file=str(model_path))
    logger.info("Loaded model: %d trees", model.num_trees())

    valid_split_df   = pd.read_parquet(get_path(cfg, "valid"))
    profiles_df      = pd.read_parquet(get_path(cfg, "user_profiles"))
    meta_df          = pd.read_parquet(get_path(cfg, "meta_clean"), columns=["parent_asin", "title"])
    nlp_df_cols      = ["parent_asin", "mean_sentiment", "top_aspect"]
    nlp_cols_avail   = [c for c in nlp_df_cols if c != "parent_asin"]
    try:
        nlp_df = pd.read_parquet(
            proc / "products_nlp.parquet",
            columns=["parent_asin"] + [c for c in ["mean_sentiment", "top_aspect"] if True],
        )
    except Exception:
        nlp_df = pd.DataFrame(columns=["parent_asin", "mean_sentiment", "top_aspect"])

    meta_lookup: dict[str, str] = dict(zip(
        meta_df["parent_asin"].astype(str),
        meta_df["title"].fillna("").astype(str),
    ))
    nlp_lookup: dict[str, dict] = {}
    for _, row in nlp_df.iterrows():
        asin = str(row["parent_asin"])
        nlp_lookup[asin] = {
            "mean_sentiment": float(row.get("mean_sentiment", 0.0) or 0.0),
            "top_aspect":     str(row.get("top_aspect", "—") or "—"),
        }

    profiles_dict: dict[str, Any] = profiles_df.set_index(
        profiles_df["user_id"].astype(str)
    ).to_dict(orient="index")

    logger.info("Loaded meta (%d items), NLP lookup (%d items), profiles (%d users)",
                len(meta_lookup), len(nlp_lookup), len(profiles_dict))

    # Select 5 diverse users
    selected_users = _select_diverse_users(valid_split_df, profiles_df, seed=seed)
    user_labels = [
        "Power User", "Cold-ish User", "RPG Fan", "FPS/Shooter Fan", "Random User"
    ]
    # Pad labels if fewer than 5 were found
    user_labels = user_labels[:len(selected_users)]
    logger.info("Selected %d users: %s", len(selected_users), selected_users)

    # Stream feature rows for just these 5 users
    logger.info("Streaming features_valid.parquet for %d users ...", len(selected_users))
    t0       = time.time()
    feat_df  = _load_features_for_users(valid_path, selected_users, feature_cols)
    logger.info("Collected %d feature rows in %.0fs", len(feat_df), time.time() - t0)

    # Build per-user reports
    reports: list[dict] = []
    for uid, label in zip(selected_users, user_labels):
        logger.info("Processing user %s [%s] ...", uid, label)
        rep = _build_user_report(
            user_id      = uid,
            user_feat_df = feat_df,
            valid_df     = valid_split_df,
            meta_lookup  = meta_lookup,
            nlp_lookup   = nlp_lookup,
            model        = model,
            feature_cols = feature_cols,
            profiles_dict= profiles_dict,
            top_k        = top_k,
        )
        rep["label"] = label
        reports.append(rep)

    # Save JSON
    json_path = results_dir / "qualitative_report.json"
    json_path.write_text(
        json.dumps(reports, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Saved qualitative_report.json: %s", json_path)

    # Save TXT
    txt_report = _format_txt_report(reports, user_labels)
    txt_path = results_dir / "qualitative_report.txt"
    txt_path.write_text(txt_report, encoding="utf-8")
    logger.info("Saved qualitative_report.txt: %s", txt_path)

    # Print summary to console
    print()
    print(txt_report)

    v1_qual = PROJECT_ROOT / "blair-v1-backup" / "qualitative_report.json"
    if v1_qual.exists():
        v1_data = json.loads(v1_qual.read_text(encoding="utf-8"))
        v1_by_id = {u["user_id"]: u for u in v1_data}

        print()
        print("=" * 70)
        print("V1 vs V2 QUALITATIVE COMPARISON")
        print("=" * 70)
        for user in reports:
            uid = user["user_id"]
            v1u = v1_by_id.get(uid)
            if not v1u:
                continue
            v2_rank = user.get("gt_lambdarank_rank", "N/A")
            v1_rank = v1u.get("gt_lambdarank_rank", "N/A")
            print(f"\n{user.get('label', 'User')} ({uid[:12]})")
            print(f"  GT rank -- V1: {v1_rank}  V2: {v2_rank}", end="  ")
            if isinstance(v2_rank, int) and isinstance(v1_rank, int):
                if v2_rank < v1_rank:
                    print(f"[V2 BETTER by {v1_rank - v2_rank}]")
                elif v2_rank > v1_rank:
                    print(f"[V1 was better by {v2_rank - v1_rank}]")
                else:
                    print("[SAME]")
            else:
                print()
            v1_top3 = [
                r.get("title", "?")[:25]
                for r in v1u.get("lambdarank_top5", [])[:3]
            ]
            v2_top3 = [
                r.get("title", "?")[:25]
                for r in user.get("lambdarank_top5", [])[:3]
            ]
            print(f"  V1 top-3: {v1_top3}")
            print(f"  V2 top-3: {v2_top3}")
        print("=" * 70)

    _update_progress()
    print("stage10 complete:")
    print(f"  json: {json_path}")
    print(f"  txt:  {txt_path}")


if __name__ == "__main__":
    run()
