"""
BLAIR Recommender — Full Pipeline Runner
=========================================
Checks which stages are already complete (by verifying output files),
skips them, then runs missing stages in order.

Stages 3 and 5 (BLAIR embedding in Colab) cannot be run locally;
the script flags them and waits for manual completion.

Usage:
    python run_pipeline.py            # run all missing stages
    python run_pipeline.py --check    # check status only, no execution
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config, get_path, PROJECT_ROOT as CFG_ROOT

# ─────────────────────────────────────────────────────────────────────────────
# Stage definitions
# ─────────────────────────────────────────────────────────────────────────────

def _make_stages(cfg: dict) -> list[dict]:
    proc    = get_path(cfg, "data_processed")
    emb_dir = get_path(cfg, "data_embeddings")
    results = get_path(cfg, "outputs_results")
    cfg7    = cfg.get("stage7", {})

    return [
        {
            "num":     1,
            "name":    "Data Pipeline",
            "module":  "src.stage1_data.main",
            "check":   proc / "meta_clean.parquet",
            "colab":   False,
            "desc":    "Parse raw JSONL/CSV → 5-core filtered parquet files",
        },
        {
            "num":     2,
            "name":    "NLP Enrichment",
            "module":  "src.stage2_nlp.main",
            "check":   proc / "products_nlp.parquet",
            "colab":   False,
            "desc":    "VADER sentiment, aspect scoring, 95-column product signals",
        },
        {
            "num":     3,
            "name":    "BLAIR Embeddings",
            "module":  None,
            "check":   emb_dir / "item_embeddings.npy",
            "colab":   True,
            "desc":    "RoBERTa-large embeddings for 137k products (GPU, Colab)",
            "colab_note": "Run notebooks/stage3_embeddings_colab.ipynb in Google Colab",
        },
        {
            "num":     4,
            "name":    "FAISS Retrieval",
            "module":  "src.stage4_faiss.main",
            "check":   emb_dir / "faiss_index.bin",
            "colab":   False,
            "desc":    "Build IVFFlat FAISS index, generate candidate sets",
        },
        {
            "num":     5,
            "name":    "User Modeling",
            "module":  "src.stage5_users.main",
            "check":   proc / "user_profiles.parquet",
            "colab":   False,
            "desc":    "Build user profiles + voice docs (local); encode voice (Colab)",
            "colab_note": "Run notebooks/stage5_voice_encoding_colab.ipynb after this stage",
        },
        {
            "num":     6,
            "name":    "Feature Engineering",
            "module":  "src.stage6_features.main",
            "check":   proc / "features_train.parquet",
            "colab":   False,
            "desc":    "Generate 27-feature vectors for all (user, candidate) pairs",
        },
        {
            "num":     7,
            "name":    "LambdaRank Training",
            "module":  "src.stage7_ranker.main",
            "check":   CFG_ROOT / cfg7.get("model_path", "outputs/results/lambdarank_model.lgb"),
            "colab":   False,
            "desc":    "Train LightGBM LambdaRank + evaluate on valid set",
        },
        {
            "num":     8,
            "name":    "Evaluation",
            "module":  "src.stage8_eval.main",
            "check":   results / "test_results.json",
            "colab":   False,
            "desc":    "Official test-set evaluation: Random / FAISS / LambdaRank",
        },
        {
            "num":     9,
            "name":    "Ablation Study",
            "module":  "src.stage9_ablation.main",
            "check":   results / "ablation_results.json",
            "colab":   False,
            "desc":    "Feature-group ablation on valid set",
        },
        {
            "num":    10,
            "name":   "Qualitative Analysis",
            "module": "src.stage10_qualitative.main",
            "check":  results / "qualitative_report.json",
            "colab":  False,
            "desc":   "Human-readable recommendation examples for 5 diverse users",
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def _is_done(stage: dict) -> bool:
    return Path(stage["check"]).exists()


def _status_icon(stage: dict) -> str:
    if _is_done(stage):
        return f"{GREEN}[DONE]{RESET}"
    if stage.get("colab"):
        return f"{YELLOW}[COLAB]{RESET}"
    return f"{RED}[MISSING]{RESET}"


def print_status(stages: list[dict]) -> None:
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  BLAIR Recommender — Pipeline Status{RESET}")
    print(f"{'='*70}")
    for s in stages:
        icon = _status_icon(s)
        print(f"  Stage {s['num']:2d}  {icon:<20}  {s['name']:<25}  {s['desc']}")
        if s.get("colab") and not _is_done(s):
            print(f"            {CYAN}           --> {s.get('colab_note', 'Run in Google Colab')}{RESET}")
    print(f"{'='*70}\n")


def run_stage(stage: dict) -> bool:
    """Run a stage module. Returns True if successful."""
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}Running Stage {stage['num']}: {stage['name']}{RESET}")
    print(f"{'─'*60}")
    t0  = time.time()
    ret = subprocess.run(
        [sys.executable, "-m", stage["module"]],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0
    if ret.returncode == 0:
        print(f"{GREEN}Stage {stage['num']} complete in {elapsed:.0f}s{RESET}")
        return True
    else:
        print(f"{RED}Stage {stage['num']} FAILED (exit code {ret.returncode}) after {elapsed:.0f}s{RESET}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="BLAIR pipeline runner")
    parser.add_argument("--check", action="store_true", help="Print status only, do not run anything")
    args = parser.parse_args()

    cfg    = load_config()
    stages = _make_stages(cfg)

    print_status(stages)

    if args.check:
        return

    pending = [s for s in stages if not _is_done(s)]
    if not pending:
        print(f"{GREEN}{BOLD}All stages complete!{RESET}")
        return

    print(f"Stages to run: {[s['num'] for s in pending if not s.get('colab')]}")
    print(f"Colab stages (manual): {[s['num'] for s in pending if s.get('colab')]}")

    t_total = time.time()
    failed  = []

    for stage in pending:
        if stage.get("colab"):
            print(f"\n{YELLOW}Stage {stage['num']} ({stage['name']}) requires Colab:{RESET}")
            print(f"  {stage.get('colab_note', 'Run in Google Colab')}")
            print("  Skipping — complete manually, then re-run this script.")
            continue

        ok = run_stage(stage)
        if not ok:
            failed.append(stage["num"])
            print(f"{RED}Stopping pipeline due to Stage {stage['num']} failure.{RESET}")
            break

    # Final summary
    print(f"\n{'='*70}")
    print(f"{BOLD}PIPELINE SUMMARY  ({time.time() - t_total:.0f}s total){RESET}")
    print(f"{'='*70}")
    print_status(stages)

    if failed:
        print(f"{RED}Failed stages: {failed}{RESET}")
        sys.exit(1)
    else:
        done_count = sum(1 for s in stages if _is_done(s))
        print(f"{GREEN}Done: {done_count}/{len(stages)} stages complete.{RESET}")


if __name__ == "__main__":
    main()
