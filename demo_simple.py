"""BLAIR Hybrid Recommender — terminal demo (no models or embeddings required)."""

import os
import time

# ── ANSI colour codes ─────────────────────────────────────────────────────────
GREEN  = '\033[92m'
RED    = '\033[91m'
YELLOW = '\033[93m'
CYAN   = '\033[96m'
BOLD   = '\033[1m'
RESET  = '\033[0m'

# ── helpers ───────────────────────────────────────────────────────────────────

def _sep(char: str = '─', width: int = 70) -> str:
    return char * width


def _print_slow(lines: list[str], delay: float = 0.10) -> None:
    for line in lines:
        print(line)
        time.sleep(delay)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — SYSTEM BANNER
# ─────────────────────────────────────────────────────────────────────────────

def section_banner() -> None:
    print(f"\n{CYAN}{BOLD}", end='')
    print('╔══════════════════════════════════════════════════╗')
    print('║     BLAIR Hybrid Recommender System              ║')
    print('║     Group 11 — IIT Indore — MLSP 2026            ║')
    print('║     Amazon Video Games 2023                      ║')
    print('╚══════════════════════════════════════════════════╝')
    print(RESET)

    stats = [
        ('Products indexed',    '137,269'),
        ('User reviews',        '814,586'),
        ('Users profiled',       '94,762'),
        ('Items (5-core)',        '25,612'),
        ('Training queries',    '625,062'),
        ('Test queries',          '94,762'),
        ('Features per pair',         '30'),
        ('NLP signals/product',       '95'),
    ]
    for label, value in stats:
        print(f"  {label:<26} {CYAN}{BOLD}{value:>10}{RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PIPELINE OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────

def section_pipeline() -> None:
    print(f"{YELLOW}{BOLD}── Pipeline Overview ──{RESET}\n")

    stages = [
        ('Stage 1',  'Data Pipeline',       '137K products, 814K reviews, 5-core filtering'),
        ('Stage 2',  'NLP Enrichment',       '95 signals/product: sentiment, aspects, temporal, reality gap'),
        ('Stage 3',  'BLAIR Embeddings',     '1024-dim CLS token, L2 normalized'),
        ('Stage 4',  'HNSW Retrieval',       '100% self-recall@200, 0.75ms latency, 20x faster than IVFFlat'),
        ('Stage 5',  'User Modeling',        '4 embedding profiles + voice document + cold-start tiers'),
        ('Stage 6',  'Feature Engineering',  '30 features, 125M rows, graded relevance 1-5 stars'),
        ('Stage 7',  'LambdaRank',           'LightGBM, NDCG-optimized'),
        ('Stage 8',  'Evaluation',           '94,762 test queries, bootstrap CI'),
        ('Stage 9',  'Ablation Study',       '10 configurations'),
        ('Stage 10', 'Qualitative',          '5 diverse users'),
    ]

    for stage, name, detail in stages:
        tick    = f"{GREEN}{BOLD}✓{RESET}"
        label   = f"{BOLD}{stage:<9}{RESET}  {name:<22}"
        note    = f"({detail})"
        print(f"  {tick} {label} {note}")
        time.sleep(0.10)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MAIN RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def section_results() -> None:
    print(f"{YELLOW}{BOLD}── Test Set Results (94,762 queries) ──{RESET}\n")

    hdr = f"  {'System':<38} {'NDCG@1':>7} {'NDCG@5':>7} {'NDCG@10':>8} {'MRR':>7} {'HR@10':>7}"
    print(hdr)
    print('  ' + _sep('─', 70))

    rows = [
        ('Random Baseline',                         '0.0052', '0.0150', '0.0229', '0.0297', '0.0500', False),
        ('FAISS+HNSW Baseline',                     '0.0004', '0.0013', '0.0018', '0.0066', '0.0038', False),
        ('LambdaRank V1 (pretrained BLAIR)',         '0.9709', '0.9729', '0.9740', '0.9733', '0.9783', False),
        ('LambdaRank V2 (custom BLAIR)',             '0.9154', '0.9234', '0.9270', '0.9242', '0.9426', True),
    ]

    for name, n1, n5, n10, mrr, hr, highlight in rows:
        colour = f"{GREEN}{BOLD}" if highlight else ''
        tag    = f" {GREEN}{BOLD}←{RESET}" if highlight else ''
        print(f"  {colour}{name:<38}{RESET}{tag}")
        nums = f"  {'':38} {n1:>7} {n5:>7} {n10:>8} {mrr:>7} {hr:>7}"
        print(f"{colour}{nums}{RESET}")

    print()
    print(f"  {CYAN}V1: pretrained BLAIR (all Amazon categories) — model converged in   5 rounds{RESET}")
    print(f"  {CYAN}V2: custom BLAIR (Video Games domain only)   — model trained for 135 rounds{RESET}")
    print()
    print(f"  {BOLD}Key insight: FAISS baseline < Random baseline{RESET}")
    print(f"  {BOLD}→ Semantic similarity alone is insufficient for ranking{RESET}")
    print(f"  {BOLD}→ LTR over rich features is essential{RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — ABLATION COMPARISON V1 vs V2
# ─────────────────────────────────────────────────────────────────────────────

def section_ablation() -> None:
    print(f"{YELLOW}{BOLD}── Ablation Study: What happens when features are removed? ──{RESET}\n")

    hdr = f"  {'Feature Group':<28} {'V1 Drop':>9} {'V2 Drop':>9}   Status"
    print(hdr)
    print('  ' + _sep('─', 62))

    ablation_rows = [
        ('Retrieval (f01-f04)',   '-73.99%', '-58.76%', 'Dominant in both',    False),
        ('Product NLP (f05-f12)',  '-1.33%',  '-1.36%', 'Consistent',          False),
        ('Aspect Features',        '-0.00%',  '-0.18%', 'V2 UNLOCKED',         True),
        ('Personalization',        '-0.01%',  '-0.34%', 'V2 UNLOCKED',         True),
        ('User Voice Cosine',      '-0.01%',  '-0.39%', 'V2 UNLOCKED',         True),
        ('ALL NLP Features',       '-0.01%',  '-0.52%', 'V2 UNLOCKED',         True),
        ('User Embeddings',        '-0.03%',  '-0.11%', 'V2 UNLOCKED',         True),
    ]

    for name, v1, v2, status, unlocked in ablation_rows:
        status_str = (f"{GREEN}{BOLD}[{status}]{RESET}" if unlocked
                      else f"{status}")
        print(f"  {name:<28} {v1:>9} {v2:>9}   {status_str}")

    print()
    print(f"  {BOLD}{CYAN}V1: Only retrieval features contributed. NLP features near-zero.{RESET}")
    print(f"  {BOLD}{CYAN}V2: Custom BLAIR embeddings unlocked MULTIPLE feature contributions!{RESET}")
    print(f"  {BOLD}{CYAN}→ Aspect features: 0.00% → 0.18% drop (18x more impactful){RESET}")
    print(f"  {BOLD}{CYAN}→ User Voice: 0.01% → 0.39% drop (39x more impactful){RESET}")
    print(f"  {BOLD}{CYAN}→ ALL NLP: 0.01% → 0.52% drop (52x more impactful){RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — QUALITATIVE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def section_qualitative() -> None:
    print(f"{YELLOW}{BOLD}── Qualitative Analysis: 5 Real Users ──{RESET}")
    print(f"{YELLOW}{BOLD}── Did the system find what they actually bought? ──{RESET}\n")

    users = [
        {
            'label':   'Power User (471 interactions)',
            'history': 'Batman Arkham Origins, Rayman Origins, WWE Stars',
            'truth':   'Batman Brave & Bold - Nintendo DS',
            'faiss':   'Tekken 4, Dragon\'s Lair, Scrapland...',
            'lr':      'Batman Brave & Bold - Nintendo DS',
            'lr_rank': 'RANK 1',
        },
        {
            'label':   'Cold-ish User (5 interactions)',
            'history': 'Driver, Dave Mirra BMX, WWF Warzone',
            'truth':   'Skate 2: Platinum Hits Edition',
            'faiss':   'Monster Jam, Rayman Arena, WWE X8...',
            'lr':      'Skate 2: Platinum Hits Edition',
            'lr_rank': 'RANK 1',
        },
        {
            'label':   'RPG Fan (13 interactions)',
            'history': 'Assassins Creed Brotherhood, inFAMOUS 2, MK',
            'truth':   'Batman Arkham Origins PS3',
            'faiss':   'Blood Omen 2, Quake 3, Legaia 2...',
            'lr':      'Batman Arkham Origins PS3',
            'lr_rank': 'RANK 1',
        },
        {
            'label':   'FPS Fan (4 interactions)',
            'history': 'Rayman Legends, God of War 3, Hasbro Pack',
            'truth':   'Rock Band 4 Wireless Guitar Bundle',
            'faiss':   'Namco Museum, Pac Man World, Spongebob',
            'lr':      'Rock Band 4 Wireless Guitar Bundle',
            'lr_rank': 'RANK 1',
        },
        {
            'label':   'Random User (3 interactions)',
            'history': 'Guitar Hero Kit, Sonic Classic, Kingdom Hearts',
            'truth':   'Taiko no Tatsujin Switch Controller',
            'faiss':   'Harvest Moon DS, Monster Hunter...',
            'lr':      'Taiko no Tatsujin Switch Controller',
            'lr_rank': 'RANK 1',
        },
    ]

    W = 63
    for u in users:
        fail_tag = f"{RED}{BOLD}[FAIL]{RESET}"
        rank_tag = f"{GREEN}{BOLD}[{u['lr_rank']}]{RESET}"
        print(f"  ┌{'─' * W}┐")
        print(f"  │ {BOLD}{u['label']:<{W-2}}{RESET} │")
        print(f"  │ History: {u['history']:<{W-10}} │")
        print(f"  │ Ground truth: {u['truth']:<{W-15}} │")
        print(f"  │ FAISS result: {u['faiss']:<{W-26}} {fail_tag} │")
        print(f"  │ LambdaRank:   {u['lr']:<{W-26}} {rank_tag} │")
        print(f"  └{'─' * W}┘")
        print()
        time.sleep(0.30)

    print(f"  {BOLD}{GREEN}LambdaRank Hit Rate@10:  5/5 (100%){RESET}")
    print(f"  {BOLD}{GREEN}FAISS Hit Rate@10:       0/5 (0%){RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — KEY SCIENTIFIC FINDING
# ─────────────────────────────────────────────────────────────────────────────

def section_finding() -> None:
    print(f"{BOLD}{YELLOW}{'═' * 46}{RESET}")
    print(f"{BOLD}{YELLOW}KEY SCIENTIFIC FINDING{RESET}")
    print(f"{BOLD}{YELLOW}{'═' * 46}{RESET}\n")

    print("  V1 — Pretrained BLAIR (all Amazon categories):")
    _print_slow([
        "    • Model converges in 5 rounds",
        "    • Only retrieval features matter",
        "    • NLP features contribute ~0%",
        "    • Reason: generic embeddings → weak negative diversity",
    ], delay=0.10)
    print()

    print("  V2 — Custom BLAIR (Video Games domain only):")
    _print_slow([
        "    • Model trains for 135 rounds (27x longer!)",
        "    • Multiple feature groups now contribute",
        "    • Aspect features: 18x more impactful",
        "    • User Voice: 39x more impactful",
        "    • ALL NLP: 52x more impactful",
        "    • Reason: domain-specific embeddings → richer signal",
    ], delay=0.10)
    print()

    print(f"  {BOLD}{CYAN}CONCLUSION: Domain-specific BLAIR embeddings unlock{RESET}")
    print(f"  {BOLD}{CYAN}the full potential of our 30-feature LTR pipeline.{RESET}")
    print(f"  {BOLD}{CYAN}This validates our multi-aspect contrastive training approach.{RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — HNSW ENGINEERING ACHIEVEMENT
# ─────────────────────────────────────────────────────────────────────────────

def section_hnsw() -> None:
    print(f"{YELLOW}{BOLD}── Retrieval Engineering: IVFFlat → HNSW ──{RESET}\n")

    hdr = f"  {'Metric':<24} {'IVFFlat':>14} {'HNSW':>14} {'Improvement':>14}"
    print(hdr)
    print('  ' + _sep('─', 70))

    hnsw_rows = [
        ('Self-recall@200',     '~4%',           '100%',          '25x better'),
        ('Query latency',       '14-16 ms',       '0.75 ms',       '20x faster'),
        ('Forced positives',    '95.9%',          '21.6%',         '4.4x better'),
        ('Model rounds',        '1',              '135',           '135x more'),
        ('Index build time',    'N/A',            '68.7 seconds',  'fast!'),
    ]

    for metric, ivf, hnsw, impr in hnsw_rows:
        print(f"  {metric:<24} {ivf:>14} {CYAN}{BOLD}{hnsw:>14}{RESET} {GREEN}{impr:>14}{RESET}")

    print()
    print(f"  {GREEN}Root cause identified: IVFFlat low recall caused cascade failure{RESET}")
    print(f"  {GREEN}Fix: HNSW with M=32, efConstruction=200, efSearch=128{RESET}")
    print(f"  {GREEN}Result: Natural recall improved from 4% to 78.4%{RESET}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def section_summary() -> None:
    print(f"{BOLD}{CYAN}{'═' * 46}{RESET}")
    print(f"{BOLD}{CYAN}BLAIR Hybrid Recommender — Final Summary{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 46}{RESET}\n")

    lines = [
        f"  {GREEN}{BOLD}✓{RESET} 10-stage research pipeline — fully implemented",
        f"  {GREEN}{BOLD}✓{RESET} 3 novel contributions:",
        f"      1. Review-Augmented BLAIR Embeddings (95 NLP signals)",
        f"      2. Rich User Voice Modeling (4 profiles + semantic fingerprint)",
        f"      3. Tiered Cold-Start Solution (0 to warm user)",
        f"  {GREEN}{BOLD}✓{RESET} HNSW: 100% recall@200, 0.75ms latency",
        f"  {GREEN}{BOLD}✓{RESET} LambdaRank V1: NDCG@10 = {CYAN}{BOLD}0.9740{RESET}",
        f"  {GREEN}{BOLD}✓{RESET} LambdaRank V2: NDCG@10 = {CYAN}{BOLD}0.9270{RESET} (135 rounds, features unlocked)",
        f"  {GREEN}{BOLD}✓{RESET} Qualitative: 5/5 users ranked correctly at position 1",
        f"  {GREEN}{BOLD}✓{RESET} GitHub: github.com/ys-0212/blair-recommender",
    ]
    _print_slow(lines, delay=0.12)
    print()
    print(f"  {GREEN}{BOLD}Thank you!{RESET}\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    os.system('cls' if os.name == 'nt' else 'clear')
    t_start = time.time()

    section_banner()
    time.sleep(0.50)

    section_pipeline()
    time.sleep(0.50)

    section_results()
    time.sleep(0.50)

    section_ablation()
    time.sleep(0.50)

    section_qualitative()
    time.sleep(0.50)

    section_finding()
    time.sleep(0.50)

    section_hnsw()
    time.sleep(0.50)

    section_summary()

    elapsed = time.time() - t_start
    print(f"  {YELLOW}Total runtime: {elapsed:.1f}s{RESET}\n")


if __name__ == '__main__':
    main()
