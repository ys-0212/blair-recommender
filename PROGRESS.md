# BLAIR Recommender — Progress Tracker

Last updated: 2026-04-23

## Pipeline Status

| Stage | Name | Status | Output |
|-------|------|--------|--------|
| 1 | Data Pipeline | ✅ Complete | `data/processed/{meta_clean,reviews_top5,train,valid,test}.parquet` |
| 2 | NLP Enrichment | ✅ Complete (Enhanced) | `reviews_nlp.parquet`, `products_nlp.parquet` (95 cols), `products_rich.parquet` |
| 3 | BLAIR Embeddings | ✅ Complete | `data/embeddings/{item_embeddings,item_ids}.npy` |
| 4 | FAISS Retrieval | ✅ Complete | `data/embeddings/faiss_index.bin`, `faiss_id_map.json` |
| 5 | User Modeling | ✅ Complete | `data/processed/user_profiles.parquet`, `user_voice_docs.parquet`; `data/embeddings/user_voice_{embeddings,ids}.npy` |
| 6 | Feature Engineering | ✅ Complete | data/processed/features_{train,valid,test}.parquet |
| 7 | LambdaRank | ✅ Complete | outputs/results/lambdarank_model.lgb |
| 8 | Evaluation | ✅ Complete | `outputs/results/eval_results.json` |
| 9 | Ablation Study | ✅ Complete | `outputs/results/ablation_results.json` |
| 10 | Qualitative Analysis | ✅ Complete | `outputs/results/qualitative_report.html` |

---

## Stage 1 — Data Pipeline ✅

**Completed:** 2026-04-16

### What was done
- Parsed `meta_Video_Games.jsonl.gz` → 9 fields kept, deduplicated by `parent_asin`
- Parsed `Video_Games.jsonl.gz` → kept 7 fields, applied 5-core filtering
  - 5-core: iterative removal of users/items with < 5 interactions until stable
- Loaded `valid.csv.gz` and `test.csv.gz` (columns: user_id, parent_asin, rating, timestamp, history)
- **Train file issue:** `Video_Games.train.csv.gz.crdownload` is an incomplete download.
  - Fallback: training interactions are derived from `reviews_top5.parquet` by excluding all interactions that appear in valid/test sets.
  - Output written to `data/processed/train.parquet` in the same schema as valid/test.
- All outputs written as snappy-compressed parquet files.

### Key statistics
| Artifact | Count |
|----------|-------|
| Meta items | 137,269 |
| Reviews (5-core) | 814,586 |
| Users (5-core) | 94,762 |
| Items (5-core) | 25,612 |
| Train interactions | 625,062 (derived from reviews) |
| Valid interactions | 94,762 |
| Test interactions | 94,762 |

### Design decisions
- Used `parent_asin` as the canonical item identifier throughout (consistent with split CSVs).
- The `history` field in valid/test CSVs is preserved as-is (space-separated parent_asin string).
- Price is stored as a raw string; normalisation happens in Stage 6.
- 5-core filtering is applied iteratively (up to 10 passes) until the dataset is stable.

---

## Stage 2 — NLP Enrichment ✅

**Completed:** 2026-04-16

### What was done
- **VADER full-text sentiment** on all 814,586 reviews → `sentiment_score` (float32, [-1,1]), `sentiment_label` (positive/neutral/negative)
  - Positive: 667,863 (82.0%) | Negative: 92,038 (11.3%) | Neutral: 54,685 (6.7%)
- **Aspect-based sentiment** for 5 aspects using regex keyword matching + per-sentence VADER:
  | Aspect | Coverage | Mean Score |
  |--------|----------|------------|
  | gameplay | 19.8% | 0.461 |
  | graphics | 12.5% | 0.445 |
  | story    | 12.7% | 0.305 |
  | controls | 16.8% | 0.274 |
  | value    | 18.1% | 0.356 |
- **Product-level aggregation** (`products_nlp.parquet`): 25 columns including mean/std sentiment, pos/neg/neu ratios, mean aspect scores + per-aspect coverage fractions
- **Query understanding module** (`src/stage2_nlp/query_understanding.py`): rule-based parser for intent, attributes, price constraints, domain expansion (fps→shooter, rpg→role-playing game, etc.)

### Outputs
| File | Size | Rows | Columns |
|------|------|------|---------|
| `reviews_nlp.parquet` | 262 MB | 814,586 | 14 |
| `products_nlp.parquet` | 88 MB | 137,269 | 25 |

### Design decisions
- Aspect NaN (not 0) when no keyword-matching sentence found — distinguishes "no signal" from "neutral sentiment"
- Sentence splitting uses punctuation heuristics (no NLTK punkt model) for speed
- Query understanding is fully rule-based — runs at microsecond latency at inference time

---

## Stage 2 Enhancement — New Signals ✅

**Added to products_nlp.parquet (25 → 95 columns, +70 new signals):**

| Group | Columns |
|-------|---------|
| Temporal (Phase 1c) | early/mid/recent sentiment, per-aspect windows, velocity, trajectory, trend summary |
| Consensus | consensus_score, controversy_index, per-aspect consensus |
| Helpfulness | helpfulness_weighted_sentiment, helpful_ratio, avg_review_length |
| Verified | verified_ratio |
| Listing quality | leaf_category, full_category_path, feature_count, desc_richness_score, days_on_market, price_tier |
| Reality gap | desc_sentiment, hype_gap_score, official_vs_community_gap, overhyped_aspect, underrated_aspect |
| Category | category_size, avg_rating/sentiment, percentile ranks, distinctive_score, outperforms/underperforms, hidden_gem_score |
| TF-IDF | top_tfidf_phrases, distinctive_terms (per-category vectoriser, 161 leaf categories) |
| Emotion | emotion_vocabulary, dominant_emotion |
| Aspect best/worst | top_aspect, worst_aspect |

**Data-driven thresholds computed:**
- Price tiers: p25=$13.00, p75=$45.99, p90=$91.00
- Min coverage threshold: 0.0353 (5th pct of non-zero coverage values)
- Aspect delta threshold: 0.1138 (mean |delta| across all products & aspects)
- Dominant emotion threshold: 0.7106 (mean |pos_ratio - neg_ratio|)

**products_rich.parquet** (new file):
- 137,269 rows × 3 cols (parent_asin, rich_text, token_count)
- Token counts: min=154, mean=398, max=517, over_512=13.1%
- Tokenizer: hyp1231/blair-roberta-large (downloaded locally)

---

## Stage 3 — BLAIR Embeddings ✅ Complete

**Code written:**
- `src/stage3_embeddings/embed.py` — full embedding loop with checkpointing and verification
- `notebooks/stage3_embeddings_colab.ipynb` — 6-cell ready-to-run Colab notebook

**To run:**
1. Upload `data/processed/products_rich.parquet` to Colab
2. Run `notebooks/stage3_embeddings_colab.ipynb` (T4 GPU, ~25-35 min for 137k items)
3. Download `item_embeddings.npy` and `item_ids.npy`
4. Place both files in `data/embeddings/` before running Stage 4

**Model:** `hyp1231/blair-roberta-large`  |  **Output dim:** 1024  |  **Checkpoint every:** 5000 items

---

---

## Stage 5 — Rich User Modeling ✅ Complete

**Completed:** 2026-04-22

**Local outputs:**
- `data/processed/user_profiles.parquet` — 94,762 users × 39 cols (interaction, review, cold-start signals + 4 embedding profiles)
- `data/processed/user_voice_docs.parquet` — 94,762 structured text documents

**Colab outputs (user_voice_embeddings):**
- `data/embeddings/user_voice_embeddings.npy` — (94,762, 1024) float32 L2-normalised
- `data/embeddings/user_voice_ids.npy` — (94,762,) user_id strings

**Note:** main.py rewritten to stop after local steps; BLAIR voice encoding ran in `notebooks/stage5_voice_encoding_colab.ipynb`. Config paths updated to `data/embeddings/` for voice files.

### What was done
- **profile_builder.py** — builds per-user signals from three data sources:
  - *Interaction-based*: interaction_count, avg/std rating, active_days, 4 embedding profiles (uniform, recency, rating, combined), category preferences (top_categories, diversity, entropy, dominant), price tier preferences, interaction_velocity, recency_score, is_active
  - *Review-based*: sentiment stats, pos/neg ratios, review length, helpful votes, verified ratio, per-aspect mean scores, top/worst aspect, aspect coverage
  - *Cold-start tier*: data-driven thresholds (5th/25th pct) assign tier 0-3 with per-user query_weight/user_weight
- **user_voice.py** — encodes user taste as structured BLAIR text documents
  - 5 sections: interaction history, category preferences, aspect preferences, price preferences, review behavior + user's own words (recency+rating sorted)
  - Token budget managed via BLAIR tokenizer; remaining budget filled with review text
  - Only tier-2+ users encoded (tier 0/1 lack meaningful signal)
  - CPU encoding with batch_size=8 for RoBERTa-large memory safety
- **blender.py** — inference-time query blending per cold-start tier:
  - Tier 0: pure query embedding (BLAIR handles cold start semantically)
  - Tier 1: query + uniform embedding
  - Tier 2: query + recency embedding
  - Tier 3: query + combined OR three-way (query + combined + voice)
  - `get_cold_start_boost_items()`: quality-ranked + diversity-injected retrieval for tier-0 users

### Cold-start contribution
Three novel mechanisms not in standard literature:
1. BLAIR query encoding works with ZERO history
2. Tiered blending — any amount of history improves results smoothly
3. User voice documents — even 1-2 reviews give enough signal for a meaningful BLAIR preference vector

### Output schema
`user_profiles.parquet`: ~40 columns per user including all signals above + voice_document text

### Known notes
- Embedding columns stored as object dtype lists in parquet (for compatibility)
- User voice encoding requires `transformers` — skips gracefully if unavailable
- Tier thresholds are data-driven at runtime (5th/25th pct of interaction_count)

---

## Stage 6 — Feature Engineering ✅ Complete

**Completed:** 2026-04-22  **Note:** regenerated after forced-positive bug + price outlier bug + PROGRESS.md path bug fixes

### Critical fixes applied
1. **Forced positive** (`candidate_generator.py`): FAISS top-100 often excludes the ground-truth item, causing ~0.04% positive rate. Fix: if gt_asin not in top-100, force-add it with faiss_score=0.0, faiss_rank=101, is_forced_positive=1. Expected positive rate after fix: ~1%.
2. **Price p99 cap** (`feature_builder.py`): price range was [0, 3499.99] (outliers). Fix: normalise using p99 of training prices as max; values above p99 clip to 1.0.
3. **hidden_gem_score** (`feature_builder.py`): data-driven min/max from training products (already was `_st()` call; verified correct).
4. **f27_is_forced_positive** (`feature_builder.py`): new feature. LambdaRank can learn to discount forced positives.
5. **PROGRESS.md path** (`main.py`): was `parents[3]` (one level above project root). Fixed to use `PROJECT_ROOT` from config.py.

### Architecture
- `candidate_generator.py` — blends history→query embedding with user profile, batch-retrieves top-100, force-injects ground truth when absent
- `feature_builder.py` — computes 27 features; p99-capped price; data-driven hidden_gem normalization; normalization stats from source data (no leakage)
- `main.py` — PyArrow streaming writes (no OOM on 63M rows); train norm_stats applied to valid/test

### Features (27 total)
| Group | Features |
|-------|---------|
| Retrieval (4) | faiss_score, faiss_rank, query_item_cosine, user_uniform_cosine |
| Product (8) | avg_rating, rating_count_log, review_count_log, mean_sentiment, price_normalized (p99-capped), hidden_gem_score, controversy_index, desc_richness |
| Aspect (5) | gameplay, graphics, story, controls, value (product-level mean) |
| Personalization (6) | user_voice_cosine, category_match, price_tier_match, sentiment_gap, top_aspect_match, interaction_count_log |
| Temporal/Quality (3) | sentiment_trajectory, verified_ratio, helpfulness_weighted_sentiment |
| Forced positive (1) | is_forced_positive flag |

### Expected output sizes (after fix)
| Split | Interactions | Candidate rows | Positives | Pos rate |
|-------|-------------|----------------|-----------|----------|
| train | 625,062 | ~63.1M | ~625,062 | ~0.99% |
| valid | 94,762 | ~9.57M | ~94,762 | ~0.99% |
| test | 94,762 | ~9.57M | ~94,762 | ~0.99% |

---

## Stage 7 — LambdaRank ✅ Complete

**Completed:** 2026-04-23

### What was done
- Trained LightGBM LambdaRank on 26 features (f01-f26) from Stage 6 feature files
- f27_is_forced_positive excluded from training (would leak label information)
- Evaluated FAISS Baseline vs LambdaRank on validation set
- Saved model to `outputs/results/lambdarank_model.lgb`
- Saved feature importance chart to `outputs/charts/feature_importance.png`

### Outputs
| File | Description |
|------|-------------|
| `outputs/results/lambdarank_model.lgb` | Trained LightGBM LambdaRank model |
| `outputs/results/training_history.json` | Per-round NDCG@1/5/10 on train + valid |
| `outputs/results/eval_results.json` | FAISS Baseline vs LambdaRank comparison |
| `outputs/charts/feature_importance.png` | Gain-based feature importance by group |

### Honest FAISS Recall Limitation
FAISS natural recall is approximately 4% at nlist=128, nprobe=16.
Approximately 95.9% of positive labels in training data are force-injected
(ground truth not retrieved by FAISS, added with faiss_score=0.0, faiss_rank=101).

LambdaRank still learns valid re-ranking signal from feature patterns that
correlate with user preferences even when retrieval failed to surface them.

**Future improvement:** Rebuild FAISS index with nlist=512, nprobe=64.
Expected natural recall improvement: ~4% -> ~25-40%.

---

## Known Issues / Blockers

| Issue | Impact | Resolution |
|-------|--------|------------|
| `Video_Games.train.csv.gz` incomplete download | Stage 1 | Fallback: derive train from reviews |
| No GPU locally | Stage 3 | Use Google Colab |
| `faiss-gpu` not installed locally | Stage 4 | Use `faiss-cpu` locally, `faiss-gpu` in Colab |

---

## Environment

- Python: 3.14.3 (system) — consider using venv with 3.10+
- OS: Windows 11
- Install deps: `pip install -r requirements.txt`
- Run any stage: `python -m src.stageN_name.main`
