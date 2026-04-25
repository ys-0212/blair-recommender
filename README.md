# BLAIR Hybrid Recommender System

A research-grade hybrid recommendation system for the Amazon Video Games dataset, built for the MLSP course. Combines semantic BLAIR embeddings, rich NLP signals from user reviews, cold-start-aware user modeling, and Learning-to-Rank reranking into a 10-stage end-to-end pipeline.

---

## Novel Contributions

1. **Review-Augmented BLAIR Embeddings** — Each product is represented by a 1024-dim `hyp1231/blair-roberta-large` vector over a rich text document combining title, description, community sentiment, aspect scores, temporal trends, and controversy signals (95 NLP features per product, not just metadata).

2. **Rich User Voice Modeling** — User preferences are encoded as structured BLAIR text documents capturing interaction history, category preferences, aspect affinities, price sensitivity, and the user's own review language. Produces a 1024-dim user voice embedding for fine-grained personalization.

3. **Cold-Start Solution via Tiered Blending** — A 4-tier cold-start framework (0 = no history, 3 = warm user) smoothly interpolates between pure BLAIR semantic search and personalized reranking. The system degrades gracefully to zero history instead of failing.

4. **26-Feature LambdaRank Pipeline** — LightGBM LambdaRank directly optimizes NDCG over 26 features spanning retrieval quality, product signals, aspect alignment, personalization, and temporal quality signals. All normalization stats are computed from training data only (no leakage).

---

## Architecture

```
data/raw/
   │
   ├─► [Stage 1: Data Pipeline]
   │       Parse JSONL/CSV, 5-core filtering
   │       → data/processed/{meta_clean, reviews_top5, train, valid, test}.parquet
   │
   ├─► [Stage 2: NLP Enrichment]
   │       VADER sentiment, 5 aspects, 95 product signals, TF-IDF
   │       → products_nlp.parquet (137k items × 95 cols)
   │
   ├─► [Stage 3: BLAIR Embeddings]  *** Google Colab (GPU) ***
   │       hyp1231/blair-roberta-large, 1024-dim CLS token
   │       → item_embeddings.npy (137k × 1024)
   │
   ├─► [Stage 4: FAISS Retrieval]
   │       IVFFlat index, cosine similarity, top-100 candidates
   │       → faiss_index.bin
   │
   ├─► [Stage 5: User Modeling]  (local + Colab for voice)
   │       39-col profiles, cold-start tiers, voice documents
   │       → user_profiles.parquet, user_voice_embeddings.npy
   │
   ├─► [Stage 6: Feature Engineering]
   │       27 features per (user, candidate) pair, streaming writes
   │       → features_{train, valid, test}.parquet  (~63M / 9.5M / 9.5M rows)
   │
   ├─► [Stage 7: LambdaRank]
   │       LightGBM LambdaRank, NDCG-optimised, early stopping
   │       → lambdarank_model.lgb
   │
   ├─► [Stage 8: Evaluation]
   │       Random / FAISS / LambdaRank comparison on test set
   │       → test_results.json, system_comparison.png
   │
   ├─► [Stage 9: Ablation Study]
   │       Feature-group ablation, 8 configurations
   │       → ablation_results.json, ablation_chart.png
   │
   └─► [Stage 10: Qualitative Analysis]
           5 diverse users, FAISS vs LambdaRank top-5 examples
           → qualitative_report.json, qualitative_report.txt
```

---

## Pipeline Stages

| Stage | Name | Input | Key Output | Notes |
|-------|------|-------|-----------|-------|
| 1 | Data Pipeline | raw JSONL/CSV | `train/valid/test.parquet` | 5-core filtering |
| 2 | NLP Enrichment | reviews + meta | `products_nlp.parquet` | 95 signals/product |
| 3 | BLAIR Embeddings | `products_rich.parquet` | `item_embeddings.npy` | **Colab GPU** |
| 4 | FAISS Retrieval | embeddings | `faiss_index.bin` | IVFFlat, cosine |
| 5 | User Modeling | reviews + profiles | `user_profiles.parquet` | + **Colab** for voice |
| 6 | Feature Engineering | all Stage 2-5 outputs | `features_*.parquet` | 27 features, 63M rows |
| 7 | LambdaRank | features | `lambdarank_model.lgb` | LightGBM |
| 8 | Evaluation | test features + model | `test_results.json` | 3-system comparison |
| 9 | Ablation Study | valid features + model | `ablation_results.json` | 8 configurations |
| 10 | Qualitative | valid + metadata | `qualitative_report.txt` | 5 user examples |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Check pipeline status
python run_pipeline.py --check

# Run all missing stages (skips completed, flags Colab stages)
python run_pipeline.py

# Run individual stages
python -m src.stage1_data.main
python -m src.stage2_nlp.main
# ... (Stages 3 and 5 require Google Colab — see notebooks/)
python -m src.stage4_faiss.main
python -m src.stage6_features.main
python -m src.stage7_ranker.main
python -m src.stage8_eval.main
python -m src.stage9_ablation.main
python -m src.stage10_qualitative.main
```

**Colab stages:**
- Stage 3: Upload `data/processed/products_rich.parquet` → run `notebooks/stage3_embeddings_colab.ipynb` → download `item_embeddings.npy`, `item_ids.npy` to `data/embeddings/`
- Stage 5 (voice): Run `notebooks/stage5_voice_encoding_colab.ipynb` after local Stage 5 → download `user_voice_embeddings.npy`, `user_voice_ids.npy` to `data/embeddings/`

---

## Results

Evaluated on held-out test set (94,762 queries):

| System | NDCG@1 | NDCG@5 | NDCG@10 | MRR | HR@10 |
|--------|--------|--------|---------|-----|-------|
| Random Baseline | — | — | — | — | — |
| FAISS Baseline | — | — | — | — | — |
| **LambdaRank (Ours)** | **—** | **—** | **—** | **—** | **—** |

*(Filled after running Stage 8.)*

**Honest Limitation:** FAISS natural recall is approximately 4% at `nlist=128, nprobe=16`. Approximately 95.9% of positive training labels are force-injected (ground truth not retrieved by FAISS). All absolute metric values are inflated relative to a system with higher natural recall. Relative comparisons between systems are internally valid.

**Future improvement:** Rebuild the FAISS index with `nlist=512, nprobe=64`. Expected natural recall improvement: ~4% → 25-40%.

---

## Dataset

**Amazon Video Games Reviews 2023** (McAuley Lab)

| Statistic | Count |
|-----------|-------|
| Products (meta) | 137,269 |
| Reviews (raw) | 814,586 |
| Users after 5-core | 94,762 |
| Items after 5-core | 25,612 |
| Train interactions | 625,062 |
| Validation interactions | 94,762 |
| Test interactions | 94,762 |

---

## Known Limitations & Future Work

- **FAISS recall ~4%** at current settings (`nlist=128, nprobe=16`). Rebuild with `nlist=512, nprobe=64` for significant recall improvement without major latency cost.
- **Model stopped at round 1** due to trivially easy re-ranking task (forced positives dominate). With higher natural recall (more negative variety), the model will train for more rounds and learn finer distinctions.
- **No online learning.** User profiles are static; a production system would update profiles incrementally.
- **Rule-based aspect detection.** Replacing keyword matching with a fine-tuned aspect extractor would improve signal quality.

---

## Team

*[Your names here]*

---

## Citation

If you use this work, please also cite the BLAIR paper:

```bibtex
@article{hou2023blair,
  title     = {BLAIR: Recommendation with Behavioral LLM And Item Representations},
  author    = {Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal   = {arXiv preprint arXiv:2304.15097},
  year      = {2023}
}
```

**Dataset:**
```bibtex
@article{hou2024bridging,
  title   = {Bridging Language and Items for Retrieval and Recommendation},
  author  = {Hou, Yupeng and others},
  journal = {arXiv preprint arXiv:2403.03952},
  year    = {2024}
}
```
