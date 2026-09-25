# 🏆 Team Approach: Business Entity Resolution (AWS/Amazon ML Challenge 2026)

## Overview

Our solution is a **two-stage hybrid ensemble** combining a classical ML (XGBoost) pipeline for general entity matching with a GPU-accelerated Multilingual Transformer pipeline specifically targeting cross-script Indian entities (Indic languages).

**Best Leaderboard Score: `0.418` (Macro F0.5)**

---

## Stage 1: XGBoost Re-Ranker Pipeline (`run_dgx_pipeline.py`)

### Problem
Naively matching 1.73 million Source 1 entities against all Source 2 and Source 3 candidates is computationally infeasible (O(n²) complexity over 1.7M+ records).

### Solution: Heuristic Blocking + Dense Retrieval + XGBoost

**1. Multi-Key Blocking (Candidate Generation)**

We first prune the candidate space using 4 fast inverted-index lookups:
- **Exact Name Match**: `(country, cleaned_name)` → matches 1:1 entity lookups
- **Pincode Match**: `(country, pincode)` → co-located businesses
- **First-Word Match**: `(country, first_word_of_name)` → partial name overlap
- **Prefix Match**: `(country, name[:3])` → handles abbreviations

**2. Dense Semantic Fallback (SentenceTransformer)**

If blocking yields zero candidates, we fall back to dense semantic retrieval using `all-MiniLM-L6-v2` with GPU-accelerated matrix multiplication against country-partitioned FAISS-style tensors. Only candidates with cosine similarity ≥ 0.80 are added.

**3. XGBoost Feature Extraction**

For each (S1, candidate) pair, we extract a 6-dimensional feature vector:
| Feature | Description |
|---|---|
| `fuzz.ratio` | Character-level Levenshtein similarity |
| `fuzz.partial_ratio` | Partial-string fuzzy match |
| `fuzz.token_set_ratio` | Token-based fuzzy match (ignores order) |
| `fuzz.token_sort_ratio` | Sorted token fuzzy match |
| `jaro_winkler` | Jaro-Winkler distance (strong for short names) |
| `address_token_overlap` | Shared word count between addresses |
| `name_length_diff` | Absolute difference in name length |

**4. XGBoost Inference**

- Model: Pre-trained `xgb_reranker.json` (trained on labeled entity pairs from `train_ground_truth.tsv`)
- Threshold: **`0.99`** (empirically optimized on train data — see calibration below)
- Rule: Max 1 match from S2 + Max 1 match from S3 per S1 entity

### Threshold Calibration (Offline, on 100K train entities)

```
Threshold 0.65 → Macro F0.5: 0.4207
Threshold 0.85 → Macro F0.5: 0.4265  ← Original baseline
Threshold 0.90 → Macro F0.5: 0.4283
Threshold 0.95 → Macro F0.5: 0.4315
Threshold 0.98 → Macro F0.5: 0.4353
Threshold 0.99 → Macro F0.5: 0.4370  ← Current (optimal)
```

**Key Insight:** F0.5 penalizes False Positives 2× more than False Negatives, so a very high threshold dramatically improves Precision without proportionally hurting Recall.

---

## Stage 2: Multilingual Indic Matcher (`src/multilingual_matcher.py`)

### Problem
~46.8% of the test dataset consists of Indian entities. Ground truth contains heavy **cross-script transliteration** — business names written in Tamil/Hindi/Marathi script matching English-language records in S2/S3. The XGBoost stage completely misses these.

### Solution: Sentence-Transformers + Address Inverted Index

**Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Supports 50+ languages natively
- Run in FP16 for GPU memory efficiency
- Encodes 867K Indic candidates in ~107 seconds on NVIDIA GB10

**Two-Tier Decision Rule (Calibrated via `src/validate_multilingual_rules.py`):**

| Tier | Rule | Precision (Offline) |
|---|---|---|
| ~~Tier 1 (Disabled)~~ | ~~Cosine ≥ 0.95 (pure)~~ | ~~23.9% (too many FP)~~ |
| **Tier 2 (Active)** | **Cosine ≥ 0.50 AND Address Token Overlap ≥ 7** | **81.3%** |

**Why Address Overlap is Essential:**
- Indian business names like "Shree Ram Enterprises" are extremely common templates.
- Pure cosine matching causes massive false positives across different states.
- Requiring ≥ 7 shared address tokens (locality, city, street name) as a grounding constraint eliminates ~95% of false positives.

**Output:** `output/multilingual_matches.tsv` — 163,690 high-precision Indic match pairs.

---

## Stage 3: Ensemble Fusion (`build_sub5_ensemble.py`)

**Priority Logic:**
1. **Primary:** Multilingual Matcher match (overrides XGBoost for Indic entities — higher precision for cross-script)
2. **Secondary:** XGBoost 0.99 match (for all non-Indic entities where Multilingual Matcher has no match)

**Constraints Enforced:**
- Exactly 1,732,544 output rows (one per S1 entity)
- Max 1 match from S2 + Max 1 match from S3 per entity
- Tab-delimited TSV with raw string formatting (no `csv.writer`) for platform compatibility

**Output:** `sub5_output/matching_results.tsv` — Submission file ✅

---

## Teammate Contributions

| Teammate | Branch | Contribution |
|---|---|---|
| **Mukul** | `main` | XGBoost re-ranker pipeline, candidate blocking, DGX execution |
| **Prateek** | `prateek` | Multilingual Indic Matcher, Phase 2 threshold calibration |
| **Harsh** | `harsh` | Advanced candidate blocking with address anchoring (`src/candidate_blocking.py`) |

---

## Files

| File | Purpose |
|---|---|
| `run_dgx_pipeline.py` | Stage 1: XGBoost matching (threshold = 0.99) |
| `src/multilingual_matcher.py` | Stage 2: GPU Indic Matcher |
| `build_sub5_ensemble.py` | Stage 3: Ensemble fusion |
| `validate_dgx_tsv.py` | Submission file format validator |
| `evaluate_local.py` | Offline threshold calibration on train data |
| `src/validate_multilingual_rules.py` | Offline rule calibration for Indic Matcher |
| `xgb_reranker.json` | Pre-trained XGBoost model weights |

---

## Submission History

| # | Score | Notes |
|---|---|---|
| Sub 1 | 0.415 | XGBoost baseline, threshold=0.85 |
| Sub 2 | 0.415 | (duplicate test) |
| Sub 3 | Failed | CSV writer format issue |
| Sub 4 | Failed | CSV writer format issue |
| Sub 5 | 0.413 | Multilingual applied with wrong priority |
| Sub 6 | 0.418 | ✅ Multilingual overrides XGBoost for Indic entities |
| **Sub 7** | **TBD** | XGBoost threshold raised 0.85→0.99 + Multilingual ensemble |
