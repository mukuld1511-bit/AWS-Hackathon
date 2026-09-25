# 🏆 Team Approach & DGX Roadmap: Business Entity Resolution (Amazon ML Challenge 2026)

## Executive Summary

Our solution is a **Multi-Stage Hybrid AI System** designed to solve high-cardinality, noisy entity resolution across 1.73 million test entities from 3 independent sources. The core pipeline blends:
1. GPU-accelerated heuristic & semantic candidate blocking.
2. Gradient Boosted Decision Trees (XGBoost) with calibrated high-precision thresholds.
3. Multilingual Cross-Script Transformer Models (`paraphrase-multilingual-MiniLM-L12-v2`) for Indian Indic records.
4. Precision-first post-processing and graph transitivity logic optimizing for the macro $F_{0.5}$ metric.

**Current Verified Leaderboard Score:** **`0.418`**  
**Immediate Milestone:** **`0.65+`**  
**Final Competition Target:** **`0.85+ / 0.90+`** (Top 1%)

---

## 🏛️ Current Production Architecture (How We Hit 0.418)

```
Test S1 (1,732,544 rows) + Test S2 (4.88M) + Test S3 (5.08M)
                        │
                        ▼ STAGE 1: Country Slicing (0% Cross-Country Leakage)
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
      [ US ]        [ India ]       [ France ]
 (38.3% data)     (46.8% data)    (15.0% data)
        │               │               │
        └───────────────┼───────────────┘
                        │
                        ▼ STAGE 2: Multi-Key Fast Candidate Blocking
     - Exact Cleaned Name Inverted Index
     - Postal Code / Pincode Matching
     - First-Word & 3-Char Prefix Lookups
     - Dense Semantic Fallback (Sentence-Transformers MiniLM)
                        │
                        ▼ (~15-20 candidates per entity)
STAGE 3: XGBoost Re-Ranking + Multilingual Indic Transformer
     - 7-D String, Token & Address Overlap Features
     - Strict Probability Threshold: 0.99 (Prunes False Positives)
     - Multilingual Transformer Bridge (Tamil/Hindi to English)
                        │
                        ▼ STAGE 4: Ensemble Fusion & Precision Clamping
     - Priority: Multilingual Indic Match > XGBoost 0.99 Match
     - Strict Clamping: Max 1 S2 match + Max 1 S3 match per S1
     - True Singletons retained as empty strings (Scores 1.0)
                        │
                        ▼
    Output TSV (Exact 1,732,544 rows, Single-Tab Delimited)
```

---

## 🚀 DGX COMPUTE ROADMAP: THE PATH TO 0.65 ➔ 0.85+

To climb from **0.418 to 0.85+**, we must address the remaining error buckets identified in our error analysis on DGX. Below is our phased roadmap:

```
[ Current: 0.418 ]
       │
       ▼ PHASE 1: Address & Pincode Normalization Engine (Target: 0.55 - 0.60)
       │ - Harsh's Address Blocking & Component Extraction (City, State, Pin)
       │ - French Address Normalization (Rue, Bd, Ave, Cedex)
       │
       ▼ PHASE 2: Hard-Negative GBDT Retraining (Target: 0.65 - 0.72)
       │ - Mine Hard Negatives (same pincode + similar name but different ID)
       │ - Train CatBoost / LightGBM on 1.5M true/hard-negative pairs on DGX
       │
       ▼ PHASE 3: Dense Embedding FAISS Dual-Encoder (Target: 0.75 - 0.82)
       │ - BAAI/bge-small-en-v1.5 + Multilingual MiniLM for all entities
       │ - Full Vector Retrieval across S2/S3 on DGX Multi-GPU
       │
       ▼ PHASE 4: Graph Transitivity & Mutual Consistency (Target: 0.88 - 0.92+)
       │ - Triangle Consistency: (S1 ↔ S2) and (S2 ↔ S3) ⇒ (S1 ↔ S3)
       │ - Final Leaderboard Lock
```

---

### 📍 PHASE 1: Advanced Address Normalization & Pincode Anchoring (Next Run)
- **Problem:** Currently, generic business names ("General Store", "Medical Agency", "Auto Works") get confused when addresses differ by just locality or plot numbers.
- **DGX Solution:**
  1. Parse street, municipal number, city, and 6-digit PIN / 5-digit US ZIP into structured attributes.
  2. Implement an **Address Hard Filter**: If pincodes/cities conflict within the same country, forcefully zero out candidate score regardless of name similarity.
  3. Support **French Address Structures**: Normalize `Rue`, `Boulevard`, `Avenue`, `BP`, and French postal codes (`75001` - `75020`).
- **Expected Score Impact:** `+0.10 to +0.15` (Pushes score into **0.55 – 0.60** range).

---

### 📍 PHASE 2: Hard-Negative Mining & GBDT Upgrade
- **Problem:** Our current `xgb_reranker.json` was trained on random negative samples. In real evaluation, false positives come from "hard negatives" (two different businesses on the same street with similar names).
- **DGX Solution:**
  1. Generate 1,000,000 Hard Negatives from training ground truth using Stage 1 candidate retrieval.
  2. Add Cross-Encoder features:
     - Soundex / Double Metaphone phonetic similarity.
     - Token Sort vs Token Set ratio asymmetry.
     - Pincode exact match boolean.
  3. Train a GPU-accelerated **LightGBM / CatBoost** model on DGX in < 15 minutes.
- **Expected Score Impact:** `+0.08 to +0.12` (Pushes score into **0.65 – 0.72** range).

---

### 📍 PHASE 3: Scale-Out Dense Vector Retrieval (FAISS GPU)
- **Problem:** Pure token/string blocking misses heavily paraphrased, legal abbreviation changes, and DBA ("doing business as") brand aliases.
- **DGX Solution:**
  1. Use DGX high-VRAM GPUs to encode all 4.88M S2 and 5.08M S3 entities using `BAAI/bge-small-en-v1.5` (Apache 2.0, <33M params).
  2. Build GPU FAISS indexes per country (US, India, France).
  3. Top-25 nearest neighbor retrieval in < 3 minutes across 1.73M S1 entities.
- **Expected Score Impact:** `+0.08 to +0.10` (Pushes score into **0.78 – 0.83** range).

---

### 📍 PHASE 4: Graph Transitivity & Mutual Consistency (Top 1% Lock)
- **Problem:** S1-S2-S3 form a tripartite graph of real-world companies. If S1 matches S2-A, and S2-A independently matches S3-B, then S1 MUST match S3-B.
- **DGX Solution:**
  1. Build a bipartite match graph across predicted (S1-S2) and (S1-S3) pairs.
  2. Apply Connected Components clustering with cycle consistency verification.
  3. Eliminate conflicting triangles where S2 and S3 belong to different clusters.
- **Expected Score Impact:** `+0.05 to +0.08` (Pushes score into **0.88 – 0.92+** winning tier).

---

## 👥 WORKFLOW & ROLE DIVISION

```
                      [ MUKUL (Lead / Main) ]
                Architecture, Main Branch Merge,
                DGX Job Orchestration & Verification
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
    [ HARSH ]               [ PRATEEK ]             [ AYUSH ]
Advanced Blocking &     Multilingual Indic     Documentation, Error
Address Normalization   Transformer Engine     Analysis & Feature Eng
   (Branch: harsh)       (Branch: prateek)       (Branch: ayush)
```

| Member | Focus Area | Deliverable for Main |
|---|---|---|
| **Mukul** | Architecture, DGX Pipeline, Ensemble Fusion, Unstop Leaderboard | `run_dgx_pipeline.py`, `build_sub5_ensemble.py`, `validate_dgx_tsv.py` |
| **Prateek** | Indic Transliteration, Multilingual MiniLM, Cross-script matching | `src/multilingual_matcher.py` (Clamped to 1 S2, 1 S3) |
| **Harsh** | Heuristic Multi-Key Inverted Index, Address/Pincode extraction | `src/candidate_blocking.py` |
| **Ayush** | Feature documentation, error distribution analysis, validation writeups | `Documentation_template.md`, metric reports |

---

## 📊 SUBMISSION TRACKER & PERFORMANCE MILESTONES

| Sub # | Timestamp | Model Architecture | Threshold / Strategy | Unstop LB Score | Status |
|:---:|:---:|---|---|:---:|:---:|
| **Sub 1** | 03:37 PM | XGBoost GBDT Re-ranker | Threshold = 0.85 | **0.415** | Evaluated ✅ |
| **Sub 2** | 04:55 PM | XGBoost GBDT Re-ranker | Threshold = 0.85 (re-test) | **0.413** | Evaluated ✅ |
| **Sub 3-4**| 05:45 PM | Multilingual Direct Attempt | Row Count Mismatch / CSV formatting | — | Failed ❌ |
| **Sub 6** | 06:40 PM | Sub 4 + Indic Multilingual Ensemble | Indic Override + Clamped 1-S2/1-S3 | **0.418** | Evaluated ✅ |
| **Sub 7** | **In Queue** | XGBoost 0.99 + Indic Matcher v2 | High-Precision Clamped Ensemble | **Target: 0.45+** | Pending ⏳ |
| **Sub 8** | **Phase 1** | Address Parsing + Pincode Filter + XGBoost | Hard Pincode Barrier | **Target: 0.55+** | Planned |
| **Sub 9** | **Phase 2** | Hard-Negative GBDT (CatBoost / LightGBM) | Mined Negative Pairs | **Target: 0.68+** | Planned |
| **Sub 10**| **Phase 3-4** | Full GPU FAISS Embeddings + Graph Transitivity | Full Ensemble Cluster | **Target: 0.85+** | Planned |
