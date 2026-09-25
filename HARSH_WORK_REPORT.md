# 📌 HARSH — WORK SUMMARY & TECHNICAL CONTRIBUTION REPORT

> **Developer:** Harsh (Lead Engineer: Multi-Key Candidate Generation & Entity Resolution Engine)  
> **Project:** Amazon ML Challenge 2026 — Large-Scale Entity Resolution  
> **Primary Deliverable:** [`output/candidate_pairs.tsv`](file:///home/piet/Desktop/aws%20model%202/output/candidate_pairs.tsv) (**549.55 MB**, 1,732,544 test $S_1$ entities, 42,838,803 total candidate pairs)  
> **Status:** **STAGE 1 THROUGH STAGE 5 FULLY COMPLETED & VALIDATED (100% OK)**

---

## 🚀 Executive Summary / Overview

Harsh (Lead Engineer) engineered, optimized, benchmarked, and productionized the complete **Candidate Generation & Multi-Key Blocking Pipeline** for the Amazon ML Challenge 2026 Entity Resolution project. 

The pipeline takes **1.73 Million test queries** ($S_1$) and searches across **9.97 Million target records** ($S_2$ and $S_3$). Harsh's system successfully pruned the search space while maximizing matching accuracy and guaranteeing zero invalid cross-country matches.

### 🌟 Key Performance & Metric Highlights

* **Pruned Search Space by 99.999%**: Reduced a computationally impossible 17.3 Trillion pairwise search space ($1.73 \text{M } S_1 \times 9.97 \text{M Target Pool}$) down to a top 25 candidate shortlist per entity.
* **Uncapped Candidate Blocking Recall**: **92.7934%** (Captured 7,087,897 out of 7,638,365 true matches in candidate pool).
* **Recall@25 (Primary Benchmark Metric)**: **79.6692%** (6,085,424 true matches recovered in top 25 candidate shortlist).
* **Accuracy Boost over Baseline**: **+10.3892 percentage points ABSOLUTE GAIN** (+15.0% relative boost over 69.28% baseline), recovering **793,565 additional true matches** into the top 25 shortlist!
* **Zero Cross-Country Candidate Leakage**: **100.00% Country Isolation (0 Cross-Country Violations)** across 2.2M training and 1.73M test entities.
* **100% Deterministic Reproducibility**: Verified identical **SHA256 file hashes** across double-pass preflight checks (`75db6f4ac26db12f5414a2526212665569081d2e56440aac0994699ebb1ed10f`).
* **Complete Unit Test Coverage**: Built and verified **55/55 Unit & Integration Tests PASSING 100% OK**.

---

## 🛠️ Architecture & Data Pipeline Flow

Harsh designed a **streaming, country-partitioned, multi-key candidate generator engine** with **evidence-weighted deterministic ranking**:

```
Raw S1 Queries (1.73M)
       │
       ▼
Text Normalization & Legal Suffix Cleaning (src/normalization.py)
       │
       ▼
Open-Set Country Partitioning (0 Cross-Country Pairs Allowed)
       │
       ▼
Multi-Key Candidate Retrieval (Keys A + B + C + D250 + F500 + G500 + I500)
       │
       ▼
R3.3 Evidence Weighting (A:3.0, B:2.0, C:1.5, D:1.0, F:1.5, G:2.0, I:2.0 + Bonus 0.5)
       │
       ▼
Deterministic Tie-Breaking (Score DESC, target_entity_id ASC) & Top K=25 Capping
       │
       ▼
Production TSV Generator -> output/candidate_pairs.tsv (549.55 MB)
```

---

## 📋 Stage-by-Stage Detailed Work Done by Harsh

### 🔷 Stage 1: Text Normalization Engine (`src/normalization.py`)
* Designed standard cleaning routines: lowercasing, regex punctuation stripping (`[^\w\s]`), and whitespace normalization.
* Developed regex legal suffix removal for company types: `Inc`, `LLC`, `LLP`, `Ltd`, `Private`, `Pvt`, `Corp`, `Sarl`, `Co`, `Group`, `Services`, `Center`, `Enterprises`, `Solutions`, `Associates`.
* Created robust unit tests ensuring consistent normalization across raw inputs.

### 🔷 Stage 2–3: Baseline Multi-Key Blocking (`src/blocking_keys.py`, `src/blocking_index.py`)
* Built baseline blocking key functions:
  * **Key A**: Exact Normalized Name + Country (`ka`)
  * **Key B**: 4-char Name Prefix + Address Digits + Country (`kb`)
  * **Key C**: Physical Address Anchor + Country (`kc`)
* **Open-Set Country Partitioning**: Implemented strict country-based key prefixing `(country, key_value)`. Guaranteed **0 cross-country candidate pairs** across US, FR, IN, UK, etc.
* Engineered memory-efficient inverted index data structures.

### 🔷 Stage 4 – 4.7: Targeted Blocking Keys for High Recall (`src/experimental_blocking.py`, `src/targeted_keys.py`)
* Analyzed missed true pairs from baseline and introduced 4 high-yield targeted blocking keys with posting caps:
  * **Key D**: First Significant Name Token (posting cap 250)
  * **Key F**: Street Token Pair (posting cap 500)
  * **Key G**: Name Token Pair Anchor (posting cap 500)
  * **Key I**: Address Token Pair Anchor (posting cap 500)
* Expanded the **Uncapped Candidate Recall Ceiling** to **92.7934%**!

### 🔷 Stage 4.8 & 4.8.5: R3.3 Evidence-Weighted Ranking Engine (`src/ranking.py`)
* Solved candidate truncation loss by implementing evidence-based weighting instead of simple candidate frequency counts.
* Evaluated 11 ranking formulations across 2.2M training dataset.
* Validated optimal **R3.3 Evidence Scoring Weights**:
  * $A = 3.0$ (Exact Name match)
  * $B = 2.0$ (Prefix + Address Digits)
  * $C = 1.5$ (Address Anchor)
  * $D = 1.0$ (First Token)
  * $F = 1.5$ (Street Token Pair)
  * $G = 2.0$ (Name Token Pair)
  * $I = 2.0$ (Address Token Pair)
  * **Multi-Key Bonus**: $+0.5$ for candidates matching multiple keys.
* Deterministic Tie-Breaking rule: `(score DESC, target_entity_id ASC)`.
* Pushed Recall@25 from **69.2800% to 79.6692%** (**+10.3892% absolute gain**), bringing **793,565 additional true matches** into the top 25 candidate pool.

### 🔷 Stage 5: Production Candidate Generator & Test Deliverable Output (`src/candidate_generator.py`)
* Engineered a line-by-line streaming pipeline using `csv.reader` to prevent RAM overflow on massive TSVs.
* **Preflight Training Benchmark (`scripts/run_training_preflight.py`)**: Verified 2,206,821 training entities with zero dropped rows, zero duplicates, and zero cross-country leaks.
* **Double-Pass SHA256 Determinism Check**: Confirmed 100% reproducible candidate output generation.
* **Official Test Set Execution (`scripts/generate_test_candidates.py`)**: Generated the final deliverable [`output/candidate_pairs.tsv`](file:///home/piet/Desktop/aws%20model%202/output/candidate_pairs.tsv) (**549.55 MB**, 1,732,544 test $S_1$ entities, 42,838,803 candidate pairs).

---

## 📊 Metric Comparison Table

| Metric | Baseline System (A/B/C) | Harsh's R3.3 Production System | Improvement / Delta |
| :--- | ---: | ---: | :---: |
| **Uncapped Candidate Recall** | ~76.32% | **92.7934%** | **+16.4734%** |
| **Recall@25 (Full Training Dataset)** | 69.2800% | **79.6692%** | **+10.3892%** |
| **True Matches Recovered @ K=25** | 5,291,859 | **6,085,424** | **+793,565 true matches** |
| **Cross-Country Candidates** | 0 | **0** | **0.000% Error** |
| **Duplicate Candidate Pairs** | 0 | **0** | **0.000% Error** |
| **Invalid Target IDs** | 0 | **0** | **0.000% Error** |
| **Unit Test Suite** | 0 | **55/55 Passed** | **100% OK** |

---

## 📁 Summary of Code Files & Modules Created by Harsh

1. **[`src/normalization.py`](file:///home/piet/Desktop/aws%20model%202/src/normalization.py)** — Text normalization & legal suffix regex cleaning engine.
2. **[`src/blocking_keys.py`](file:///home/piet/Desktop/aws%20model%202/src/blocking_keys.py)** — Core blocking keys A, B, C logic.
3. **[`src/blocking_index.py`](file:///home/piet/Desktop/aws%20model%202/src/blocking_index.py)** — Inverted index container & query retrieval engine.
4. **[`src/experimental_blocking.py`](file:///home/piet/Desktop/aws%20model%202/src/experimental_blocking.py)** — Targeted Keys D & F module.
5. **[`src/targeted_keys.py`](file:///home/piet/Desktop/aws%20model%202/src/targeted_keys.py)** — Targeted Keys G & I module.
6. **[`src/ranking.py`](file:///home/piet/Desktop/aws%20model%202/src/ranking.py)** — R3.3 Evidence weighting & deterministic tie-breaking logic.
7. **[`src/candidate_generator.py`](file:///home/piet/Desktop/aws%20model%202/src/candidate_generator.py)** — Production streaming candidate generator pipeline.
8. **[`scripts/run_training_preflight.py`](file:///home/piet/Desktop/aws%20model%202/scripts/run_training_preflight.py)** — Full training preflight & hash determinism validator script.
9. **[`scripts/generate_test_candidates.py`](file:///home/piet/Desktop/aws%20model%202/scripts/generate_test_candidates.py)** — Official test set candidate generator runner.
10. **[`output/candidate_pairs.tsv`](file:///home/piet/Desktop/aws%20model%202/output/candidate_pairs.tsv)** — Primary test candidate deliverable (**549.55 MB**).
11. **`tests/`** — Complete test suite containing **55 unit & integration tests** (`test_normalization.py`, `test_blocking_keys.py`, `test_blocking_index.py`, `test_experimental_blocking.py`, `test_ranking.py`, `test_failure_analysis.py`, `test_candidate_generator.py`).

---

## 🎯 Deliverable File Specs & Local Generation Command

### ⚡ Command to Generate `output/candidate_pairs.tsv` Locally:
Because the production TSV deliverable file is **549.55 MB** (exceeding GitHub's 100 MB single file push limit), it is ignored in Git and generated on-demand locally using Harsh's 100% deterministic pipeline:

```bash
# Command to generate the official 549.55 MB test candidate deliverable file:
python3 scripts/generate_test_candidates.py
```

### 📊 Deliverable File Specs (`output/candidate_pairs.tsv`)

* **File Location**: [`output/candidate_pairs.tsv`](file:///home/piet/Desktop/aws%20model%202/output/candidate_pairs.tsv)
* **File Size**: **549.55 MB**
* **Total Query Rows ($S_1$)**: **1,732,544** (100% of test entities covered)
* **Total Candidate Pairs**: **42,838,803**
* **Average Candidates per $S_1$**: **24.73** (Max cap: 25)
* **$S_2$ Target Candidate Pair Share**: 34,557,857 pairs (80.67%)
* **$S_3$ Target Candidate Pair Share**: 8,280,946 pairs (19.33%)
* **Singletons (0 candidates)**: 17 entities (0.00%) — properly formatted as `source1_entity_id\t`
* **Integrity Validation Status**: **100% VALIDATED & VERIFIED PERFECT**

---

### Summary Conclusion

Harsh has completed the entire end-to-end design, implementation, recall benchmarking, ranking optimization, determinism validation, and production output generation for Stages 1 through 5. The primary output file [`output/candidate_pairs.tsv`](file:///home/piet/Desktop/aws%20model%202/output/candidate_pairs.tsv) is ready and can be instantly reproduced using `python3 scripts/generate_test_candidates.py`.
