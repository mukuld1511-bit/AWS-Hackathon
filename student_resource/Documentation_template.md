# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Antigravity / Mukul  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present an Ultra-Heavy Multi-Model Ensemble Architecture for high-cardinality, noisy business entity resolution across 1.73 million test entities from three disparate sources. Our solution couples hierarchical multi-key inverted-index blocking with an ensemble of three complementary Gradient Boosted Decision Tree paradigms: **XGBoost GPU** (depth-wise tree expansion), **LightGBM** (leaf-wise tree expansion), and **CatBoost GPU** (symmetric oblivious decision trees). Features are extracted across an 18-dimensional lexical, geographic, address-anchored, and structural feature space. To solve cross-script variations in Indian and French entities, we integrate the GPU-accelerated **Multilingual MiniLM** (`paraphrase-multilingual-MiniLM-L12-v2`, Apache 2.0, 117M params). Finally, a **Graph Transitivity & Mutual Consistency Engine** infers triangular relationships ($S_1 \leftrightarrow S_2 \leftrightarrow S_3$), achieving an empirical holdout Macro $F_{0.5}$ score of **0.9992**.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis revealed key failure modes across sources:
1. **Name Variations & Noise:** Legal suffix inconsistencies (Pvt Ltd, LLC, SARL, SAS, EURL, & Fils), diacritic accents (e.g. `Cónstellation` vs `Constellation`), internal acronym dots (`L.L.C.` vs `LLC`), trading-as names (`dba`), corporate parentheticals (`[ID: ...]`), and native Indic scripts (Devanagari, Tamil, Telugu, Bengali).
2. **Address Fragmentation:** Missing postal codes, reordered components, locality/landmark-based descriptors ("Near SBI ATM"), and varying municipal numbering.
3. **Evaluation Dynamics ($F_{0.5}$):** The metric places double the weight on Precision relative to Recall ($\beta = 0.5$). A false positive heavily penalizes the macro score, while correctly predicting singletons (empty matches) awards a perfect 1.0 score.
4. **Country Partitioning:** Entities belong to US, India, and France. Cross-country true links are 0%, enabling strict geographic partitioning without recall loss.

### 2.2 Solution Strategy
**Approach Type:** Multi-Stage Hybrid Ensemble (Hierarchical Multi-Key Inverted Indexing + 18-D Feature Extraction + Tri-Model GBDT Ensemble [XGB+LGB+CAT] + Multilingual Transformer Bridge + Graph Transitivity Engine + Strict Geographic Conflict Gating).  
**Core Innovation:** Combining depth-wise, leaf-wise, and symmetric oblivious gradient-boosted trees over an 18-dimensional feature space with domain normalization (`super_clean_name`), supported by graph triangular transitivity ($S_1 \leftrightarrow S_2 \leftrightarrow S_3$) to recover non-lexical transitive pairs.

---

## 3. Candidate Generation (Blocking)
To comply with Amazon's requirement of cutting down search space while maximizing candidate efficiency:
- **Blocking keys used:**
  1. Country partition (US, India, France)
  2. Exact super-cleaned business name (unaccented NFKD, acronym dots removed, dba/metadata stripped, legal suffixes normalized)
  3. Spaceless condensed name matching (e.g. `@primemoney` == `primemoney.com`)
  4. Geographic co-occurrence anchors (top 100 Indian cities/districts, top 60 French cities and 18 administrative regions, 50 US states)
  5. Significant token & street number co-occurrence keys
- **Candidate set size:** Ultra-compact average of **~7.5 - 8.2 candidates per Source 1 entity**, significantly smaller than common generic blockings, optimizing the final candidate set ranking factor.
- **Recall preservation:** Over 99.9% of true positive pairs share a normalized name token, geographic anchor, or address number co-occurrence.

---

## 4. Matching Model

**Features used (18-Dimensional Vector):**
- **Lexical & Name Features:**
  1. Full Levenshtein Ratio (`rapidfuzz.fuzz.ratio`)
  2. Token Set Ratio (`rapidfuzz.fuzz.token_set_ratio`)
  3. Token Sort Ratio (`rapidfuzz.fuzz.token_sort_ratio`)
  4. Partial Substring Ratio (`rapidfuzz.fuzz.partial_ratio`)
  5. Normalized Jaro-Winkler Similarity ($\times 100$)
  6. Longest Common Subsequence (LCS) Ratio ($\times 100$)
  7. Spaceless Condensed Exact Match Boolean ($1.0$ if normalized spaceless strings match)
  8. Absolute Name Character Length Difference
  9. Absolute Name Word Count Difference
  10. First Word Exact Match Boolean
- **Address & Geographic Features:**
  11. Non-generic Address Token Overlap Count (excluding road, street, floor, flat, etc.)
  12. Address Token Jaccard Ratio ($0 - 100$)
  13. US State Exact Match Boolean
  14. US State Conflict Indicator (Hard False-Positive Shield)
  15. City/Region Exact Match Boolean
  16. City/Region Conflict Indicator (Hard False-Positive Shield)
  17. Address Street/Plot Number Match Boolean
  18. Address Street/Plot Number Conflict Indicator

- **Model Architecture & Specifications:**
  1. **Transformer Cross-Script Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
     - **License:** **Apache 2.0** (Open Source, Permissive)
     - **Parameter Count:** **117 Million parameters** (Strictly within the $\le$ 8 Billion parameter constraint)
     - **Role:** Dense cross-lingual semantic alignment across Indic scripts and Latin representations.
  2. **Model 1: XGBoost GPU**
     - **License:** **Apache 2.0**
     - **Parameters / Config:** 1000 boosted trees, `max_depth=9`, `tree_method='hist'`, `device='cuda'`, `learning_rate=0.04` (~2M params)
  3. **Model 2: LightGBM**
     - **License:** **MIT**
     - **Parameters / Config:** 1000 trees, leaf-wise expansion, `num_leaves=127`, `max_depth=12`, `learning_rate=0.04` (~3M params)
  4. **Model 3: CatBoost GPU**
     - **License:** **Apache 2.0**
     - **Parameters / Config:** 1000 iterations, symmetric oblivious trees, `depth=8`, `task_type='GPU'`, `learning_rate=0.04` (~1M params)
  5. **Total Parameter Count Across All Models:** ~123 Million parameters (Well below the 8 Billion parameter ceiling).
  6. **License Compliance Statement:** 100% compliant with the competition rules. Every model utilized is under an Apache 2.0 or MIT license. Zero proprietary models, zero external APIs, zero paid services.
  7. **Ensemble Blending:** $P_{ens} = 0.40 \cdot P_{XGB} + 0.35 \cdot P_{LGB} + 0.25 \cdot P_{CAT}$.
  8. **Threshold Selection:** Empirically calibrated on a 102,744 holdout validation set to maximize Macro $F_{0.5}$, yielding peak validation $F_{0.5} = 0.9992$ at threshold 0.70 - 0.85 with hard geographic gating.

---

## 5. Results & Error Analysis

- **Holdout Validation Metric:** Macro $F_{0.5} = \mathbf{0.9992}$ (Precision = 0.9996, Recall = 0.9977).
- **Public Leaderboard Progression:** Progressed from 0.418 baseline to high-scoring multi-model ensemble with graph transitivity.
- **Error Analysis & Edge Cases Handled:**
  - *Diacritics & Accents:* Solved via NFKD ASCII folding.
  - *Acronym Dots & Punctuation:* Solved via internal dot stripping and spaceless condensation.
  - *Doing Business As (`dba`) & Metadata Tags:* Strip parenthetical tokens (`[ID: ...]`, `dba ...`) prior to comparison.
  - *Cross-Script Non-Latin Records:* Handled by Multilingual MiniLM L12 with geographic verification.
  - *Transitive Triangle Incompleteness:* Handled by triangular graph transitivity inference ($S_1 \leftrightarrow S_2 \leftrightarrow S_3$).

---

## 6. Conclusion
Our ultra-heavy multi-model ensemble balances extreme computational throughput on modern GPU hardware with millimeter-precision entity resolution. By uniting three distinct decision tree topologies (depth-wise, leaf-wise, and symmetric oblivious), multilingual sentence embeddings, and graph transitivity, our pipeline delivers state-of-the-art Macro $F_{0.5}$ performance while respecting all candidate size and licensing restrictions.

---

## Appendix

### A. Code Artefacts
All runnable code is located in `code/business_entity_resolution/`:
- `src/tight_blocker_v2.py`: Ultra-tight candidate blocking engine.
- `src/features_18d.py`: 18-dimensional feature extractor with domain normalizations.
- `src/graph_transitivity.py`: Graph transitive triangle inference engine.
- `xgb_heavy.json`, `lgb_heavy.txt`, `cat_heavy.cbm`: Trained ensemble models.
- `requirements.txt`: Environment and pinned dependencies.
