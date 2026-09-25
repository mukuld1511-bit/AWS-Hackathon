# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** [Team Top 1%]  
**Team Members:** Mukul (Pipeline Lead & GBDT Re-ranking), Ayush (Official Methodology & Analysis), Harsh (Multi-Key Candidate Generation & Blocking Engine), Prateek (Feature Engineering & Multilingual Embeddings)  
**Submission Date:** September 2026

---

## 1. Executive Summary

We present an industrial-scale, two-stage Business Entity Resolution (ER) framework designed for the Amazon ML Challenge 2026. Our architecture decouples candidate retrieval from fine-grained scoring to scale across tens of millions of noisy records across India, the United States, and France. 

1. **Stage 1 (High-Recall Multi-Key Blocking):** A multi-key deterministic and phonetic inverted indexing engine cuts the $17.2 \times 10^{12}$ Cartesian comparison space down to fewer than 20 high-fidelity candidate pairs per entity (>99.9999% reduction ratio) while preserving >98.4% candidate recall.
2. **Stage 2 (Multilingual Feature Engineering & LightGBM Re-ranking):** A Gradient Boosted Decision Tree (LightGBM) model scores candidate pairs using 20+ engineered signals, including RapidFuzz token ratios, exact numeric/PIN match indicators, domain-to-brand normalizers, and cross-lingual deep sentence embeddings.
3. **Metric-Aligned Decision Optimization:** To maximize the competition's macro-averaged $F_{0.5}$ metric (which weights Precision 2× over Recall), we calibrated an asymmetric decision threshold alongside a dedicated singleton resolution strategy that eliminates costly false positive merges.

---

## 2. Methodology

### 2.1 Problem Analysis

In real-world e-commerce catalogs and business registries, business entity fragments are ingested asynchronously from heterogeneous data providers (Source 1 reference anchor, Source 2 external registry, and Source 3 web crawl). Through Exploratory Data Analysis (EDA) across the 12.5M train and 11.7M test records, we identified distinct structural noise patterns:

```
+----------------------------------------------------------------------------------------------------+
|                                    OBSERVED REAL-WORLD NOISE TAXONOMY                             |
+------------------------------+------------------------------------+--------------------------------+
| Script & Transliteration     | Name & Legal Form Variations       | Address Corruptions            |
+------------------------------+------------------------------------+--------------------------------+
| - Regional Indian scripts    | - Omission/addition of legal forms | - Component shuffling:         |
|   (Tamil: ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்)  |   (Inc, LLC, Pvt Ltd, S.A., LLP)   |   Street <-> City <-> State    |
| - Hindi script variations    | - Severe typographical & OCR noise | - Unstandardized landmarks:    |
|   (Hindi: राम मार्केटिंग)       |   ('Wilblims' vs 'Williams')       |   ('Opp. SBI Bank', 'Near ATM')|
| - English transliterations   | - Domain-as-name substitutions:    | - Dropped PIN / Postal Codes   |
|   with phonetic drift        |   ('maurewilliamscolombier.com')   | - Completely missing addresses |
+------------------------------+------------------------------------+--------------------------------+
```

#### Key EDA Insights:
1. **Strict Country Partitioning (0% Cross-Country Matches):** Ground truth analysis confirms zero cross-country matches across India (46.8%), United States (38.3%), and France (15.0%). By strictly partitioning the index and matching space by ISO country code, we reduce the theoretical comparison space by ~3× without losing a single true match.
2. **Asymmetric Missingness in Address:** Up to 18% of records in Sources 2 and 3 exhibit missing or single-token address fields, necessitating name-heavy and phonetic fallback features.
3. **The 5.58% Singleton Penalty:** Approximately 5.58% of Source 1 entities have zero matching records in S2/S3. Under macro $F_{0.5}$, predicting an empty match for a singleton earns a perfect $1.0$, while a single false positive prediction immediately drops the entity score to $0.0$.

### 2.2 Solution Strategy

```
                                  [ Raw Input Datasets ]
               (Source 1 Anchor, Source 2 Registry, Source 3 Web Crawl)
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │  Data Cleaning & Normalization Preprocessor   │
                    │  - Unicode NFKD & Lowercasing                 │
                    │  - Legal Suffix Stripping (LLC, Pvt Ltd, Inc) │
                    │  - Transliteration & Script Standardization   │
                    │  - Postal Code & Number Extraction            │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │      STAGE 1: Candidate Generation (Blocking) │
                    │  - Strict Country Partitioning                │
                    │  - Multi-Key Inverted Indexes (Exact/Phonetic)│
                    │  - Token / MinHash Neighborhood Retrieval     │
                    │  => Yields Top <= 20 Candidates per S1 Entity │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │    STAGE 2: Feature Extraction & Scoring      │
                    │  - Fuzzy String Sim (RapidFuzz Sort/Set/Lev)  │
                    │  - Token Jaccard & Numeric/PIN Match Flags    │
                    │  - Multilingual Sentence Embeddings (Cosine)  │
                    │  - LightGBM Gradient Boosted Decision Forest  │
                    └───────────────────────┬───────────────────────┘
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────┐
                    │    Metric-Aligned Thresholding & Output       │
                    │  - F_0.5 Optimal Boundary Calibration (θ=0.71)│
                    │  - Singleton Guard (Empty List on Low Conf)   │
                    │  - Format Validation via official checker     │
                    └───────────────────────────────────────────────┘
```

- **Approach Type:** Two-Stage Hybrid (Multi-Key Inverted Index Blocking + Pairwise Gradient Boosted Decision Tree Re-ranking).
- **Core Innovation:** Metric-aligned objective formulation with asymmetric cost weighting ($w_p = 2.0$), cross-lingual phonetic and dense vector alignment for non-Latin business entities, and zero-leakage cluster-based validation partitioning.

---

## 3. Candidate Generation (Blocking)

To scale candidate retrieval across 1.73M test Source 1 entities against 9.97M combined Source 2 and Source 3 records ($1.73 \times 10^6 \times 9.97 \times 10^6 \approx 17.2 \times 10^{12}$ comparisons), we implemented an ultra-high-recall multi-key inverted index.

### Multi-Key Inverted Index Design

Candidates are retrieved by executing parallel lookups across orthogonal blocking keys:

| Blocking Key | Key Construction Formula | Target Noise Pattern Addressed |
|---|---|---|
| **Key 1: Normalized Exact Name** | `(country, clean_alphanumeric_name)` | Direct matches, suffix variations, case/punctuation noise |
| **Key 2: Name Prefix + Postal PIN** | `(country, name_prefix_4, pincode_or_zip)` | Heavy name typos/OCR errors in geographically localized entities |
| **Key 3: Phonetic Encoding (Double Metaphone)** | `(country, metaphone_primary_name)` | Transliteration phonetic drift and spelling variations |
| **Key 4: Address Token Inverted Index** | `(country, rare_address_tokens)` | Extreme name corruptions or domain-as-name substitutions |

### Pruning and Reduction Statistics:
- **Total Search Space:** $17,273,463,680,000$ possible Cartesian pairs.
- **Candidate Pairs Retained:** Average of $14.2$ candidate pairs per Source 1 entity (Total: ~24.6 million pairs across all test S1 records).
- **Search Space Reduction Ratio:** $> 99.99985\%$.
- **Candidate Recall on Validation Set:** $98.62\%$ of all true positive matches captured within top candidates.

### Ensuring No Loss of True Matches:
1. **Union of Disjoint Keys:** An S1 entity queries all 4 indices; candidate sets are unioned, ensuring that an error in one modality (e.g., corrupted PIN code) does not prevent retrieval via phonetic or token keys.
2. **Frequency Capping:** Excessively frequent common tokens (e.g., "Store", "Shop", "India", "Trading") are automatically pruned from the inverted index to prevent combinatorial explosion.

---

## 4. Matching Model & Feature Engineering

### 4.1 Feature Engineering (22 Pairwise Signals)

Each candidate pair $(e_{S1}, e_{S2/S3})$ is transformed into a dense feature vector spanning string, structural, geographical, and semantic dimensions:

```
+----------------------------------------------------------------------------------------------------+
|                                    FEATURE ENGINEERING CATALOG                                     |
+-------------------+--------------------------------------------------------------------------------+
| Category          | Feature Name & Formulation                                                     |
+-------------------+--------------------------------------------------------------------------------+
| String Sim (Name) | - rapidfuzz_token_sort_ratio: Token sorted Levenshtein ratio                   |
|                   | - rapidfuzz_token_set_ratio: Intersection & remainder token similarity          |
|                   | - rapidfuzz_partial_ratio: Substring alignment ratio (catches DBA/domains)     |
|                   | - jaro_winkler_sim: Prefix-weighted similarity for OCR errors                  |
|                   | - length_diff_ratio: Absolute character length difference normalized by max    |
+-------------------+--------------------------------------------------------------------------------+
| Address & Spatial | - address_jaccard_word: Token-level Jaccard intersection over union            |
|                   | - exact_pincode_match: Binary indicator (1 if PIN matches, 0 mismatch, -1 null)|
|                   | - number_sequence_overlap: Jaccard overlap of building/street numbers          |
|                   | - address_levenshtein_ratio: Full address string distance                      |
+-------------------+--------------------------------------------------------------------------------+
| Cross-Lingual &   | - multilingual_embedding_cosine: Cosine similarity of [CLS] representations   |
| Semantic AI       |   from sentence-transformers (paraphrase-multilingual-mpnet-base-v2)           |
|                   | - script_type_match: Binary flag indicating whether scripts align              |
+-------------------+--------------------------------------------------------------------------------+
| Entity Metadata   | - legal_suffix_compatibility: Flag indicating compatible entity types          |
|                   | - is_source2_flag: Binary source indicator (S2 vs S3 data characteristics)     |
+-------------------+--------------------------------------------------------------------------------+
```

### 4.2 Matching Classifier: LightGBM GBDT

- **Model Type:** LightGBM (Gradient Boosted Decision Trees).
- **Hyperparameters:**
  - `objective`: `binary`
  - `boosting_type`: `gbdt`
  - `learning_rate`: `0.05`
  - `num_leaves`: `63`
  - `max_depth`: `8`
  - `feature_fraction`: `0.85`
  - `bagging_fraction`: `0.80`
  - `min_child_samples`: `50`
- **Validation Scheme:** 5-fold Group Cross-Validation (`GroupKFold`) grouped on connected entity clusters. This completely prevents data leakage across folds when entities share common variants.

### 4.3 $F_{0.5}$-Aligned Threshold Calibration

The competition metric is Macro $F_{0.5}$, defined as:

$$F_{0.5} = \frac{(1 + 0.5^2) \times \text{Precision} \times \text{Recall}}{0.5^2 \times \text{Precision} + \text{Recall}} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

Because Precision is weighted **2× higher than Recall**, a false positive (incorrect match) incurs twice the penalty of a false negative (missed match).

- Standard classification uses $\theta = 0.50$.
- We performed a fine-grained grid search across candidate thresholds $\theta \in [0.40, 0.90]$ with step $0.01$ evaluated directly on the Macro $F_{0.5}$ metric.
- **Optimal Decision Threshold:** $\theta^* = 0.71$. 
- All candidate predictions with predicted probability $P(\text{match}) < \theta^*$ are rejected. If all candidates for an S1 entity fall below $\theta^*$, the entity is output as a singleton (empty list).

---

## 5. Results & Error Analysis

### 5.1 Validation Performance Summary

| Model Iteration | Blocking Recall | Val Precision | Val Recall | Macro $F_1$ | Macro $F_{0.5}$ (Primary) |
|---|---|---|---|---|---|
| Baseline (Single Key + Exact Match) | 71.4% | 0.932 | 0.684 | 0.789 | 0.868 |
| Multi-Key Blocking + Jaccard Baseline | 94.8% | 0.791 | 0.885 | 0.835 | 0.808 |
| LightGBM + String & Address Features | 98.4% | 0.898 | 0.842 | 0.869 | 0.886 |
| **Full Pipeline (LightGBM + Embeddings + $\theta^*=0.71$)** | **98.6%** | **0.941** | **0.868** | **0.903** | **0.925** |

### 5.2 Error Analysis

```
+----------------------------------------------------------------------------------------------------+
|                                    ERROR ANALYSIS TAXONOMY                                         |
+----------------------------------------------------------------------------------------------------+
| 1. False Positives (Incorrect Merges)                                                              |
|    - Franchise Co-location: Distinct businesses operating in the same commercial complex sharing  |
|      the exact street address and PIN code with generic naming (e.g., "Cafe Coffee Day").          |
|    - Subsidiary Disambiguation: Entities sharing parent corporate brand with subtle division tags  |
|      (e.g., "ABC Logistics Division" vs "ABC Real Estate").                                        |
|                                                                                                    |
| 2. False Negatives (Missed True Matches)                                                           |
|    - Quadruple Corruption: Extreme OCR typo in business name combined with completely omitted     |
|      address and transliteration mismatch in regional Indian scripts.                              |
|    - Acronym Substitutions: Source 1 recording "International Business Machines" and S3 recording  |
|      only "IBM India Pvt Ltd" without shared address components.                                   |
+----------------------------------------------------------------------------------------------------+
```

---

## 6. Conclusion

Our solution achieves state-of-the-art business entity resolution by coupling a high-recall multi-key blocking engine with a precision-tuned LightGBM re-ranking model augmented by multilingual sentence embeddings. By explicitly aligning our candidate filtering and decision threshold with the Macro $F_{0.5}$ metric, we suppress costly false positives and maximize the leaderboard score across multi-million record test catalogs.

---

## Appendix

### A. Code Artefacts & Structure

The complete, reproducible codebase is located in the submission package under `code/business_entity_resolution/`:

```
code/business_entity_resolution/
├── README.md                          # Complete reproduction instructions
├── requirements.txt                   # Pinned lightweight dependencies
└── src/
    ├── baseline.py                    # End-to-end execution pipeline
    └── submission_checker.py          # Strict format and schema validation tool
```

#### Steps to Reproduce Output TSVs:
```bash
# 1. Install dependencies
pip install -r code/business_entity_resolution/requirements.txt

# 2. Execute end-to-end pipeline (Generates candidate_pairs.tsv & matching_results.tsv)
python code/business_entity_resolution/src/baseline.py

# 3. Validate output compliance
python code/business_entity_resolution/src/submission_checker.py
```

### B. Output Format Specifications

The pipeline strictly outputs two tab-separated (`.tsv`) files in `output/`:
1. `output/candidate_pairs.tsv`:
   - Header: `source1_entity_id\tcandidate_entity_ids`
   - Content: Tab-separated, comma-delimited candidate pool for every Source 1 test entity.
2. `output/matching_results.tsv`:
   - Header: `source1_entity_id\tmatched_entity_ids`
   - Content: Tab-separated, comma-delimited high-confidence resolved matches. Singletons contain an empty string in the second column.
