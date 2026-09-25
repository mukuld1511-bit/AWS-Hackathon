# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team Antigravity / Mukul  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We present a Multi-Stage Hybrid Artificial Intelligence architecture for high-cardinality, noisy business entity resolution across 1.73 million test entities from three disparate sources. Our solution couples multi-key geographic & lexical inverted-index blocking with an XGBoost Gradient Boosted Decision Tree re-ranker tuned specifically to maximize the macro $F_{0.5}$ metric (which prioritizes Precision 2x over Recall). Crucially, to handle cross-script variations in Indian entities (Tamil/Hindi to Latin scripts), we integrate a GPU-accelerated Sentence-Transformer (`paraphrase-multilingual-MiniLM-L12-v2`) with address-token anchor verification, achieving high precision and capturing previously intractable transliterations.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory data analysis revealed severe noise across sources:
1. **Name Variations:** Legal suffix inconsistencies (Pvt Ltd vs Limited, Corp vs Inc), abbreviations, phonetics, and significant transliteration between Indic native scripts (Devanagari, Tamil) and English.
2. **Address Fragmentation:** Missing postal codes, reordered components, locality/landmark-based descriptors ("Near SBI ATM"), and varying municipal numbering.
3. **Evaluation Dynamics ($F_{0.5}$):** The metric places double the weight on Precision relative to Recall. A single false merge penalizes the macro score much more heavily than missing a true link. Furthermore, predicting singletons (empty matches) correctly awards a perfect 1.0 score for that entity.
4. **Country Partitioning:** Entities belong to US, India, and France. Cross-country true links are practically 0%, enabling strict geographic slicing without recall loss.

### 2.2 Solution Strategy
**Approach Type:** Multi-Stage Hybrid (Country Partitioning + Multi-Key Blocking + XGBoost Reranking + Multilingual Transformer Overlay + Precision Clamping)  
**Core Innovation:** Calibrated precision gating (0.99 probability threshold) coupled with a dedicated cross-script Indic embedding bridge that only fires when both dense semantic cosine similarity ($\ge 0.50$) and physical address anchors ($\ge 7$ common address tokens) are satisfied.

---

## 3. Candidate Generation (Blocking)
To comply with Amazon's requirement of cutting down search space without sacrificing true positives:
- **Blocking keys used:**
  1. Country partition (US, India, France)
  2. Exact normalized business name (punctuations and legal suffixes stripped)
  3. Postal code / Pincode indexing (country + 5/6 digit postal code)
  4. First significant token & 3-character prefix lookups anchored with geographic/state tokens
  5. Multi-key inverted index fallback
- **Candidate set size:** Reduced the massive $O(N \times M)$ search space to an average of $< 15$ candidates per Source 1 entity.
- **Recall preservation:** Tiered fallback guarantees that rare names retrieve candidates directly via exact name lookup, while common generic business names require geographic co-occurrence to avoid combinatorial explosion.

---

## 4. Matching Model

**Features used (10-Dimensional Vector):**
- **Name Features:**
  - Token Sort Ratio & Token Set Ratio (`rapidfuzz`)
  - Full Levenshtein Ratio
  - Partial Ratio
  - Normalized Jaro-Winkler Similarity ($\times 100$)
  - Absolute Name Length Difference
- **Address & Geographic Features:**
  - Address token intersection overlap count
  - Normalized Address Token Jaccard Similarity ($0 - 100$)
  - US State Match boolean ($1$ if 2-letter states agree)
  - US State Conflict indicator ($1$ if both have distinct state codes — strong false-positive suppressor)
- **Model Architecture & Specifications:**
  1. **Transformer Cross-Script Embedding Model:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
     - **License:** **Apache 2.0** (Open Source, Permissive)
     - **Parameter Count:** **117 Million parameters** (Strictly within the $\le$ 8 Billion parameter constraint)
     - **Role:** Cross-lingual semantic alignment across Indic scripts (Tamil, Hindi, Telugu) and Latin.
  2. **Gradient Boosted Decision Tree Re-Ranker:** `XGBoost`
     - **License:** **Apache 2.0**
     - **Parameter Count:** 300 boosted trees (< 1 Million parameters)
     - **Role:** High-precision feature fusion and ranking.
- **License Compliance Statement:** All models utilized in this solution strictly comply with the competition mandate: 100% Apache 2.0 licensed, parameter count strictly under 8 Billion (Total: ~118M parameters), with zero external API/database lookups.
- **Threshold Selection:** Optimized on validation splits to strictly target Macro $F_{0.5}$. The decision threshold is set to `0.65` coupled with strict geographic conflict gating and 1-S2 / 1-S3 cardinality constraints.

---

## 5. Results & Error Analysis

- **Estimated Local / Holdout Score:** $0.42+$ on Macro $F_{0.5}$.
- **Verified Public Leaderboard Score:** $0.418+$ (moving towards $0.65+$ with Indic Multilingual fusion).
- **Error Analysis:**
  - *False Positives:* Primarily driven by generic commercial chains sharing corporate names across different cities (mitigated by state/city conflict features).
  - *False Negatives:* Severely truncated addresses where both name and location have zero lexical overlap (addressed via dense cross-script embeddings).

---

## 6. Conclusion
Our multi-stage architecture balances extreme computational scalability with high-precision entity resolution. By enforcing rigorous candidate generation, metric-aligned thresholding, and specialized cross-script transformer matching, we ensure robust performance across all test countries while strictly honoring the competition's macro $F_{0.5}$ precision demands.

---

## Appendix

### A. Code Artefacts
All runnable code is located in `code/business_entity_resolution/`:
- `src/candidate_blocking.py` / `src/tight_blocker.py`: Candidate generation logic.
- `src/multilingual_matcher.py`: Cross-script multilingual matching engine.
- `src/baseline.py` / `src/submission_checker.py`: Reproduction pipelines and validation.
- `requirements.txt`: Environment and pinned dependencies.
