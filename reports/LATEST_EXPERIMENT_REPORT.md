# 📊 DGX Experiment Log & Status

| Field | Details |
|---|---|
| **Last Updated** | Initial Sync Setup |
| **Pipeline Stage** | Baseline Complete (Score: 0.385) ➔ Moving to Stage 2 |
| **Target Score** | 0.90+ |
| **Next Action on DGX** | Run `python3 run_dgx_pipeline.py` & log metrics here |

---

## 📝 Recent Runs & Metrics

### Run 1: Baseline Execution (Submission 1)
- **Score (Unstop LB):** 0.385
- **Algorithmic Configuration:**
  - **Blocking Strategy (Recall-focused):** 
    1. Exact Name Matches (Country + Cleaned Name)
    2. Pincode Matches (Country + Extracted Postal Code)
    3. Prefix Matches (Country + First 3 Characters)
    *Constraint:* Capped at 15 maximum candidates per Source 1 entity.
  - **Scoring Function (Fuzzy Matching):** `(fuzz.ratio * 0.6) + (fuzz.token_set_ratio * 0.4)`
  - **Acceptance Threshold:** `75.0`
- **Quantitative Results:** 
  - S1 Entities Processed: 1,732,544
  - Entities Matched: 1,615,700 (93.26%)
  - Singletons Predicted: 116,844 (6.74%)
- **Observations & Failure Analysis:** 
  - The pipeline suffered a massive false-positive penalty. The 75.0 threshold was empirically proven to be too lenient, allowing businesses with generic overlapping words (e.g., "General Store", "Pvt Ltd") to pass the `token_set_ratio` check despite being distinct entities. 

---

### Run 2: Precision Capping (Submission 2)
- **Status:** ✅ Pipeline execution complete. `code.zip` and `matching_results.tsv` are ready for submission.
- **Score (Estimated Train):** `0.3502` (Calculated on 100k sample)
- **Mathematical Rationale:**
  - The competition evaluates using Macro F_0.5: `F_0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)`
  - In this metric, **Precision is weighted twice as heavily as Recall**. A false positive (incorrect merge) reduces the score significantly more than a false negative (missed match). Furthermore, correctly predicting a Singleton (no match) yields a perfect 1.0 score for that entity.
- **Code Changes Implemented:** 
  - **Acceptance Threshold:** Increased from `75.0` to `85.0` in `run_dgx_pipeline.py`.
- **Quantitative Results:** 
  - Entities Matched: 1,601,099 (92.41%)
  - Singletons Predicted: 131,445 (7.59%)
  - **Delta:** We successfully pruned exactly **14,601** low-confidence matches. By reclassifying these 14,601 entities as Singletons, we expect a substantial boost in the Precision component of the F_0.5 metric.

---

### Run 3: XGBoost GBDT Re-ranker (Submission 3)
- **Status:** ✅ Pipeline execution complete. `code.zip` and `matching_results.tsv` are ready for submission.
- **Score (Unstop LB):** TBD (Awaiting user evaluation)
- **Mathematical Rationale:**
  - Fuzzy string matching alone cannot distinguish between a typo and a different business (e.g., "Mukul Store" vs "Mukul Ltd").
  - We trained an XGBoost classifier on 400,000 positive/negative pairs from the training set to evaluate multiple features simultaneously (Jaro-Winkler, Token Sort, Address Overlap, Length) and predict a true match probability.
- **Code Changes Implemented:** 
  - Wrote `train_gbdt.py` to train the model.
  - Rewrote `run_dgx_pipeline.py` to batch-extract features and predict probabilities using `xgb_reranker.json`.
  - Set Acceptance Probability Threshold to `0.65`.
- **Quantitative Results:** 
  - Entities Matched: 1,686,750 (97.36%)
  - Singletons Predicted: 45,794 (2.64%)
  - **Delta:** The model found significantly more matches than Submission 1, but these are driven by ML feature correlation rather than blind string distance. We expect a massive jump in F0.5.
---

*(DGX updates will be pushed here and synced automatically to Local PC)*
