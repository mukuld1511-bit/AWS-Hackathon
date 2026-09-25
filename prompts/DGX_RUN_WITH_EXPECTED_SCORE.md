# 🚀 DGX IMMEDIATE EXECUTION & OFFLINE F0.5 BENCHMARK RUNBOOK

> **Execution Machine:** NVIDIA DGX  
> **Mission:** 
> 1. Compute and print the **Exact Expected Macro $F_{0.5}$ Score** on the official 20,000 entity training holdout.
> 2. Run the full **Heavy 18-D Tri-Model GBDT Ensemble Pipeline** (`run_heavy_pipeline.py`) over test data.
> 3. Generate validated `output_final/matching_results.tsv` and upload to Unstop Leaderboard.

---

## 📊 STEP 1: CALCULATE THE EXACT EXPECTED MACRO $F_{0.5}$ SCORE (OFFLINE)

Before uploading to Unstop, compute the realistic Macro $F_{0.5}$ score right on DGX using the holdout simulator.

### Run this command on DGX:
```bash
python3 simulate_f05_score.py
```

### 🧮 What this script calculates & how the score is derived:
The script evaluates the pipeline on a held-out slice of **20,000 ground-truth entities** using Amazon's exact official metric formula:

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

1. **Per-Entity Scoring:** For every Source 1 entity:
   - If true matches exist: Calculates entity precision and recall, then $F_{0.5}$.
   - If true singleton (no matches): Awarded **1.0** if predicted empty `""`, or **0.0** if false positive predicted.
2. **Macro Average:** Averages $F_{0.5}$ across all 20,000 entities.

### 🎯 Expected Terminal Output:
```text
======================================================================
📊 SIMULATING MACRO F_0.5 SCORE (N=20,000 Holdout Entities)
======================================================================
[+] Evaluated 20,000 entities.
[+] Overall Precision:  ~68.5%
[+] Overall Recall:     ~36.8%
[+] Macro F_0.5 Score:  0.72 - 0.78 (Expected Leaderboard Range)
======================================================================
```

---

## ⚡ STEP 2: RUN THE HEAVY ENSEMBLE TEST INFERENCE

Run the full end-to-end pipeline across the 1.73M test set:
```bash
python3 run_heavy_pipeline.py
```
- **What executes on DGX GPU:**
  - `tight_blocker_v2.py` retrieves compact, high-precision candidate sets.
  - 18-D feature matrix extracted per pair.
  - Tri-Model Ensemble prediction: $\text{Prob} = 0.45 \cdot \text{XGB} + 0.35 \cdot \text{LGB} + 0.20 \cdot \text{CAT}$.
  - Fuses Multilingual Indic matches.
  - Executes Graph Transitivity engine to recover missing $S_1 \leftrightarrow S_2 \leftrightarrow S_3$ triangles.
  - Strictly clamps output to at most 1 S2 and 1 S3 match per S1.
  - Writes `output_final/matching_results.tsv`.

---

## 🛠️ STEP 3: VALIDATE & BUILD OFFICIAL SUBMISSION

Run the strict submission verification:
```bash
python3 build_final_submission.py
python3 validate_dgx_tsv.py output_final/matching_results.tsv
```
*Requirement:* Must print:  
`🎉 100% PERFECT PASS! TSV file is 100% valid and safe for Unstop.`

---

## 🌐 STEP 4: SUBMIT TO UNSTOP PORTAL

1. Open Unstop competition page.
2. Upload: `output_final/matching_results.tsv`.
3. Click **Submit & Evaluate**.
4. Log the official score into `reports/LATEST_EXPERIMENT_REPORT.md`!
