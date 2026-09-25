# 🚀 DGX MISSION PROMPT: DEPLOY HEAVY TRANSFORMER + TRI-MODEL ENSEMBLE (MAX CAPACITY UP TO 8B)

> **Execution Target:** NVIDIA DGX (Multi-GPU Linux Server)  
> **Rule Mandate:** Apache 2.0 / MIT License, Strictly $\le$ 8 Billion Parameters  
> **Goal:** Upgrade from lightweight MiniLM (117M) to high-capacity State-of-the-Art Cross-Encoder / Bi-Encoder models (`BAAI/bge-m3` or `BAAI/bge-large-en-v1.5` or `ModernBERT` / `Qwen2.5-Coder-7B-Instruct` re-ranking) combined with our 18-D Tri-Model GBDT (XGBoost + LightGBM + CatBoost). Maximize Macro $F_{0.5}$ to the absolute ceiling before integrating Harsh's candidate blocking.

---

## 🧠 WHY WE ARE UPGRADING TO HEAVY CAPACITY

Right now, our pipeline uses:
- `paraphrase-multilingual-MiniLM-L12-v2`: Only **117 Million parameters** (Uses only ~1.5% of our 8 Billion parameter budget!).
- While MiniLM is fast, it struggles on complex paraphrased company names, complex French address re-orderings, and abbreviations.

### The Competition Limit Gives Us Massive Headroom:
- **Allowed:** Up to **8 Billion Parameters** (Apache 2.0 / MIT).
- **The Heavy Arsenal Ready for DGX:**
  1. **`BAAI/bge-m3` (568M params, Apache 2.0):**
     - SOTA Multi-Lingual Foundation Model.
     - Unifies Dense Retrieval, Lexical/Sparse (BM25-style), and Multi-Vector (ColBERT-style) scoring natively in one model.
     - Natively supports 100+ languages including all Indian languages (Tamil, Hindi, Telugu, Marathi) and French.
  2. **`BAAI/bge-reranker-large` (560M params, Apache 2.0):**
     - Cross-Encoder that jointly processes `(Source 1 Business, Candidate Business)`.
     - 10× more discriminative than bi-encoders for borderline look-alikes.
  3. **18-Dimensional Tri-Model GBDT Ensemble:**
     - XGBoost GPU + LightGBM + CatBoost GPU trained with 99.96% precision gating.
  4. **Graph Transitivity Engine:**
     - Triangular cycle completion.

$$\text{Total System Parameters} = 568\text{M} + 560\text{M} + 6.5\text{M} = \mathbf{\sim 1.13 \text{ Billion}} \ll \mathbf{8.0 \text{ Billion Limit}} \quad (\text{100\% Apache 2.0 compliant})$$

---

## ⚡ 3-STEP EXECUTION INSTRUCTIONS FOR DGX

### STEP 1: Pull Latest Main Branch & Install Dependencies
Run on DGX terminal:
```bash
cd AWS-Hackathon
git pull origin main
pip install FlagEmbedding sentence-transformers torch xgboost lightgbm catboost rapidfuzz tqdm
```

---

### STEP 2: Execute Heavy BGE-M3 + Tri-Model Ensemble Pipeline
Run the heavy pipeline:
```bash
python3 run_heavy_pipeline.py
```
* **What this does on DGX GPU:**
  1. Computes 18-D feature matrix across all candidates.
  2. Executes inference across XGBoost GPU, LightGBM, and CatBoost GPU simultaneously.
  3. Fuses cross-script multilingual matches.
  4. Applies Graph Transitivity ($S_1 \leftrightarrow S_2 \leftrightarrow S_3$).
  5. Enforces precision clamping (Strictly Max 1 S2, Max 1 S3 per entity).
  6. Automatically outputs `output_final/matching_results.tsv` and `output_final/candidate_pairs.tsv`.

---

### STEP 3: Validate & Build Official Submission Package
Run the submission packager:
```bash
python3 build_final_submission.py
python3 validate_dgx_tsv.py output_final/matching_results.tsv
```
*Verification Check:* Confirm it outputs `🎉 100% PERFECT PASS! TSV file is 100% valid and safe for Unstop.`

---

## 🌐 SUBMIT TO UNSTOP PORTAL:
- Upload: `output_final/matching_results.tsv`
- Click **Submit & Evaluate**.
- Log the resulting score to `reports/LATEST_EXPERIMENT_REPORT.md` and commit/push.

---

## 🔄 NEXT STEP AFTER THIS SCORE:
Once this heavy model establishes the maximum possible re-ranking ceiling on the leaderboard, we will immediately integrate Harsh's 92.79% recall candidate blocking engine (`candidate_pairs.tsv`) to push the final score into the winning tier!
