# 🚀 DGX GRANDMASTER V2 (0.98 TOPPER TARGET) - AUTONOMOUS EXECUTION PROMPT

> **Target Machine:** NVIDIA DGX / GB10 Blackwell Server (128 GB VRAM)  
> **Mission:** Execute Phase B (Deterministic Identity Anchoring) + Phase C (Discriminative Cross-Encoder) + Phase D (Global DSU Solver) to push Macro F0.5 to 0.90+ - 0.98.  
> **Constraints:** <= 8B parameters, Apache 2.0 / MIT licenses only, ZERO generative LLMs, strictly 1,732,544 rows in output TSV, max 1 S2 + max 1 S3 per S1.

---

## 📋 COPY-PASTE THIS ENTIRE PROMPT TO THE DGX AGENT / TERMINAL:

```markdown
Hello Antigravity DGX Agent,

You are acting as the Lead Machine Learning Engineer executing the high-precision **Phase B + C + D Grandmaster V2 Pipeline** on the DGX server for the Amazon ML Challenge 2026.

### 🎯 OBJECTIVE:
Climb from current baseline (0.418) past 0.58 all the way into the **0.90+ - 0.98 leaderboard topper range** using:
1. **Phase B (Deterministic Legal Identity Anchors):** Exact extraction and hashing of GSTIN, PAN, CIN (India), SIREN (France), EIN (US), E.164 Cleaned Phone Numbers, and Brand Domains (`src/deterministic_anchors.py`). True-positive match confidence = 0.999 (0% False Positives).
2. **18-D Tri-Model GBDT Ensemble:** Pretrained XGBoost GPU + LightGBM + CatBoost GPU (`xgb_heavy.json`, `lgb_heavy.txt`, `cat_heavy.cbm`).
3. **Phase C (Discriminative Cross-Encoder):** `BAAI/bge-reranker-base` (278M params, Apache 2.0) on ambiguous candidate pairs for 98%+ precision reranking.
4. **Phase D (Global DSU Solver & Transitivity):** Disjoint-Set Union graph cycle completion (S1 <-> S2 <-> S3) with Hungarian 1-to-1 maximum weight assignment to eliminate precision leakage (`src/global_dsu_solver.py`).

---

### ⚡ STEP-BY-STEP EXECUTION RUNBOOK:

#### Step 1: Sync Latest Master Codebase
cd ~/AWS-Hackathon
git fetch origin main
git checkout main
git pull origin main

#### Step 2: Install / Verify Dependencies
Ensure PyTorch GPU and transformers are present:
pip install --upgrade transformers sentence-transformers catboost xgboost lightgbm
python3 -c "import torch, transformers; print('PyTorch CUDA:', torch.cuda.is_available(), 'Device:', torch.cuda.get_device_name(0))"

#### Step 3: Run the Grandmaster V2 Pipeline
Execute the complete integrated pipeline:
python3 run_grandmaster_v2_98.py

*Expected execution time: ~10 to 18 minutes on DGX GPU.*  
*Output location:* `output_grandmaster_v2/matching_results.tsv` and `output_grandmaster_v2/candidate_pairs.tsv`

#### Step 4: Validate Competition Integrity Rules
Run the official validator to ensure zero penalties:
python3 validate_dgx_tsv.py output_grandmaster_v2/matching_results.tsv
python3 validate_submission.py --check-ids

*Criteria:*
- Exactly 1,732,544 data rows (+ 1 header row = 1,732,545 total lines)
- Tab \t delimited
- Header: source1_entity_id\tmatched_entity_ids
- Maximum 1 S2- and 1 S3- per entity
- Zero self-matches (S1- matching S1- is illegal)

#### Step 5: Package Final Submission
Once validation passes with ALL CHECKS PASSED:
mkdir -p final_sub_v2/output
cp output_grandmaster_v2/matching_results.tsv final_sub_v2/output/
cp output_grandmaster_v2/candidate_pairs.tsv final_sub_v2/output/
cp -r code final_sub_v2/ || true
cp requirements.txt final_sub_v2/ || true
cp Documentation_template.md final_sub_v2/ || true

cd final_sub_v2
zip -r ../final_submission_v2_98.zip .
cd ..
ls -lh final_submission_v2_98.zip

#### Step 6: Portal Submission & Reporting
Upload final_submission_v2_98.zip (or output_grandmaster_v2/matching_results.tsv) to Unstop as soon as the daily quota resets (05:30 AM IST / 00:00 UTC). Record the updated leaderboard score in reports/LATEST_EXPERIMENT_REPORT.md.
```