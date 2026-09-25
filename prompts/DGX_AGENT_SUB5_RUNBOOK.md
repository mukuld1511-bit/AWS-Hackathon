# 🤖 INSTRUCTIONS FOR DGX AI AGENT: EXECUTE SUBMISSION 5 (TARGET: 0.65+ / 0.80+)

> **Agent Role:** Autonomous DGX Compute & Pipeline Runner  
> **Environment:** NVIDIA DGX (Multi-GPU Linux Server, Python 3.10+, PyTorch CUDA)  
> **Mission:** Pull latest `main` branch, execute the GPU-accelerated Multilingual Indic Matcher, fuse with Sub 4's proven XGBoost GBDT model (0.415 LB base), validate submission integrity, and prepare `sub5_output/matching_results.tsv` for Unstop Leaderboard upload.

---

## 🎯 OBJECTIVE & MATHEMATICAL CONTEXT

1. **Current Baseline:** Submission 4 scored **`0.415`** on Unstop Leaderboard using XGBoost re-ranking with a strict `0.85` threshold.
2. **The Problem Being Solved:**
   - 46.8% of the test dataset consists of Indian entities.
   - Ground truth contains heavy cross-script transliteration noise (e.g. business names written in Tamil `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` or Hindi matching English Source 2/3 entities).
   - Standard string matching fails completely on cross-script names.
3. **The Solution (Submission 5 Ensemble):**
   - Run `src/multilingual_matcher.py` (Sentence-Transformers `paraphrase-multilingual-MiniLM-L12-v2` in FP16).
   - Calibrated 2-tier decision rule:
     - **Tier 1:** Transliteration similarity $\ge 0.95$.
     - **Tier 2:** Transliteration similarity $\ge 0.45$ AND Address token overlap $\ge 7$.
   - Fuse these new high-precision matches into Sub 4's base where Sub 4 had no match.
   - Enforce Competition Constraints: Max 1 match from S2, Max 1 match from S3 per entity, strictly 1,732,544 rows.

---

## ⚡ TASK EXECUTION STEPS (EXECUTE SEQUENTIALLY)

### STEP 1: Git Sync & Dependency Verification
Run the following in the project root:
```bash
git pull origin main
pip install sentence-transformers torch rapidfuzz tqdm
```
Verify GPU availability:
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

### STEP 2: Execute Multilingual Indic Matcher
Run the matcher on GPU:
```bash
python3 src/multilingual_matcher.py --batch-size 512 --chunk-size 25000
```
- **Expected Artifacts Produced:**
  - `output/indic_candidates_cache.pt` (Precomputed candidate embeddings cache)
  - `output/multilingual_matches.tsv` (Output TSV containing high-precision Indic matches)
  - `output/matcher_checkpoint.json` (Progress tracking)

---

### STEP 3: Build & Validate Submission 5 Ensemble
Run the fusion script:
```bash
python3 build_sub5_ensemble.py
```
- **What this script executes automatically:**
  1. Reads `sub3_output/matching_results.tsv` (Sub 4 base) and `output/multilingual_matches.tsv`.
  2. Merges predictions adhering strictly to macro $F_{0.5}$ precision rules.
  3. Formats output with 1,732,544 rows and single-tab delimiters into `sub5_output/matching_results.tsv`.
  4. Runs `validate_dgx_tsv.py` to confirm `100% PERFECT PASS`.
  5. Packages `sub5_team_submission.zip`.

---

### STEP 4: Confirm Final Submission File
Confirm the output file exists and is ready for portal submission:
```bash
ls -lh sub5_output/matching_results.tsv
python3 validate_dgx_tsv.py sub5_output/matching_results.tsv
```

---

## 🌐 PORTAL SUBMISSION INSTRUCTIONS
On the Unstop competition page (`https://unstop.com/competitions/1743604/round/1593683/play/code`):
1. In **Upload Matching Results File**, upload:
   👉 `sub5_output/matching_results.tsv`
2. Click **Submit & Evaluate**.
3. Record the resulting leaderboard score and log it to `reports/LATEST_EXPERIMENT_REPORT.md`.
