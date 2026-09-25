# 🚀 DGX EXECUTION PROMPT: SUBMISSION 5 (TARGET: 0.65+ / 0.80+)

> **Mission:** Run Prateek's Multilingual Indic Transliteration Matcher on GPU and fuse with Sub 4's XGBoost Re-ranker to boost score to 0.65+ / 0.80+ on the Unstop Leaderboard.
> **Current Baseline:** 0.415 (Sub 4)
> **Key Gain:** 46.8% of test data is India — Prateek's model captures Tamil, Hindi & Devanagari cross-lingual matches that standard models miss completely.

---

## ⚡ 3-STEP COMMAND SEQUENCE ON DGX

Copy-paste these commands sequentially into the DGX terminal:

### STEP 1: Git Pull Latest Code & Install Model Requirements
```bash
cd AWS-Hackathon
git pull origin main

# Install sentence-transformers if not already present
pip install sentence-transformers torch rapidfuzz tqdm
```

---

### STEP 2: Run Prateek's Multilingual Matcher on DGX GPU
```bash
python3 src/multilingual_matcher.py --batch-size 512 --chunk-size 25000
```
* **What it does:**
  - Encodes 867k Indic candidates in GPU FP16 (`output/indic_candidates_cache.pt`).
  - Streams India entities in 25,000 chunks with auto-checkpointing.
  - Generates `output/multilingual_matches.tsv` (high precision matches: Cosine $\ge 0.95$ OR Cosine $\ge 0.45$ with Address Overlap $\ge 7$).

---

### STEP 3: Fuse Sub 4 + Multilingual Matcher into Submission 5
```bash
python3 build_sub5_ensemble.py
```
* **What it does:**
  1. Takes proven Sub 4 matches (0.415 score base).
  2. Injects high-confidence Indic transliteration matches for entities where Sub 4 had no match.
  3. Enforces strict competition constraints (Max 1 S2, Max 1 S3, exactly 1,732,544 rows).
  4. Automatically runs `validate_dgx_tsv.py` (prints `100% PERFECT PASS`).
  5. Produces:
     - **TSV File for Direct Portal Upload:** `sub5_output/matching_results.tsv`
     - **Full Package Zip (Backup):** `sub5_team_submission.zip`

---

## 🌐 UPLOAD TO UNSTOP PORTAL:
- Open Unstop code round page.
- In **Upload Matching Results File**, select:
  👉 `sub5_output/matching_results.tsv`
- Click **Submit & Evaluate**!
