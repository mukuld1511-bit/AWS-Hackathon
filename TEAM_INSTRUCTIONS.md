# 🚨 TEAM GUIDELINES & PROTOCOLS (Amazon ML Challenge 2026)

> **ATTENTION (All 4 Team Members & AI Assistants):**  
> Read this carefully before touching the repository or running any experiments. Adhere strictly to these DOs and DONTs.

---

## 👑 1. SUBMISSION AUTHORITY
* **DO NOT** make any submission directly on the Unstop portal.
* **ONLY Mukul** will submit solutions on Unstop.
* Daily limit is strictly **5 submissions per day** (Day 1, 2, 3 = max 15 total). Wasting even 1 submission is not an option.

---

## ✅ DOs (Mandatory Practices)

1. **Validate Before Sharing:**
   - Always run the validator script on any generated CSV before handing it to Mukul:
     ```bash
     python src/submission_checker.py <path_to_submission_csv> <path_to_test_csv>
     ```
   - Verify: zero NaNs, exact row count, and correct ID order matching `test.csv`.

2. **Track Every Run in `submissions/SUBMISSION_TRACKER.csv`:**
   - Log your model name, local Cross-Validation (CV) score, metric used, and key changes.

3. **Use Feature Branches:**
   - Create your own branch for experiments (e.g., `git checkout -b experiment/<your-name>-<technique>`).
   - Pull the latest `main` before starting new experiments.

4. **Document Experiments as You Go:**
   - Keep notes of what worked and what didn’t (features, model configs, scores).
   - A mandatory **1-2 page Approach Document** is required for the best solution to qualify for the Top 100.

## ⚡ 2. COMPUTE & HARDWARE GUIDELINES
* Train on our high-performance GPU environment.
* Keep training scripts framework-agnostic (standard PyTorch / HuggingFace / LightGBM).
* Focus compute power on:
  - Deep Transformers (DeBERTa-v3 / modern Cross-Encoders).
  - High-speed vector candidate retrieval (FAISS / ScaNN).
  - Robust 5-Fold Cross-Validation runs without cutting corners.

## ❌ DONTs (Strictly Prohibited)

1. **DO NOT commit datasets or model checkpoints to Git:**
   - Never commit `train.csv`, `test.csv`, `.zip`, `.pt`, `.bin`, or large generated parquet files.
   - Keep raw and processed data strictly inside `data/raw/` and `data/processed/` (which are gitignored).

2. **DO NOT push directly to `main`:**
   - Never push untested or breaking changes directly to the `main` branch.

3. **DO NOT log in simultaneously to Unstop:**
   - The challenge rules strictly prohibit simultaneous logins on Unstop from multiple devices/browsers. It can lead to account suspension or disqualification. Only the designated leader should access the challenge portal.

4. **DO NOT rely on single train-test splits:**
   - Never evaluate models on a simple random train/test split. Always use a proper K-Fold (GroupKFold / StratifiedKFold) to prevent data leakage and leaderboard shakeups.

5. **DO NOT submit without local CV improvement:**
   - Never burn one of our 5 daily submissions on an unverified intuition or a model that didn't show measurable improvement on local CV.
