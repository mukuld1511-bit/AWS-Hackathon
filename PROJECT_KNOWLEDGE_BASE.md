# 📖 Amazon ML Challenge 2026: Simple Knowledge Base & Log

> **Motto:** Explain everything in simple, clear English. No unnecessary academic jargon. Every concept, feature, and model added to this project will be documented here with its plain-English meaning and why it helps us win.

---

## 📑 Table of Contents
1. [Simple Definitions of Key Terms](#1-simple-definitions-of-key-terms)
2. [Our 2-Stage Winning Pipeline](#2-our-2-stage-winning-pipeline)
3. [Evaluation Metrics Explained Simply](#3-evaluation-metrics-explained-simply)
4. [Leaderboard Strategy: How We Climb to Top 1%](#4-leaderboard-strategy-how-we-climb-to-top-1)
5. [Feature Catalog (What each feature does & why)](#5-feature-catalog-what-each-feature-does--why)
6. [Daily Experiment Log](#6-daily-experiment-log)

---

## 1. Simple Definitions of Key Terms

### A. Candidate Generation (Shortlisting the Top Matches)
* **In Simple Words:** If Amazon has 1,000,000 items in a catalog, comparing an item against all 1,000,000 one-by-one requires billions of comparisons. The computer will freeze or run out of memory.
* **What we do:** We use a lightning-fast search filter (like keyword search or quick vector lookup) to grab the **top 30 to 50 most promising candidates** in milliseconds.
* **The Goal:** Make sure the real match is always inside that top 30–50 list.

### B. Blocking Strategy (Smart Filtering)
* **In Simple Words:** Immediately throwing away obvious non-matches.
* **Example:** If searching for a "Nike Running Shoe", never look in the "Kitchen Blender" or "Sony Headphones" category. Only look inside the "Footwear + Nike" block.
* **Benefit:** Saves 99% of computation time and memory.

### C. Re-ranking (Fine-Grained Scoring)
* **In Simple Words:** Now that we have only 30 candidates from Stage 1, we run our powerful AI models (Deep Transformers, LightGBM, string matchers) on these 30 items to accurately decide which one is the true match (#1, #2, #3).

### D. Data Leakage (Cheating by Mistake)
* **In Simple Words:** Accidentally letting the model see the answers or patterns from the validation test set during training.
* **The Danger:** Your model will show a fake 99% score on your laptop, but when you submit to Unstop, it crashes down to 40% and you lose.
* **The Fix:** Use `GroupKFold` so that related items always stay together in either training OR validation—never split between both.

---

## 2. Our 2-Stage Winning Pipeline

```
[ Full Amazon Catalog / Test Pool (Millions of rows) ]
                       │
                       ▼  (STAGE 1: Fast Filter / Candidate Generation)
[ Only 30 to 50 Probable Candidates per Item ]  <-- (BM25 + Brand Filtering + Embeddings)
                       │
                       ▼  (STAGE 2: Smart Re-ranking)
[ Final Scored & Ranked Predictions ]           <-- (LightGBM + DeBERTa Transformer)
                       │
                       ▼  (Quality Check via src/submission_checker.py)
[ Final Submission CSV uploaded to Unstop ]
```

---

## 3. Evaluation Metric: $F_{0.5}$ (Macro-Averaged)

$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

### Why this changes our entire strategy:
* In standard $F_1$, Precision and Recall are weighted equally ($1:1$).
* In $F_{0.5}$, **Precision is weighted 2× more than Recall!**
* **Golden Rule:** **False Positives (wrong guesses) hurt us twice as much as missing a match.**
  - If unsure about a candidate $\rightarrow$ DO NOT MATCH IT.
  - High confidence predictions only!
* **Singletons (5.58% of data):**
  - If an entity in S1 has no matches, predicting an empty list gives a **perfect 1.0 score**.
  - Making even one wrong guess on a singleton drops its score straight to **0.0**!

---

## 4. Real Data Insights & Noise Patterns Discovered

1. **Total Data Scale:**
   - Train S1: 2,206,821 records | Train S2: 5,034,616 | Train S3: 5,285,603
   - Test S1: 1,732,544 records | Test S2: 4,887,273 | Test S3: 5,082,316
   - Test Country Split: India (46.8%), US (38.3%), France (15.0%).

2. **Strict Country Partitioning (0% Cross-Country Matches):**
   - S1 entities from US **only** match US records in S2/S3.
   - S1 entities from India **only** match India records in S2/S3.
   - S1 entities from France **only** match France records in S2/S3.
   - **Action:** Partition data by country first. This cuts comparison space by ~3× without losing a single true match!

3. **Observed Noise Categories:**
   - **Name Typos & OCR:** `Maure Wilblims` vs `Maure Williams`, `Dahlia Ponr` vs `Dahlia Power`.
   - **Legal Suffixes:** `LLP`, `Inc`, `LLC`, `Pvt Ltd`, `Center` added or stripped.
   - **Website Domains:** `maurewilliamscolombier.com` used as the business name.
   - **Multilingual Names (India):** Same company written in Tamil (`ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`), Hindi, or English.
   - **Address Reordering:** Street, city, and state components shuffled (`KANSAS CITY, MO, 630 45ND TERRACE`).
   - **Empty Address:** Many S2/S3 records have empty address strings, meaning name similarity must carry the weight.

---

## 4. Leaderboard Strategy: How We Climb to Top 1%

1. **Submission #1 (Tonight by 2:30 AM):** Build and submit a simple working baseline immediately. This registers our name near the top of the leaderboard while other teams are still figuring out what to do.
2. **The 5-Submissions-Per-Day Rule:**
   - **Sub 1:** Baseline pipeline verification.
   - **Sub 2:** Clean feature engineering (big score jump).
   - **Sub 3:** Improved blocking & candidate retrieval.
   - **Sub 4:** Deep Learning / Transformer model.
   - **Sub 5:** Ensemble of all best models to lock in our top rank for the night.
3. **Local Validation is Truth:** If a new model does not improve our local 5-fold CV score, we DO NOT submit it to Unstop. We never waste our 5 daily attempts.

---

## 5. Feature Catalog (What each feature does & why)

| Feature Name | What it Measures (Simple English) | Why it Helps in E-Commerce Data |
|---|---|---|
| `jaccard_word` | Word overlap ratio between two texts | Captures title similarity even if word order changes (e.g. "Nike Shoes Black" vs "Black Shoes Nike"). |
| `fuzzy_token_sort` | Typo and spelling tolerance | Catches slight spelling mistakes or different ordering of specs. |
| `num_unit_mismatch` | Number and Unit comparison (e.g., 500ml vs 1L) | If titles are identical but one is "500ml" and the other is "1L", they are DIFFERENT products. Crucial for high precision! |
| `dense_sim` | AI Meaning similarity (Embeddings) | Matches items that mean the same thing using different words (e.g. "wireless earphones" vs "bluetooth earbuds"). |

---

## 6. Daily Experiment Log

| Exp # | Timestamp | Owner | What was Changed | Local CV Score | Unstop LB Score | Notes / Verdict |
|---|---|---|---|---|---|---|
| 01 | 12:00 AM | Team | Initial Setup & Data Inspection | - | - | Ready for 12 AM Problem Statement |
