# 📝 TASK FOR AYUSH: Official Approach Documentation & Analysis

> **HOW TO USE THIS FILE:**  
> Copy this entire markdown and give it directly to your AI assistant.  
> Make sure you are working on your branch: `git checkout ayush`

---

## 🎯 YOUR OBJECTIVE
You are responsible for the **Official Solution Documentation** in the Amazon ML Challenge 2026.
- **The Context:** Amazon requires every team in the Top 100 to submit a comprehensive methodology document based on `student_resource/Documentation_template.md`.
- **Your Mission:** Continuously maintain, polish, and fill out this template as our team iterates on models. Prioritize technical depth, clarity, and Amazon-specific terminology.

---

## 📁 RELEVANT FILES
- Base Template: `student_resource/Documentation_template.md`
- Project Knowledge Base (Reference): `PROJECT_KNOWLEDGE_BASE.md`
- Target Output: `docs/Documentation_template.md` (or `.pdf`) inside your branch.

---

## ✍️ WHAT SECTIONS TO DRAFT & FILL

### Section 1: Executive Summary
- 2–3 sentences summarizing our two-stage architecture:
  1. Multi-key deterministic & phonetic blocking (Stage 1).
  2. Tabular GBDT (LightGBM) re-ranking with deep semantic and multilingual embeddings (Stage 2).
  3. Metric-aligned threshold tuning for macro $F_{0.5}$ score.

### Section 2: Methodology & Problem Analysis
- Detail real-world noise patterns observed during EDA:
  - Multilingual scripts: English vs Tamil (`ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`) and Hindi (`राम मार्केटिंग`).
  - Name variations: Legal suffix omissions (`Inc`, `LLC`, `Pvt Ltd`), typos (`Wilblims` vs `Williams`), DBA/domain name substitutions (`maurewilliamscolombier.com`).
  - Address corruptions: Component transposition (Street before City vs City before Street), landmark references (`Near SBI ATM`), missing PIN codes.
  - Explain why strict country partitioning (0% cross-country matching) reduces search space by 3× without loss of recall.

### Section 3: Candidate Generation (Blocking)
- Document the multi-key inverted index:
  - Key 1: `(country, normalized_name)`
  - Key 2: `(country, name_prefix + pincode)`
  - Key 3: `(country, address_tokens)`
- Reduction ratio achieved: Pruning 17 Trillion possible comparisons down to an average of <20 candidates per entity.

### Section 4: Matching Model & Features
- Explain feature engineering:
  - String similarity: RapidFuzz token sort ratio, token set ratio, Levenshtein distance.
  - Address matching: Token Jaccard overlap, exact PIN/number match indicators.
  - Multilingual: Cross-lingual semantic similarity via multilingual sentence transformers.
- Classifier: Gradient Boosted Decision Trees (LightGBM) trained on pairwise features with $F_{0.5}$-optimized decision boundary.

### Section 5: Evaluation & Error Analysis
- Explain why $F_{0.5}$ weights Precision 2× over Recall ($1.25 \times P \times R / (0.25P + R)$).
- Explain singleton strategy: Correctly predicting empty match yields 1.0; false positives penalize heavily.

### Git Instructions
1. Update `Documentation_template.md`.
2. Commit inside your branch:
   ```bash
   git checkout ayush
   git add Documentation_template.md
   git commit -m "Ayush: Update solution methodology documentation"
   git push origin ayush
   ```
