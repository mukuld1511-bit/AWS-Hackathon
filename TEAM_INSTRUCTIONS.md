# 🚨 TEAM ROLES & WORKFLOW PROTOCOL (Amazon ML Challenge 2026)

> **Motto:** Mukul owns the core engine and official submissions. Teammates run parallel experiments to boost our ensemble and manage documentation.

---

## 👑 1. Submission Authority & Core Pipeline
* **Portal Submissions:** **ONLY Mukul** will submit files to Unstop.
* **Core Engine:** Mukul manages the primary training pipeline, local validation (CV), and benchmark models.
* **Rule:** Daily limit is strictly **5 submissions per day**.

---

## 👥 2. Strategic Technical Task Division

### 🎯 Mukul (Lead / Core Pipeline & Feature Re-ranker)
- **Branch:** `main`
- **Focus:** 
  1. Build local validation split (100k S1 samples) with exact macro $F_{0.5}$ scorer.
  2. Develop tabular feature engineering & primary GBDT (LightGBM) re-ranking model.
  3. Optimize classification probability threshold specifically for $F_{0.5}$ precision.
  4. Final model ensembling and portal submissions.

### ⚡ Harsh (Candidate Generation & Multi-Key Blocking)
- **Branch:** `harsh`
- **Focus:**
  1. Expand candidate generation beyond exact names to capture misspelled entities and address-only matches.
  2. Implement Multi-Key blocking:
     - `(country, normalized_name)`
     - `(country, zip/pin + name_prefix)`
     - `(country, address_tokens)`
  3. Ensure candidate pool per S1 entity stays between 15–25 candidates while maximizing recall.
  4. Output clean `candidate_pairs.tsv` for downstream ranking.

### 🧠 Prateek (Multilingual Embeddings & Deep Matching)
- **Branch:** `prateek`
- **Focus:**
  1. Handle transliterations and cross-lingual matches (Tamil, Hindi, English in Indian records).
  2. Use a multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) to compute semantic similarities.
  3. Build a lightweight Cross-Encoder / pair-scorer to capture non-lexical semantic equivalence.
  4. Feed similarity scores to Mukul for blending with the GBDT re-ranker.

### 📝 Ayush (Documentation & Error Analysis)
- **Branch:** `ayush`
- **Focus:**
  1. Fill the official `student_resource/Documentation_template.md` with problem formulation and architecture details.
  2. Track false positives vs false negatives on validation sets.
  3. Maintain final submission package readiness.

---

## 🌿 3. Branching & Git Guidelines
- Mukul: `main`
- Ayush: `ayush`
- Harsh: `harsh`
- Prateek: `prateek`

**Workflow:**
1. Work inside your branch.
2. Push your models and notebooks to your branch so Mukul can pull and inspect them.
3. If you generate a candidate prediction CSV, validate it using:
   ```bash
   python src/submission_checker.py <your_file.csv> <test.csv>
   ```
   and pass it to Mukul for local CV evaluation and potential ensemble.
