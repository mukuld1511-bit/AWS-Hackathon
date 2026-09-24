# 🧠 TASK FOR PRATEEK (Antigravity / AI Assistant Prompt)
## Module: Multilingual Semantic & Indic Transliteration Matcher

> **INSTRUCTION FOR PRATEEK'S ANTIGRAVITY / AI ASSISTANT:**  
> You are an expert Machine Learning & NLP engineer working on the **Amazon ML Challenge 2026 (Business Entity Resolution)**.  
> You are collaborating in a 4-person team where **Mukul (Lead)** owns the `main` branch, the primary heavy GBDT re-ranking pipeline, and official portal submissions.  
> **Your workspace branch:** `prateek`. Strictly follow the Git and memory guardrails outlined below.

---

## 🏢 1. COMPLETE PROJECT CONTEXT

### The Business Problem
We are resolving business entities across 3 independent data sources:
- **Source 1 (S1):** Clean reference entity source.
- **Source 2 (S2) & Source 3 (S3):** Noisy partner sources.
- **Goal:** For each Source 1 entity, identify corresponding matching records in Source 2 and Source 3.

### The Specific Challenge You Own: Multilingual Disconnect
- **Dataset Analysis:** **46.8% of all test entities are located in India** (809,986 records in S1, and millions in S2/S3).
- **The Empirical Discovery:** In Indian records, business identities frequently switch between English and regional Indic scripts:
  - English: `Raj Investments LLP`
  - Tamil: `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`
  - Hybrid: `Raj Investments எல்எல்பி`
  - Hindi: `राम मार्केटिंग प्राइवेट लिमिटेड` vs `Ram Marketing Pvt Ltd`
- Standard string matchers (Levenshtein, Jaccard, Token Sort) yield **0.0 similarity** between `Raj Investments` and `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`!
- **Your Mission:** Build a targeted, high-precision add-on matcher for **India records with Indic text** that recovers these cross-lingual matches.

### Mathematical Objective & Evaluation Metric
Macro-Averaged $F_{0.5}$ Score:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Metric Behavior:** Precision is weighted 2× more than Recall.  
  False Positives (incorrect merges) are heavily penalized.
- **Requirement:** Only predict matches when confidence is high (e.g. cosine similarity $> 0.85$ or combined with matching address components).

---

## 📊 2. DATASET SPECIFICATIONS & LOCATION

Files are Tab-Separated (`.tsv`):
- `student_resource/dataset/test/test_source1.tsv`
- `student_resource/dataset/test/test_source2.tsv`
- `student_resource/dataset/test/test_source3.tsv`
- (Ground truth for development/validation: `student_resource/dataset/train/train_ground_truth.tsv`)

### Data Filtering Filter:
Do NOT process the full 10M records. **Filter exclusively for:**
1. `country == 'India'`
2. Records where `business_name` contains non-ASCII characters:
   - Regex check: `bool(re.search(r'[\u0900-\u0D7F]', text))` (covers Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam).

---

## 🛠️ 3. YOUR EXACT TECHNICAL DELIVERABLES

You will create a standalone, runnable Python script:
`src/multilingual_matcher.py`

### Strategy Options (Implement Option A or Option B):

#### Option A: Multilingual Dense Embeddings (Recommended)
1. Load a lightweight multilingual sentence embedding model:
   ```python
   from sentence_transformers import SentenceTransformer
   # Lightweight, fast, high semantic alignment across Indic languages & English:
   model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
   ```
2. For Indian S1 entities containing Indic text (or whose address matches a potential candidate with Indic text), encode normalized names into 384-dimensional unit-norm vectors.
3. Compute cosine similarity between S1 name vector and candidate name vector.
4. If `cosine_sim >= 0.85`, record as a candidate match.

#### Option B: Fast Rule-Based Indic Script Detection & Address Bridging
1. If an S1 entity is in English (e.g. `6(29), C.I.T. Colony Mylapore, Chennai`) and candidate S2/S3 is in Tamil (`ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`), but their **address numbers and PIN code match exactly** (e.g. both have `6(29)` and `Mylapore` / `Chennai`):
   - Flag as an address-anchored cross-lingual match.

### Output File Format:
Export your high-confidence pairs to:
`output/multilingual_matches.tsv`
- Tab-separated header: `source1_entity_id\tmatched_entity_ids`
- Format: `S1-XXXXX\tS2-YYYYY,S3-ZZZZZ`
- Only include entities where your model confirmed a cross-lingual match.

---

## 🌿 4. GIT WORKFLOW & INTEGRATION PROTOCOL

1. **Switch to Your Dedicated Branch:**
   ```bash
   git checkout prateek
   git pull origin main
   ```
2. **Execute & Test Locally:**
   ```bash
   python src/multilingual_matcher.py
   ```
3. **Commit & Push to Your Branch Only:**
   - Never commit raw `.tsv` files or `.pt` model weights (they are gitignored).
   - Commit your code file:
     ```bash
     git add src/multilingual_matcher.py
     git commit -m "Prateek: Implement multilingual and Indic transliteration matcher"
     git push origin prateek
     ```
4. **Integration Handover:**
   Notify **Mukul**. Mukul will pull from `origin/prateek`, merge your multilingual discoveries into the main re-ranking pipeline, and calculate the overall local $F_{0.5}$ gain before submitting to Unstop.
