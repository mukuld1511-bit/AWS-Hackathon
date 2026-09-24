# ⚡ TASK FOR HARSH (Antigravity / AI Assistant Prompt)
## Module: High-Recall Multi-Key Candidate Generation & Blocking Engine

> **INSTRUCTION FOR HARSH'S ANTIGRAVITY / AI ASSISTANT:**  
> You are an expert Machine Learning engineer working on the **Amazon ML Challenge 2026 (Business Entity Resolution)**.  
> You are collaborating in a 4-person team where **Mukul (Lead)** owns the `main` branch, the primary heavy GBDT re-ranking pipeline, and official portal submissions.  
> **Your workspace branch:** `harsh`. Strictly follow the Git and memory guardrails outlined below.

---

## 🏢 1. COMPLETE PROJECT CONTEXT

### The Business Problem
In commercial platforms (like Amazon), merchant and business data arrives from heterogeneous, noisy external sources without shared primary keys.  
We are given 3 independent data sources:
- **Source 1 (S1):** Clean, deduplicated reference anchor source.
- **Source 2 (S2) & Source 3 (S3):** Noisy external sources containing spelling errors, legal suffix variations, swapped address components, transliterated names, or domain names as business identities.
- **Goal:** For every Source 1 entity, identify all corresponding matching entity IDs from Source 2 and Source 3.

### Mathematical Objective & Evaluation Metric
Submissions are evaluated on **Macro-Averaged $F_{0.5}$ Score**:
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Why this matters for your module:**  
  Downstream re-ranking models cannot evaluate all $1.73\text{M} \times 10\text{M} = 17\text{ Trillion}$ pairs.  
  **Your module (Stage 1 Blocking) establishes the Recall Ceiling of our entire team.**  
  If a true match is excluded from your candidate pool, the downstream classifier has a 0% chance of predicting it. However, if your candidate pool is too large (e.g., >50 per entity), downstream models will choke or suffer high false positive rates.
  **Target:** Establish **>92% Recall** while keeping the average candidate pool between **10 to 25 candidates per S1 entity**.

---

## 📊 2. DATASET SPECIFICATIONS & CONSTRAINTS

### Directory Structure & Delimiters
All files are Tab-Separated (`.tsv`) to prevent parsing errors with comma-containing addresses:
```
student_resource/
└── dataset/
    ├── train/
    │   ├── train_source1.tsv           (2,206,821 records)
    │   ├── train_source2.tsv           (5,034,616 records)
    │   ├── train_source3.tsv           (5,285,603 records)
    │   └── train_ground_truth.tsv      (2,206,821 records)
    └── test/
        ├── test_source1.tsv            (1,732,544 records)
        ├── test_source2.tsv            (4,887,273 records)
        └── test_source3.tsv            (5,082,316 records)
```

### Record Columns:
1. `entity_id`: e.g. `S1-925783039`, `S2-166376419`, `S3-202863386`
2. `business_name`: Text with typos (`Maure Wilblims`), legal tags (`LLC`, `Pvt Ltd`), domain names (`xyz.com`).
3. `business_address`: Address with reordered tokens, missing state, or landmark references (`85 Wayne Ave, NY`).
4. `country`: String label (`US`, `India`, and in test set: `France`).

### 🚨 Golden Empirical Discovery (Zero Cross-Country Matching)
Analysis of millions of training ground-truth pairs confirms: **0.000% of matches cross country boundaries.**
- An entity with `country == 'US'` **ONLY** matches S2/S3 records where `country == 'US'`.
- An entity with `country == 'India'` **ONLY** matches S2/S3 records where `country == 'India'`.
- An entity with `country == 'France'` **ONLY** matches S2/S3 records where `country == 'France'`.
- **Mandatory Rule:** Country is your first hard blocking partition. This immediately prunes the search space by ~3× without losing a single true match.

---

## 🛠️ 3. YOUR EXACT TECHNICAL DELIVERABLES

You will create a production-grade Python script:
`src/candidate_blocking.py`

### Strategy: Multi-Key Inverted Index Blocking
A single blocking key (e.g. exact name) is insufficient because many entities match by address despite complete trade-name divergence (e.g. `Dréxkor` matching `Maure Williams Colombier Inc` at `85 Wayne Avenue`).

Implement an inverted index with 3 complementary keys:
1. **Key A (Normalized Name Anchor):**
   - Clean punctuation (`re.sub(r'[^\w\s]', ' ', text)`).
   - Strip corporate legal suffixes: `\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|co|company|enterprises|services|center)\b`.
   - Token-sort or normalized string: `(country, norm_name)`.
2. **Key B (Name Prefix + Address Number Anchor):**
   - Extracts numbers from address (e.g., `85`, `630`, PIN/ZIP codes like `560001` or `90210`).
   - First 4 letters of normalized name + address number: `(country, name[:4] + "_" + addr_numbers)`.
3. **Key C (Street Address Anchor):**
   - For businesses with completely different names (subsidiaries, website domains):
   - Address numbers + first non-numeric street token (e.g. `(country, "85_wayne")`).

### Execution Constraints & Performance Guardrails:
1. **Memory Discipline:** Do NOT load full 10M rows into memory with raw `pandas.read_csv`. Stream rows with Python's built-in `csv.reader(..., delimiter='\t')` or use chunking.
2. **Candidate Cap:** Limit each S1 entity's candidate pool to **at most 25 candidates**.
3. **Output File Format:** Save directly to `output/candidate_pairs.tsv`.
   - Header: `source1_entity_id\tcandidate_entity_ids` (separated by a single tab `\t`).
   - S1 with no candidates must output an empty tab: `S1-XXXXX\t\n`.
   - Comma-separated candidate list without quotes or spaces: `S2-001,S3-002,S2-005`.
   - Must contain **all 1,732,544 Source 1 entities** in the exact test set order.

---

## 🌿 4. GIT WORKFLOW & INTEGRATION PROTOCOL

1. **Verify Active Branch:**
   ```bash
   git checkout harsh
   git pull origin main
   ```
2. **Run and Verify Locally:**
   Execute your script on the test set:
   ```bash
   python src/candidate_blocking.py
   ```
   Verify your output with the competition validator:
   ```bash
   python student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test
   ```
3. **Commit & Push to Your Branch Only:**
   - **DO NOT** commit datasets, parquet files, or `.zip` archives (they are gitignored).
   - Commit your code file:
     ```bash
     git add src/candidate_blocking.py
     git commit -m "Harsh: Implement multi-key inverted index candidate blocking"
     git push origin harsh
     ```
4. **Integration Handover:**
   Notify **Mukul**. Mukul will merge your branch into `main`, evaluate candidate recall on the 100k local validation split, and feed your candidate pool into the heavy GBDT re-ranker.
