# ⚡ TASK FOR HARSH: Multi-Key Candidate Generation & Blocking

> **HOW TO USE THIS FILE:**  
> Copy this entire markdown and give it directly to your AI coding assistant (Cursor / Claude / Copilot).  
> Make sure you are working on your branch: `git checkout harsh`

---

## 🎯 YOUR OBJECTIVE
You are responsible for the **Candidate Generation (Blocking) stage** in the Amazon ML Challenge 2026 Business Entity Resolution challenge.
- **The Problem:** We have ~1.73M Source 1 entities and ~10M records in Source 2 and Source 3. Full comparison ($O(N^2)$) is impossible. Our baseline model only matched exact business names, missing any entity with typos, missing legal suffixes, or matching solely on address.
- **Your Mission:** Build a fast, lightweight, memory-efficient candidate generator that outputs **top 10 to 20 candidate matches per Source 1 entity** with high recall.

---

## 📁 RELEVANT FILES & DATA
- Data Location: `student_resource/dataset/test/`
  - `test_source1.tsv` (1,732,544 rows: `entity_id`, `business_name`, `business_address`, `country`)
  - `test_source2.tsv` (4,887,273 rows)
  - `test_source3.tsv` (5,082,316 rows)
- Evaluation / Target Format: `output/candidate_pairs.tsv`
  - Format: Tab-separated (`.tsv`)
  - Header: `source1_entity_id\tcandidate_entity_ids`
  - Example: `S1-00001\tS2-00047,S2-00193,S3-00812`

---

## 🛠️ REQUIRED TECHNICAL SPECIFICATION
Write a standalone Python script: `src/candidate_blocking.py`.

### 1. Hard Rule: Strict Country Partitioning
- S1 entities from `US` only match S2/S3 records from `US`.
- S1 entities from `India` only match S2/S3 records from `India`.
- S1 entities from `France` only match S2/S3 records from `France`.
- *Never match across different countries!*

### 2. Multi-Key Inverted Index Strategy
Build an inverted index over Source 2 and Source 3 using 3 complementary blocking keys:
1. **Key 1: Normalized Name Key:**
   - Lowercase, strip punctuation (`.,-&/'"`), remove legal suffixes (`inc`, `llc`, `pvt`, `ltd`, `corp`, `co`, `limited`, `corporation`).
2. **Key 2: Name-Prefix + Location Anchor:**
   - First 4 characters of normalized business name + extracted numbers/PIN from address (e.g. `(country, name[:4] + pin)`).
3. **Key 3: Clean Address Key (for entities matching by location despite DBA/Trade name changes):**
   - Numbers in address + first token of street name (e.g., `85 wayne`).

### 3. Execution & Performance Constraints
- Keep it lightweight! Do NOT load everything into massive uncompressed pandas dataframes that cause Out-Of-Memory (OOM).
- Use standard Python `csv` or `defaultdict(list)`.
- Cap candidates at **maximum 20 candidates per S1 entity** to keep downstream inference fast.
- If no candidates match, output an empty candidate string: `S1-XXXXX\t\n`.

### 4. Git Instructions
1. Run and verify your script generates `output/candidate_pairs.tsv`.
2. Commit your code inside your branch:
   ```bash
   git checkout harsh
   git add src/candidate_blocking.py
   git commit -m "Harsh: Implement multi-key candidate blocking generator"
   git push origin harsh
   ```
3. Notify Mukul to inspect and merge your candidates into the main re-ranking pipeline.
