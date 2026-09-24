# 🧠 TASK FOR PRATEEK: Multilingual & Transliteration Matching Model

> **HOW TO USE THIS FILE:**  
> Copy this entire markdown and give it directly to your AI coding assistant (Cursor / Claude / Copilot).  
> Make sure you are working on your branch: `git checkout prateek`

---

## 🎯 YOUR OBJECTIVE
You are responsible for the **Multilingual & Transliteration Add-On Model** in the Amazon ML Challenge 2026.
- **The Problem:** 46.8% of test records are from India. In real Indian business data, the same business name appears in English, Tamil (`ராஜ் இன்வெஸ்ட்மெண்ட்ஸ்`), Hindi (`राम मार्केटिंग`), or Kannada. Standard string matching algorithms (Levenshtein, Jaccard) give a score of 0.0 to these transliterated pairs!
- **Your Mission:** Build a lightweight add-on model that detects and scores cross-lingual / transliterated business name matches.

---

## 📁 RELEVANT FILES & DATA
- Data Location: `student_resource/dataset/test/`
  - `test_source1.tsv`
  - `test_source2.tsv`
  - `test_source3.tsv`
- We only need to apply this model to **India** records (`country == 'India'`) where non-ASCII characters appear!

---

## 🛠️ REQUIRED TECHNICAL SPECIFICATION
Write a standalone Python script: `src/multilingual_matcher.py`.

### 1. Technique Options (Choose either A or B):
- **Option A (Lightweight Fast Pretrained Embeddings - Recommended):**
  - Use `sentence-transformers` with a compact multilingual model:
    `model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')`
  - For Indian records containing Indic scripts (Unicode range `\u0900-\u0D7F`), generate 384-dimensional embeddings.
  - Calculate cosine similarity between S1 and S2/S3 business names.
  - If cosine similarity $> 0.82$, output as a high-confidence match!
- **Option B (Indic Transliteration / Romanization):**
  - Use `indic-transliteration` or `unicodedata` / phonetic mapping to convert Devanagari/Tamil scripts to Roman script.
  - Then run token matching on the romanized strings.

### 2. Output Format
Export a high-confidence match dictionary or TSV: `output/multilingual_matches.tsv`:
- Header: `source1_entity_id\tmatched_entity_ids`
- Only include Indian entities where a multilingual match was discovered.

### 3. Execution Constraints
- Do not process English-only records if already handled by exact match; filter specifically for records where at least one entity has non-ASCII / Indic characters.
- Keep batch size moderate (e.g. 64 or 128) if using PyTorch/GPU.

### 4. Git Instructions
1. Run and verify your script.
2. Commit your code inside your branch:
   ```bash
   git checkout prateek
   git add src/multilingual_matcher.py
   git commit -m "Prateek: Implement multilingual transliteration matcher"
   git push origin prateek
   ```
3. Notify Mukul so he can integrate your cross-lingual matches into the main ensemble.
