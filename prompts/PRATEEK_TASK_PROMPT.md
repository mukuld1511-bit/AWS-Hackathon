# 🧠 TASK FOR PRATEEK — Antigravity / AI Assistant Prompt
## Module: Multilingual Semantic & Indic Transliteration Matcher

> **INSTRUCTION FOR PRATEEK'S AI ASSISTANT:**  
> You are an expert NLP & Deep Learning engineer working on the **Amazon ML Challenge 2026 (Business Entity Resolution)**.  
> You are collaborating in a 4-person team. **Mukul (Lead)** owns the `main` branch, the primary heavy GBDT re-ranking pipeline, and official portal submissions. Your role is to build a **standalone multilingual matching add-on** that Mukul will merge into the main pipeline to boost cross-lingual entity resolution.  
> **Your dedicated Git branch:** `prateek`. You must NEVER push to `main`.

---

## 🏢 1. COMPLETE PROBLEM STATEMENT

### What is Entity Resolution?
We are given business records from 3 independent sources. Each source has noisy, inconsistent data about the same real-world businesses. The task is to determine which records across sources refer to the same business.

### The 3 Data Sources
- **Source 1 (S1):** Clean, deduplicated reference source (the "anchor").
- **Source 2 (S2) & Source 3 (S3):** Noisy external sources with spelling errors, legal suffix variations, transliterated names in regional languages, and domain names used as business identities.

### The Task
For every Source 1 entity, find all matching records from Source 2 and Source 3. An S1 entity may have zero, one, or many matches.

### Evaluation Metric: Macro-Averaged $F_{0.5}$
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Precision is weighted 2× more than Recall.** False merges are catastrophic.
- **Singletons:** Correctly predicting "no match" (empty list) earns a perfect 1.0. One wrong guess on a singleton drops it to 0.0.
- **YOUR IMPLICATION:** Only output matches when you are **highly confident** (cosine similarity ≥ 0.85 or strong address+name overlap). Low-confidence guesses will destroy the team's overall score.

---

## 📥 2. DATASET DOWNLOAD & SETUP

### Step 1: Download the dataset
Download from this URL (1.09 GB zip):
```
https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip
```

### Step 2: Extract into the repo root
After extraction, the directory structure should be:
```
AWS-Hackathon/
├── student_resource/
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── train_source1.tsv        (2,206,821 rows)
│   │   │   ├── train_source2.tsv        (5,034,616 rows)
│   │   │   ├── train_source3.tsv        (5,285,603 rows)
│   │   │   └── train_ground_truth.tsv   (2,206,821 rows)
│   │   └── test/
│   │       ├── test_source1.tsv         (1,732,544 rows)
│   │       ├── test_source2.tsv         (4,887,273 rows)
│   │       └── test_source3.tsv         (5,082,316 rows)
│   ├── utils/
│   │   └── validate_submission.py
│   └── Documentation_template.md
├── src/
├── prompts/
├── output/
└── ...
```

### Step 3: File format
**ALL files are Tab-Separated `.tsv`.** Always use `sep="\t"` or `delimiter="\t"`:
```python
import csv
with open("student_resource/dataset/test/test_source1.tsv", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)  # ['entity_id', 'business_name', 'business_address', 'country']
```

---

## 📂 2.5. EXACT FILE USAGE CONTRACT — READ THIS CAREFULLY

### ✅ FILES YOU MUST READ (Input):
| File | Purpose | When to use |
|---|---|---|
| `student_resource/dataset/test/test_source1.tsv` | **S1 entities — your "anchors" to match against** | ALWAYS — filter for India country |
| `student_resource/dataset/test/test_source2.tsv` | **S2 candidates — find Indic-script names here** | ALWAYS — filter for India + Indic chars |
| `student_resource/dataset/test/test_source3.tsv` | **S3 candidates — find Indic-script names here** | ALWAYS — filter for India + Indic chars |
| `student_resource/dataset/train/train_ground_truth.tsv` | **Known matches for local validation** | OPTIONAL — to check your matcher's accuracy |
| `student_resource/dataset/train/train_source1.tsv` | Train S1 entities | OPTIONAL — for local validation |
| `student_resource/dataset/train/train_source2.tsv` | Train S2 entities | OPTIONAL — for local validation |
| `student_resource/dataset/train/train_source3.tsv` | Train S3 entities | OPTIONAL — for local validation |

### 📝 FILE YOU MUST WRITE (Output):
| File | Format | Description |
|---|---|---|
| `output/multilingual_matches.tsv` | TSV with header `source1_entity_id\tmatched_entity_ids` | Only rows where you found cross-lingual matches. NOT all 1.7M rows — only the ones you matched. |

### 🚫 FILES YOU MUST NEVER TOUCH / MODIFY:
- **`src/baseline.py`** — Mukul's main pipeline. Do not edit.
- **`src/candidate_blocking.py`** — Harsh's module. Do not edit.
- **Any file inside `student_resource/`** — Dataset is read-only.
- **Any file on the `main` branch** — You work only on `prateek` branch.

### 🔄 HOW YOUR OUTPUT GETS USED (Merge Protocol):
> **You do NOT need to produce the final submission.**
> Your job is ONLY to produce `output/multilingual_matches.tsv`.
> 
> **Mukul will:**
> 1. Pull your `prateek` branch
> 2. Run your `src/multilingual_matcher.py` to regenerate matches
> 3. Fix any bugs, format issues, or false positives in your output
> 4. **Union-merge** your matches into the main pipeline's final `matching_results.tsv`
> 5. Submit on Unstop
> 
> **So don't worry about:**
> - Producing `matching_results.tsv` (that's Mukul's job)
> - Matching non-India entities or English-to-English pairs (Mukul's baseline handles those)
> - Integrating with Harsh's blocking code (Mukul merges everything)
> 
> **Just focus on:** Finding India-country cross-lingual matches (English ↔ Tamil/Hindi/Kannada/etc.) that the English-only baseline would miss.

---

## 📊 3. THE SPECIFIC PROBLEM YOU ARE SOLVING

### Why Standard String Matching Fails on Indian Data
**46.8% of all test entities are from India** (809,986 S1 entities). In Indian records, the same business name frequently appears in different scripts:

| S1 (English) | S2/S3 (Regional Script) | Levenshtein Score | Jaccard Score |
|---|---|---|---|
| `Raj Investments LLP` | `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` (Tamil) | 0.0 | 0.0 |
| `Ram Marketing Pvt Ltd` | `राम मार्केटिंग प्राइवेट लिमिटेड` (Hindi) | 0.0 | 0.0 |
| `EFS Print Ventures Ltd` | `Pvt. EFS Print Ventures Ltd.` (with Kannada state: `ಕರ್ನಾಟಕ`) | Partial | Partial |

Standard string metrics give **0.0** between English and Indic script names. These matches are completely invisible to Mukul's baseline and GBDT re-ranker.

### Real Matched Pair from Training Data:
```
ANCHOR:  S1-55344266
  Name:    "Raj Investments LLP"
  Address: "6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu"
  Country: India

MATCH 1: S2-249013014
  Name:    "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"      ← Full Tamil transliteration
  Address: "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu"

MATCH 2: S2-197070651
  Name:    "Raj Investments LLP"                       ← Exact English (already caught by baseline)
  Address: "6(29), C.I.T. COLONY, 2ND MAIN ROAD MYLAPORE, CHENNAI, Tamil Nadu"

MATCH 3: S3-478195123
  Name:    "Raj Investments எல்எல்பி"                  ← Hybrid: English name + Tamil legal suffix
  Address: "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN"

MATCH 4: S3-384364074
  Name:    "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"         ← Full Tamil
  Address: "6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, தமிழ்நாடு"  ← Tamil state name
```

**Your module must catch Match 1, Match 3, and Match 4** (the cross-lingual ones that Mukul's baseline misses entirely).

### Indic Script Unicode Ranges (for detection):
| Language | Unicode Range |
|---|---|
| Devanagari (Hindi/Marathi) | `\u0900–\u097F` |
| Bengali | `\u0980–\u09FF` |
| Gurmukhi (Punjabi) | `\u0A00–\u0A7F` |
| Gujarati | `\u0A80–\u0AFF` |
| Tamil | `\u0B80–\u0BFF` |
| Telugu | `\u0C00–\u0C7F` |
| Kannada | `\u0C80–\u0CFF` |
| Malayalam | `\u0D00–\u0D7F` |

Combined regex to detect any Indic script:
```python
import re
HAS_INDIC = re.compile(r'[\u0900-\u0D7F]')

def has_indic_chars(text):
    return bool(HAS_INDIC.search(text)) if text else False
```

---

## 🔬 4. CRITICAL EMPIRICAL DISCOVERIES

### Discovery 1: Zero Cross-Country Matches
**0.000%** of matches in training data cross country boundaries.
- India entities ONLY match India records. US ONLY matches US. France ONLY matches France.
- **MANDATORY:** Filter by country first.

### Discovery 2: Address as a Bridge
When business names are in completely different scripts (English vs Tamil), the **address often shares identical street numbers, PIN codes, and city names in ASCII**. Use address overlap as a confidence booster:
- S1 address: `6(29), C.I.T. Colony, Mylapore, Chennai`
- S2 address: `6(29), C.I.T. COLONY, CHENNAI`
- Shared tokens: `6(29)`, `colony`, `chennai` → High confidence bridge!

### Country Distribution in Test Set:
| Country | S1 Entities | Percentage |
|---|---|---|
| India | 809,986 | 46.8% |
| US | 663,106 | 38.3% |
| France | 259,452 | 15.0% |

---

## 🛠️ 5. YOUR EXACT TECHNICAL DELIVERABLE

Create a standalone Python script: **`src/multilingual_matcher.py`**

### Recommended Approach: Multilingual Dense Embeddings + Address Bridging

```python
"""
Multilingual Entity Matcher for Indian Indic-Script Business Names.
Detects cross-lingual matches (English ↔ Tamil/Hindi/Kannada) using
multilingual sentence embeddings + address overlap bridging.
"""
import csv
import re
import sys
import os
import time
from collections import defaultdict

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COSINE_THRESHOLD = 0.85      # High threshold for F_0.5 precision safety
BATCH_SIZE = 128              # GPU batch size for encoding
MAX_CANDIDATES_PER_ENTITY = 15

HAS_INDIC = re.compile(r'[\u0900-\u0D7F]')

def has_indic_chars(text):
    return bool(HAS_INDIC.search(text)) if text else False

def extract_address_tokens(addr):
    """Extract normalized tokens from address for overlap comparison."""
    if not addr: return set()
    return set(re.findall(r'\w+', addr.lower()))

def main():
    start = time.time()
    
    # Step 1: Load only India records from S1 that might need cross-lingual matching
    print("[*] Loading India S1 entities...")
    s1_india = []
    with open("student_resource/dataset/test/test_source1.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[3].strip() == "India":
                s1_india.append(row)
    print(f"[+] Loaded {len(s1_india):,} India S1 entities")
    
    # Step 2: Load India records from S2/S3 that contain Indic characters
    print("[*] Loading India S2/S3 entities with Indic text...")
    indic_candidates = []
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        with open(f"student_resource/dataset/test/{src}", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if row[3].strip() == "India" and has_indic_chars(row[1]):
                    indic_candidates.append(row)
    print(f"[+] Loaded {len(indic_candidates):,} Indic-text candidates from S2/S3")
    
    # Step 3: Encode business names using multilingual model
    print(f"[*] Loading embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("[*] Encoding S1 India names...")
    s1_names = [row[1] for row in s1_india]
    s1_embeddings = model.encode(s1_names, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    
    print("[*] Encoding Indic candidate names...")
    cand_names = [row[1] for row in indic_candidates]
    cand_embeddings = model.encode(cand_names, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    
    # Step 4: Compute cosine similarities and find matches
    import numpy as np
    print("[*] Computing similarities (this may take a while)...")
    
    matches = defaultdict(list)
    
    # Process in chunks to avoid OOM
    CHUNK = 1000
    for i in range(0, len(s1_india), CHUNK):
        chunk_embs = s1_embeddings[i:i+CHUNK]
        sims = np.dot(chunk_embs, cand_embeddings.T)  # (CHUNK, num_candidates)
        
        for local_idx in range(len(chunk_embs)):
            global_idx = i + local_idx
            s1_id = s1_india[global_idx][0]
            s1_addr_tokens = extract_address_tokens(s1_india[global_idx][2])
            
            top_indices = np.where(sims[local_idx] >= COSINE_THRESHOLD)[0]
            
            for cand_idx in top_indices:
                cand_row = indic_candidates[cand_idx]
                cand_addr_tokens = extract_address_tokens(cand_row[2])
                
                # Extra confidence: check address token overlap
                addr_overlap = len(s1_addr_tokens & cand_addr_tokens) if s1_addr_tokens and cand_addr_tokens else 0
                
                if sims[local_idx][cand_idx] >= 0.90 or (sims[local_idx][cand_idx] >= COSINE_THRESHOLD and addr_overlap >= 2):
                    matches[s1_id].append(cand_row[0])
        
        if (i + CHUNK) % 10000 == 0:
            print(f"    Processed {i+CHUNK:,} / {len(s1_india):,} S1 entities...")
    
    # Step 5: Write output
    output_path = "output/multilingual_matches.tsv"
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, match_ids in matches.items():
            unique_ids = list(dict.fromkeys(match_ids))[:MAX_CANDIDATES_PER_ENTITY]
            f.write(f"{s1_id}\t{','.join(unique_ids)}\n")
    
    elapsed = time.time() - start
    print(f"\n[+] DONE in {elapsed:.1f}s!")
    print(f"[+] Found multilingual matches for {len(matches):,} S1 entities")
    print(f"[+] Output: {output_path}")

if __name__ == "__main__":
    main()
```

### Required Packages:
```bash
pip install sentence-transformers torch numpy
```

### Performance Notes:
- The model `paraphrase-multilingual-MiniLM-L12-v2` is only ~120MB and runs fast on GPU.
- On a GPU machine, encoding 800K names takes ~5–10 minutes.
- On CPU-only, it will take ~30–60 minutes. Still feasible within the hackathon.
- The dot-product similarity computation for 800K × N_candidates can be memory-intensive. Process in chunks of 1000 as shown above.

---

## 📤 6. EXACT OUTPUT FORMAT

### File: `output/multilingual_matches.tsv`
- Tab-separated, UTF-8 encoded.
- Header: `source1_entity_id\tmatched_entity_ids`
- Only include S1 entities where you found cross-lingual matches (not every S1 entity).
- Mukul will merge these matches with the main pipeline's output.

Example:
```
source1_entity_id	matched_entity_ids
S1-55344266	S2-249013014,S3-478195123,S3-384364074
S1-12345678	S2-98765432
```

---

## 🌿 7. COMPLETE GIT SETUP & WORKFLOW (START FROM SCRATCH)

### Step 0: Prerequisites
Make sure you have Git installed. Check with:
```bash
git --version
```
If not installed, download from https://git-scm.com/downloads

You also need Python 3.8+ and pip. **GPU recommended** (CUDA-capable) for faster embedding computation.

### Step 1: Clone Mukul's Repository
```bash
# Open terminal / command prompt / PowerShell
# Navigate to where you want to keep the project
cd D:\  # or wherever you want

# Clone the repo (this downloads all code, NOT the dataset)
git clone https://github.com/mukuld1511-bit/AWS-Hackathon.git

# Enter the project folder
cd AWS-Hackathon
```

After cloning, you'll see this structure:
```
AWS-Hackathon/
├── src/
│   ├── baseline.py               ← Mukul's code (DO NOT EDIT)
│   └── submission_checker.py     ← Mukul's code (DO NOT EDIT)
├── prompts/
│   ├── HARSH_TASK_PROMPT.md      ← Harsh's instructions
│   └── PRATEEK_TASK_PROMPT.md    ← This file (your instructions)
├── requirements.txt
├── TEAM_INSTRUCTIONS.md
├── PROJECT_KNOWLEDGE_BASE.md
├── .gitignore
└── output/                       ← Your output goes here
```

⚠️ **You will NOT see `student_resource/` folder** — the dataset is too large for Git (2.5 GB) and is excluded via `.gitignore`. You must download it separately (Step 3).

### Step 2: Switch to YOUR Branch
```bash
# Your branch already exists on the remote. Switch to it:
git checkout prateek

# Verify you're on the correct branch:
git branch
# Output should show: * prateek
```

⚠️ **NEVER work on `main`.** If `git branch` shows `* main`, you're on the wrong branch. Run `git checkout prateek` first.

### Step 3: Download the Dataset
The dataset is NOT in Git. Download it manually:

**Option A — Direct browser download:**
1. Open this URL in your browser: https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip
2. Save the file (1.09 GB)
3. Extract the zip into the `AWS-Hackathon/` folder

**Option B — Command line (PowerShell/Linux):**
```bash
# PowerShell (Windows):
Invoke-WebRequest -Uri "https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip" -OutFile "student_resource.zip"
Expand-Archive -Path "student_resource.zip" -DestinationPath "."
Remove-Item "student_resource.zip"

# Linux/Mac:
wget "https://cdn.unstop.com/files/6ab10eb3b23ba_student_resource.zip"
unzip student_resource.zip
rm student_resource.zip
```

After extraction, verify you have:
```bash
ls student_resource/dataset/test/
# Should show: test_source1.tsv  test_source2.tsv  test_source3.tsv
```

### Step 4: Install Python dependencies
```bash
pip install -r requirements.txt
pip install sentence-transformers torch numpy
```

### Step 5: Create your script and run it
Create your file at `src/multilingual_matcher.py` (see Section 5 for the full code).
```bash
python src/multilingual_matcher.py
```

### Step 6: Commit and Push YOUR CODE ONLY
```bash
# Check what files changed
git status

# Add ONLY your Python script
git add src/multilingual_matcher.py

# Commit with a descriptive message
git commit -m "Prateek: Multilingual Indic transliteration matcher with semantic embeddings"

# Push to YOUR branch on GitHub
git push origin prateek
```

### Step 7: Notify Mukul
Send a message to Mukul saying: "Branch `prateek` is ready for merge. I've pushed `src/multilingual_matcher.py`."
Mukul will pull your branch, run your code, fix any issues, and merge it into the main pipeline.

### ❌ THINGS YOU MUST NEVER DO:
- `git add student_resource/` or `git add output/` — too large for Git, will break the repo
- `git push origin main` — only Mukul pushes to main
- Edit or delete any file you didn't create — especially `src/baseline.py`
- Use external APIs, geocoding services, or web scraping (instant disqualification)
- Use models larger than 8 Billion parameters or non MIT/Apache-2.0 licensed models

---

## ✅ 8. COMPETITION CONSTRAINTS REMINDER
1. **No external data lookup:** No Google Maps, no government databases, no geocoding APIs, no web scraping.
2. **Model size limit:** Up to 8 Billion parameters, MIT or Apache 2.0 license only.
3. **`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`** is ~33M parameters, Apache 2.0 license. ✅ Fully compliant.
