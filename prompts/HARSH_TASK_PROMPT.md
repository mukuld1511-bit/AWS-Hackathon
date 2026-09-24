# ⚡ TASK FOR HARSH — Antigravity / AI Assistant Prompt
## Module: High-Recall Multi-Key Candidate Generation & Blocking Engine

> **INSTRUCTION FOR HARSH'S AI ASSISTANT:**  
> You are an expert Machine Learning engineer working on the **Amazon ML Challenge 2026 (Business Entity Resolution)**.  
> You are collaborating in a 4-person team. **Mukul (Lead)** owns the `main` branch, the primary heavy GBDT re-ranking pipeline, and official portal submissions. Your role is to build a **standalone candidate generation module** that Mukul will merge into the main pipeline.  
> **Your dedicated Git branch:** `harsh`. You must NEVER push to `main`.

---

## 🏢 1. COMPLETE PROBLEM STATEMENT

### What is Entity Resolution?
In large-scale commercial platforms (like Amazon), business identity data arrives from multiple independent sources. Each source contributes partial, noisy fragments of information about the same real-world businesses. These fragments share **no common identifiers** (no shared primary key). The challenge of determining which records across sources refer to the same business is called **Entity Resolution (ER)**.

### The 3 Data Sources
- **Source 1 (S1):** The **deduplicated reference** source. Think of it as the "anchor" or "master" list.
- **Source 2 (S2):** A noisy external source with spelling errors, legal suffix variations, swapped address components, transliterated names.
- **Source 3 (S3):** Another noisy external source, independent from S2, with similar noise patterns plus domain-name-as-business-name entries.

### The Task
For **every** Source 1 entity in the test set, find **all** matching records from Source 2 and Source 3.
- A Source 1 entity may match **zero** records (called a "singleton"), **one** record, or **many** records from S2 and S3.
- Matches are **never** between two S1 records or between S2 and S3 directly. Always S1 ↔ S2 or S1 ↔ S3.

### Evaluation Metric: Macro-Averaged $F_{0.5}$
$$F_{0.5} = \frac{1.25 \times \text{Precision} \times \text{Recall}}{0.25 \times \text{Precision} + \text{Recall}}$$

- **Precision is weighted 2× more than Recall.** A false merge (saying two different businesses are the same) is punished much harder than missing a true match.
- **Singletons matter:** If an S1 entity truly has no matches, correctly predicting an empty list earns a perfect 1.0. Predicting even one wrong match on a singleton drops it to 0.0.
- **Why this matters for YOUR module:** Your blocking module determines the **recall ceiling** of the entire pipeline. If a true match is not in your candidate pool, no downstream model can ever recover it. But if your pool is too large or noisy, the downstream classifier will produce false positives that destroy Precision and tank the $F_{0.5}$ score.

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
│   │   └── validate_submission.py       (official format validator)
│   └── Documentation_template.md
├── src/
│   └── baseline.py                      (Mukul's existing baseline)
├── prompts/
├── output/                              (generated output files go here)
└── ...
```

### Step 3: Verify file format
**ALL files are Tab-Separated `.tsv`.** Read them with explicit tab delimiter:
```python
import csv
with open("student_resource/dataset/test/test_source1.tsv", encoding="utf-8") as f:
    reader = csv.reader(f, delimiter="\t")
    header = next(reader)  # ['entity_id', 'business_name', 'business_address', 'country']
```
⚠️ Using `pd.read_csv()` without `sep="\t"` will silently produce garbage (single column with the entire line as one string).

---

## 📂 2.5. EXACT FILE USAGE CONTRACT — READ THIS CAREFULLY

### ✅ FILES YOU MUST READ (Input):
| File | Purpose | When to use |
|---|---|---|
| `student_resource/dataset/test/test_source1.tsv` | **All S1 entities you must generate candidates for** | ALWAYS — iterate every row |
| `student_resource/dataset/test/test_source2.tsv` | **S2 candidate pool to index** | ALWAYS — build inverted index from this |
| `student_resource/dataset/test/test_source3.tsv` | **S3 candidate pool to index** | ALWAYS — build inverted index from this |
| `student_resource/dataset/train/train_ground_truth.tsv` | **Known matches for local validation** | OPTIONAL — only for testing your recall |
| `student_resource/dataset/train/train_source1.tsv` | Train S1 entities | OPTIONAL — for local validation |
| `student_resource/dataset/train/train_source2.tsv` | Train S2 entities | OPTIONAL — for local validation |
| `student_resource/dataset/train/train_source3.tsv` | Train S3 entities | OPTIONAL — for local validation |

### 📝 FILE YOU MUST WRITE (Output):
| File | Format | Description |
|---|---|---|
| `output/candidate_pairs.tsv` | TSV with header `source1_entity_id\tcandidate_entity_ids` | One row per S1 test entity (1,732,544 rows). Candidates are comma-separated S2/S3 IDs. |

### 🚫 FILES YOU MUST NEVER TOUCH / MODIFY:
- **Any file inside `src/baseline.py`** — Mukul's main pipeline, do not edit.
- **Any file inside `student_resource/`** — Dataset is read-only.
- **Any file on the `main` branch** — You work only on `harsh` branch.

### 🔄 HOW YOUR OUTPUT GETS USED (Merge Protocol):
> **You do NOT need to produce the final submission.**
> Your job is ONLY to produce `output/candidate_pairs.tsv`.
> 
> **Mukul will:**
> 1. Pull your `harsh` branch
> 2. Run your `src/candidate_blocking.py` to regenerate candidates
> 3. Fix any bugs or format issues in your code
> 4. Merge your candidate pool with the main pipeline's Stage 2 re-ranker
> 5. Produce the final `matching_results.tsv` and submit on Unstop
> 
> **So don't worry about:**
> - Producing `matching_results.tsv` (that's Mukul's job)
> - Scoring / evaluation (Mukul handles that)
> - Integrating with other team members' code (Mukul merges everything)
> 
> **Just focus on:** Generating the highest-recall, reasonable-size candidate pool.

---

## 📊 3. EXACT DATA SCHEMA & REAL EXAMPLES

### Columns in every source file:
| Column | Description | Example |
|---|---|---|
| `entity_id` | Unique ID with source prefix | `S1-925783039`, `S2-166376419`, `S3-202863386` |
| `business_name` | Business name (noisy) | `Raj Investments LLP`, `ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி` |
| `business_address` | Address (noisy, may be empty) | `1795 Westchester Drive, High Point, NC` |
| `country` | Country string label | `US`, `India`, `France` |

### Ground Truth format (`train_ground_truth.tsv`):
| Column | Description |
|---|---|
| `source1_entity_id` | S1 entity ID |
| `matched_entity_ids` | Comma-separated list of matching S2/S3 IDs (empty for singletons) |

### Real Matched Examples from Training Data:

**Example 1 (US — Name typos + Empty address):**
```
ANCHOR:  S1-965667 | "Maure Williams Colombier Inc" | "85 Wayne Avenue, Ticonderoga, NY" | US
MATCH 1: S2-681193310 | "Maure Wilblims Colombier Inc" | "" | US          ← Name typo, empty address
MATCH 2: S2-743505751 | "Maure Williams Colombier" | "" | US              ← Missing "Inc", empty address
MATCH 3: S3-775321672 | "Dréxkor" | "85 Wanye Avenue, Ticonderoga" | US   ← Completely different trade name! Address matches
MATCH 4: S3-11291185  | "maurewilliamscolombier.com" | "Wayne Ave" | US    ← Website domain as name
MATCH 5: S3-860443364 | "Maure Williams Inc Center" | "" | US             ← Partial name, empty address
```

**Example 2 (India — Multilingual transliteration):**
```
ANCHOR:  S1-55344266 | "Raj Investments LLP" | "6(29), C.I.T. Colony, Mylapore, Chennai, Tamil Nadu" | India
MATCH 1: S2-249013014 | "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி" | "6(29), C.I.T. COLONY, CHENNAI" | India  ← Full Tamil
MATCH 2: S2-197070651 | "Raj Investments LLP" | "6(29), C.I.T. COLONY, CHENNAI" | India                   ← Exact English
MATCH 3: S3-478195123 | "Raj Investments எல்எல்பி" | "6(29), C.i.t. Colony, Chennai, TN" | India          ← Hybrid English+Tamil
MATCH 4: S3-384364074 | "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி" | "C.i.t. Colony, Chennai, தமிழ்நாடு" | India ← Full Tamil name + Tamil state
```

**Example 3 (US — Address component reordering + typos):**
```
ANCHOR:  S1-343815751 | "Dahlia Power Reliable Scientific LLC" | "630 45th Terrace, Kansas City, MO" | US
MATCH 1: S2-790675320 | "Dahlia Power Reliable" | "KANSAS CITY, MO, 630 45ND TERRACE, null" | US   ← Missing "Scientific LLC", address swapped, "45ND" typo
MATCH 2: S2-479876582 | "Dahlia Power Reliable Scientific" | "45ND TERRACE, null, KANSAS CITY, MO" | US  ← Missing "LLC", address reordered
MATCH 3: S3-878454467 | "Dahlia Ponr Reliable Scientific LLC" | "Missouri, 630 45th Terrace, Kansas City" | US ← "Ponr" typo, state spelled out
```

### Country Distribution in Test Set:
| Country | S1 Entities | Percentage |
|---|---|---|
| India | 809,986 | 46.8% |
| US | 663,106 | 38.3% |
| France | 259,452 | 15.0% |
| **Total** | **1,732,544** | **100%** |

---

## 🔬 4. CRITICAL EMPIRICAL DISCOVERIES

### Discovery 1: Zero Cross-Country Matches
Analysis of millions of training ground-truth pairs confirms: **0.000% of matches cross country boundaries.**
- US entity → ONLY matches US records in S2/S3.
- India entity → ONLY matches India records in S2/S3.
- France entity → ONLY matches France records in S2/S3.
- **MANDATORY:** Country is your **first hard blocking partition**. This prunes 66% of the search space instantly.

### Discovery 2: Ground Truth Statistics
- Total S1 train entities: 2,206,821
- Singletons (0 matches): 123,247 (5.58%)
- Entities with matches: 2,083,574 (94.42%)
- Average matches per non-singleton: 3.67

### Discovery 3: Many S2/S3 Records Have Empty Addresses
A significant fraction of S2/S3 records have empty `business_address` fields. Your blocking must NOT rely exclusively on address — name-based keys are essential.

---

## 🛠️ 5. YOUR EXACT TECHNICAL DELIVERABLE

Create a standalone Python script: **`src/candidate_blocking.py`**

### Architecture: Multi-Key Inverted Index Blocking
A single blocking key (e.g., exact normalized name) misses entities that match only by address (like `Dréxkor` at `85 Wayne Avenue`) or by fuzzy name variants. You need **3 complementary blocking keys**:

#### Key A: Normalized Name Key
```python
import re
LEGAL_SUFFIXES = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|'
    r'co|company|enterprises|enterprise|group|services|service|center|'
    r'solutions|associates|consulting|consultants|technologies|tech)\b',
    re.IGNORECASE
)
PUNCT = re.compile(r'[^\w\s]', re.UNICODE)

def normalize_name(name):
    if not name: return ""
    text = PUNCT.sub(' ', name.lower())
    text = LEGAL_SUFFIXES.sub(' ', text)
    return " ".join(text.split())

# Blocking key: (country, normalize_name(business_name))
```

#### Key B: Name Prefix + Address Number Anchor
```python
def extract_numbers(text):
    """Extract all digit sequences from text."""
    if not text: return ""
    nums = re.findall(r'\d+', text)
    return "_".join(sorted(set(nums)))  # deterministic order

# Blocking key: (country, normalize_name(name)[:4], extract_numbers(address))
# Example: ("US", "dahl", "45_630") for "Dahlia Power" at "630 45th Terrace"
```

#### Key C: Address Street Anchor (for trade-name divergence)
```python
def address_anchor(address):
    """Extract first number + first alphabetic word from address."""
    if not address: return ""
    tokens = address.lower().split()
    nums = [t for t in tokens if t.isdigit()]
    words = [t for t in tokens if t.isalpha() and len(t) > 2]
    first_num = nums[0] if nums else ""
    first_word = words[0] if words else ""
    return f"{first_num}_{first_word}"

# Blocking key: (country, address_anchor(business_address))
# Example: ("US", "85_wayne") for "85 Wayne Avenue, Ticonderoga, NY"
```

### Candidate Pool Construction Logic:
```python
from collections import defaultdict

# Phase 1: Build inverted indices over S2 and S3
index_name = defaultdict(list)    # Key A
index_prefix = defaultdict(list)  # Key B
index_addr = defaultdict(list)    # Key C

# Stream S2 and S3 records (DO NOT load into pandas)
for source_file in ["test_source2.tsv", "test_source3.tsv"]:
    with open(f"student_resource/dataset/test/{source_file}", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)  # skip header
        for row in reader:
            eid, name, addr, country = row[0], row[1], row[2], row[3].strip()
            norm_n = normalize_name(name)
            
            if norm_n and country:
                index_name[(country, norm_n)].append(eid)
            
            prefix = norm_n[:4] if norm_n else ""
            addr_nums = extract_numbers(addr)
            if prefix and addr_nums:
                index_prefix[(country, prefix, addr_nums)].append(eid)
            
            addr_anch = address_anchor(addr)
            if addr_anch and "_" in addr_anch:
                index_addr[(country, addr_anch)].append(eid)

# Phase 2: For each S1 entity, collect candidates from all 3 indices
MAX_CANDIDATES = 25

with open("student_resource/dataset/test/test_source1.tsv", encoding="utf-8") as f_in, \
     open("output/candidate_pairs.tsv", "w", encoding="utf-8", newline="") as f_out:
    
    reader = csv.reader(f_in, delimiter="\t")
    next(reader)  # skip header
    f_out.write("source1_entity_id\tcandidate_entity_ids\n")
    
    for row in reader:
        s1_id, name, addr, country = row[0], row[1], row[2], row[3].strip()
        norm_n = normalize_name(name)
        candidates = set()
        
        # Key A lookup
        candidates.update(index_name.get((country, norm_n), []))
        
        # Key B lookup
        prefix = norm_n[:4] if norm_n else ""
        addr_nums = extract_numbers(addr)
        if prefix and addr_nums:
            candidates.update(index_prefix.get((country, prefix, addr_nums), []))
        
        # Key C lookup
        addr_anch = address_anchor(addr)
        if addr_anch and "_" in addr_anch:
            candidates.update(index_addr.get((country, addr_anch), []))
        
        # Cap candidates
        cand_list = list(candidates)[:MAX_CANDIDATES]
        cand_str = ",".join(cand_list)
        f_out.write(f"{s1_id}\t{cand_str}\n")
```

### Performance & Memory Rules:
1. **NEVER** do `pd.read_csv("test_source2.tsv")` — that loads 500MB into RAM as a DataFrame. Use `csv.reader` streaming.
2. **Cap candidates at 25 per entity.** More than 25 dramatically increases downstream inference time and false positive risk.
3. Print progress every 1,000,000 records so you can monitor execution.
4. Expected runtime: 2–4 minutes on a modern machine.

---

## 📤 6. EXACT OUTPUT FORMAT

### File: `output/candidate_pairs.tsv`
- **Delimiter:** Single tab character (`\t`) between columns.
- **Header row (mandatory):** `source1_entity_id\tcandidate_entity_ids`
- **One row per S1 entity.** Must contain **exactly 1,732,544 rows** (excluding header).
- **Candidate list:** Comma-separated S2/S3 entity IDs. No spaces, no quotes.
- **Singletons:** If no candidates found, leave empty after the tab: `S1-XXXXX\t\n`
- **No S1 IDs in candidate list.** Only S2- and S3- prefixed IDs.
- **No duplicate IDs** within a single candidate list.

Example output:
```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
S1-00004	S2-12345,S3-67890
```

---

## 🌿 7. COMPLETE GIT SETUP & WORKFLOW (START FROM SCRATCH)

### Step 0: Prerequisites
Make sure you have Git installed. Check with:
```bash
git --version
```
If not installed, download from https://git-scm.com/downloads

You also need Python 3.8+ and pip.

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
│   ├── HARSH_TASK_PROMPT.md      ← This file (your instructions)
│   └── PRATEEK_TASK_PROMPT.md    ← Prateek's instructions
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
git checkout harsh

# Verify you're on the correct branch:
git branch
# Output should show: * harsh
```

⚠️ **NEVER work on `main`.** If `git branch` shows `* main`, you're on the wrong branch. Run `git checkout harsh` first.

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
```

### Step 5: Create your script and run it
Create your file at `src/candidate_blocking.py` (see Section 5 for the full code).
```bash
python src/candidate_blocking.py
```

### Step 6: Commit and Push YOUR CODE ONLY
```bash
# Check what files changed
git status

# Add ONLY your Python script
git add src/candidate_blocking.py

# Commit with a descriptive message
git commit -m "Harsh: Multi-key inverted index candidate blocking engine"

# Push to YOUR branch on GitHub
git push origin harsh
```

### Step 7: Notify Mukul
Send a message to Mukul saying: "Branch `harsh` is ready for merge. I've pushed `src/candidate_blocking.py`."
Mukul will pull your branch, run your code, fix any issues, and merge it into the main pipeline.

### ❌ THINGS YOU MUST NEVER DO:
- `git add student_resource/` or `git add output/` — too large for Git (2.5 GB), will break the repo
- `git push origin main` — only Mukul pushes to main
- Edit or delete any file you didn't create — especially `src/baseline.py`
- Use any external API, geocoding service, or web scraping (instant disqualification)

---

## ✅ 8. VALIDATION BEFORE HANDOVER

After generating `output/candidate_pairs.tsv`, run the official competition validator:
```bash
python student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test
```
If you only have `candidate_pairs.tsv` and not `matching_results.tsv`, you can create a temporary matching file where candidates = matches (for validation purposes only):
```bash
copy output\candidate_pairs.tsv output\matching_results.tsv
```
The validator should print **PASS**. If it prints errors, fix them before pushing.
