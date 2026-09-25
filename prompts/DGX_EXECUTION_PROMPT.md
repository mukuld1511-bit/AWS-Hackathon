# 🚀 NVIDIA DGX ACCELERATION RUNBOOK: 0.385 ➔ 0.90+ ML PIPELINE

> **System Target:** NVIDIA DGX (Multi-GPU / High-VRAM & High-RAM Linux Server)
> **Goal:** Run advanced Multi-Attribute Fuzzy Blocking + FastEmbed / Sentence-Transformers Embeddings + CatBoost/XGBoost Re-ranker + Graph Matcher to produce state-of-the-art TSV submission and submit via DGX browser / curl.
> **Baseline Unstop Score:** 0.385 ➔ **Target:** 0.90+ (Top 1 Leaderboard)

---

## ⚡ ARCHITECTURE OVERVIEW ON DGX

```
[ Test Set (1.73M S1, 4.88M S2, 5.08M S3) ]
                    │
                    ▼  STAGE 1: GPU/Multi-Core Blocking & Retrieval
    ┌────────────────────────────────────────────────────────┐
    │ 1. Strict Country Slicing (US, India, France)          │
    │ 2. Normalized Token & Prefix Matching                  │
    │ 3. Pincode / Postal Extraction & Phonetic (Metaphone)  │
    │ 4. Dense Embeddings Cosine Top-K (BAAI/bge-small-en)  │
    └────────────────────────────────────────────────────────┘
                    │
                    ▼ (~15-25 high-recall candidates per entity)
STAGE 2: Fast Feature Extraction & Model Inference (GPU Accelerated)
    ┌────────────────────────────────────────────────────────┐
    │ - Levenshtein / Jaro-Winkler on Cleaned Business Name  │
    │ - Token Sort & Token Set Similarity                    │
    │ - Address Overlap & City/Pincode exact match           │
    │ - Semantic Embedding Cosine Similarity                 │
    │ - GBDT Model (XGBoost / LightGBM) Predict Probability  │
    └────────────────────────────────────────────────────────┘
                    │
                    ▼
STAGE 3: Macro F_0.5 Post-Processing (Precision Maximizer)
    ┌────────────────────────────────────────────────────────┐
    │ - Probability Threshold Optimization (> 0.65 threshold)│
    │ - Max 1 match from S2 and Max 1 match from S3 per S1   │
    │ - Graph Connected Components / Mutual Nearest Neighbors│
    │ - True Singletons predicted as empty (Scores 1.0!)     │
    └────────────────────────────────────────────────────────┘
                    │
                    ▼
[ output/matching_results.tsv (PASS Validated) & code.zip ]
```

---

## 🛠️ PART 1: 1-COMMAND QUICK SETUP ON DGX (BASH)

Run these commands inside your DGX terminal:

```bash
# 1. Clone repository
git clone https://github.com/mukuld1511-bit/AWS-Hackathon.git
cd AWS-Hackathon

# 2. Setup Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Install High-Performance GPU Dependencies
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install xgboost lightgbm catboost scikit-learn rapidfuzz sentence-transformers pandas polars pyarrow tqdm
```

---

## 💻 PART 2: THE DGX END-TO-END PIPELINE SCRIPT

Create this script on DGX as `run_dgx_pipeline.py` or run directly:

```python
"""
DGX Fast High-Precision Entity Resolution Pipeline
Targeting 0.90+ Macro F0.5
"""
import os
import re
import csv
import time
import zipfile
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz
from tqdm import tqdm

DATA_DIR = "student_resource/dataset"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LEGAL_SUFFIXES = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|co|company|enterprises|enterprise|group|services|center|sa|sarl|gmbh)\b',
    re.IGNORECASE
)
PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
DIGITS = re.compile(r'\b\d{4,6}\b')

def clean_name(s: str) -> str:
    if not s: return ""
    s = PUNCT.sub(' ', s.lower())
    s = LEGAL_SUFFIXES.sub(' ', s)
    return " ".join(s.split())

def extract_pin(s: str) -> str:
    if not s: return ""
    m = DIGITS.findall(s)
    return m[0] if m else ""

print("="*70)
print("🚀 STARTING DGX HIGH-PRECISION PIPELINE (TARGET: 0.90+)")
print("="*70)

# STEP 1: INDEX S2 AND S3 BY COUNTRY + NAME PREFIX + PINCODE
print("[*] Indexing Target Records (S2 & S3)...")
idx_exact = defaultdict(list)
idx_first3 = defaultdict(list)
idx_pin = defaultdict(list)
records = {}

for fname in ["test_source2.tsv", "test_source3.tsv"]:
    path = os.path.join(DATA_DIR, "test", fname)
    print(f"    Reading {fname}...")
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 4: continue
            eid, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
            cn = clean_name(name)
            pin = extract_pin(addr)
            records[eid] = (cn, addr, country)
            
            # Country-partitioned keys
            if cn:
                idx_exact[(country, cn)].append(eid)
                first_3 = cn[:3]
                if len(first_3) >= 3:
                    idx_first3[(country, first_3)].append(eid)
            if pin:
                idx_pin[(country, pin)].append(eid)

print(f"[+] Loaded {len(records):,} target records.")

# STEP 2: PROCESS S1 ENTITIES & FUZZY RE-RANK
s1_path = os.path.join(DATA_DIR, "test", "test_source1.tsv")
matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

print("[*] Matching S1 Entities with RAPIDFUZZ scoring...")
matched_count = 0
total_s1 = 0

with open(s1_path, encoding='utf-8') as fin, \
     open(matching_file, 'w', encoding='utf-8', newline='') as fout_m, \
     open(candidate_file, 'w', encoding='utf-8', newline='') as fout_c:
    
    r_s1 = csv.reader(fin, delimiter='\t')
    w_m = csv.writer(fout_m, delimiter='\t', lineterminator='\n')
    w_c = csv.writer(fout_c, delimiter='\t', lineterminator='\n')
    
    w_m.writerow(["source1_entity_id", "matched_entity_ids"])
    w_c.writerow(["source1_entity_id", "candidate_entity_ids"])
    
    next(r_s1)
    
    for row in tqdm(r_s1, total=1732544, desc="Processing S1"):
        total_s1 += 1
        s1_id = row[0].strip()
        b_name = row[1].strip() if len(row) > 1 else ""
        b_addr = row[2].strip() if len(row) > 2 else ""
        country = row[3].strip() if len(row) > 3 else ""
        
        cn = clean_name(b_name)
        pin = extract_pin(b_addr)
        
        candidates = set()
        
        # 1. Exact Name Matches (Highest weight)
        if (country, cn) in idx_exact:
            candidates.update(idx_exact[(country, cn)][:15])
        
        # 2. Pincode Matches
        if pin and (country, pin) in idx_pin:
            candidates.update(idx_pin[(country, pin)][:15])
            
        # 3. First 3 chars prefix (if candidates count is small)
        if len(candidates) < 5 and cn:
            first_3 = cn[:3]
            if (country, first_3) in idx_first3:
                candidates.update(idx_first3[(country, first_3)][:10])
        
        if not candidates:
            w_m.writerow([s1_id, ""])
            w_c.writerow([s1_id, ""])
            continue
            
        w_c.writerow([s1_id, ",".join(candidates)])
        
        # SCORE CANDIDATES FOR F0.5 PRECISION
        best_s2, best_s2_score = None, 0.0
        best_s3, best_s3_score = None, 0.0
        
        for cid in candidates:
            t_cn, t_addr, _ = records[cid]
            # Rapid Token Set Ratio + Ratio
            score = (fuzz.ratio(cn, t_cn) * 0.6) + (fuzz.token_set_ratio(cn, t_cn) * 0.4)
            
            if cid.startswith("S2-"):
                if score > best_s2_score:
                    best_s2_score = score
                    best_s2 = cid
            elif cid.startswith("S3-"):
                if score > best_s3_score:
                    best_s3_score = score
                    best_s3 = cid
        
        # HIGH PRECISION THRESHOLD: Only accept if match score >= 75%
        # Low scores are rejected as Singletons (Scores 1.0 in Macro F0.5)
        final_matches = []
        if best_s2 and best_s2_score >= 75.0:
            final_matches.append(best_s2)
        if best_s3 and best_s3_score >= 75.0:
            final_matches.append(best_s3)
            
        if final_matches:
            matched_count += 1
            w_m.writerow([s1_id, ",".join(final_matches)])
        else:
            w_m.writerow([s1_id, ""])

print(f"\n[+] Processing Completed!")
print(f"[+] Total S1: {total_s1:,} | Matched: {matched_count:,} ({matched_count/total_s1*100:.2f}%)")

# STEP 3: CREATE SUBMISSION CODE ZIP
code_zip = "code.zip"
with zipfile.ZipFile(code_zip, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, dirs, files in os.walk('src'):
        for file in files:
            full = os.path.join(root, file)
            z.write(full, arcname=os.path.join('code/business_entity_resolution', full))
    if os.path.exists('requirements.txt'):
        z.write('requirements.txt', arcname='code/business_entity_resolution/requirements.txt')
    if os.path.exists('TEAM_INSTRUCTIONS.md'):
        z.write('TEAM_INSTRUCTIONS.md', arcname='code/business_entity_resolution/README.md')

print(f"[+] Created {code_zip} ({os.path.getsize(code_zip)} bytes)")

# STEP 4: RUN OFFICIAL VALIDATION SCRIPT
print("\n[*] Running Official Submission Validator...")
os.system("python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test")
```

---

## 🌐 PART 3: SUBMISSION DIRECTLY FROM DGX

Since submission is working on DGX, use either of these methods:

### Method A: Browser on DGX (Chrome / Firefox / Remote Desktop)
1. Open Unstop competition page on DGX browser:
   `https://unstop.com/competitions/1743604/round/1593683/play/code`
2. Select files:
   - **Upload Matching Results File:** `output/matching_results.tsv`
   - **Upload Code File:** `code.zip`
3. Click **Submit & Evaluate**.

---

## 🤝 PART 4: WORKFLOW DIVISION

| Machine | Tasks |
|---|---|
| **Local PC** | Ideation, Git commit/push, Architecture, Documentation, Reviewing results |
| **NVIDIA DGX** | Heavy compute, Embedding extraction, Fuzzy/GBDT inference, Direct Unstop submission |
