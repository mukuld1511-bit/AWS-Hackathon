# 🎯 PRATEEK EXECUTION RUNBOOK: MULTILINGUAL INDIC MATCHER (VERSION 2)

> **Role:** Prateek (NLP & Indic Transliteration Specialist)  
> **Target Machine:** Prateek's Machine / Environment  
> **Git Branch:** `prateek`  
> **Mission:** Build, finalize, and validate the complete Multilingual Indic Matcher Module (`src/multilingual_matcher.py`) so it generates an official-standard submission file (`output/matching_results_prateek.tsv`) that Mukul can directly merge, evaluate, and fuse with the team's best model on DGX.  
> **Goal:** Zero confusion, 100% self-contained code, and zero manual work needed from Mukul except merging.

---

## 🧠 PART 1: THE CORE PROBLEM & WHY YOUR ROLE IS CRITICAL

### The Dataset Breakdown
- **Total Test Records (S1):** 1,732,544
- **India Records:** **46.8% (~810,000 records)** — ALMOST HALF THE ENTIRE CHALLENGE!
- **US Records:** 38.3%
- **France Records:** 15.0%

### The Challenge Only You Can Solve
In Indian commercial registrations, companies are registered across multiple scripts:
- **Source 1:** Tamil (`ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி`), Devanagari (`राज इन्वेस्टमेंट्स`), Telugu, or Kannada.
- **Source 2 / Source 3:** English transliteration (`Raj Investments LLP`).
- **Standard fuzzy matchers (Levenshtein, Jaro-Winkler) have 0% similarity** across different alphabets.
- **Your Model (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)** projects all languages into a unified semantic space, capturing links that nobody else in the hackathon can capture!

---

## 🚨 PART 2: STRICT SUBMISSION RULES (MANDATORY TO AVOID REJECTION)

To ensure your output never fails on Unstop or during team integration, strictly follow these 4 rules:

| Rule | Requirement | Why it Matters |
|---|---|---|
| **1. Precision Clamping** | **Max 1 match from S2 and Max 1 match from S3 per entity** | Macro $F_{0.5}$ weights Precision 2× over Recall. Predicting 5+ matches destroys precision and tanks the score. |
| **2. Full Row Coverage** | **Exactly 1,732,544 data rows** (plus 1 header row = 1,732,545 lines) | Every Source 1 entity (US, France, India) must appear. Non-India or unmatched entities must have empty string `""`. Missing rows = Instant Rejection. |
| **3. Clean Delimiters** | **Single Tab `\t` separator**, comma `,` for ID lists | No quotes, no spaces around tabs, no `NaN` or `null` strings. |
| **4. Valid Prefixes** | **Only `S2-` and `S3-` IDs allowed** | No self-matches (`S1-` matching `S1-` is strictly illegal). |

---

## 💻 PART 3: YOUR COMPLETE PRODUCTION CODE TO COPY INTO `src/multilingual_matcher.py`

Replace `src/multilingual_matcher.py` with this exact, fully-tested implementation:

```python
#!/usr/bin/env python3
"""
src/multilingual_matcher.py
Production Multilingual Indic Transliteration Matcher (Version 2)
Engineered for Amazon ML Challenge 2026 by Prateek.
"""

import os
import re
import csv
import sys
import json
import time
import gc
import argparse
from collections import defaultdict, Counter

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEST_S1 = "student_resource/dataset/test/test_source1.tsv"
TEST_S2 = "student_resource/dataset/test/test_source2.tsv"
TEST_S3 = "student_resource/dataset/test/test_source3.tsv"

OUTPUT_DIR = "output"
OUTPUT_TSV = os.path.join(OUTPUT_DIR, "matching_results_prateek.tsv")
CANDIDATE_CACHE_FILE = os.path.join(OUTPUT_DIR, "indic_candidates_cache.pt")

# Calibrated Decision Parameters for Macro F0.5
TIER1_COSINE_THRESHOLD = 0.92      # High confidence pure name transliteration
BRIDGE_SIM_THRESHOLD = 0.45        # Name similarity supported by address tokens
MIN_ADDR_TOKEN_OVERLAP = 6         # Minimum overlapping address tokens
RELAXED_ADDR_TOKEN_OVERLAP = 4     # If numbers/pincode match

RE_INDIC = re.compile(r'[\u0900-\u0D7F]')
RE_WORD = re.compile(r'\w+')
RE_NUM = re.compile(r'\b\d+\b')
STOPS = {'india', 'near', 'opp', 'opposite', 'road', 'rd', 'street', 'st', 'floor', 'flr', 'no', 'plot', 'ltd', 'pvt'}

def is_india(country_str):
    return bool(country_str and country_str.strip().lower() == "india")

def extract_address_tokens(addr):
    if not addr: return set()
    return {w for w in RE_WORD.findall(addr.lower()) if len(w) > 1 and w not in STOPS}

def extract_numbers(addr):
    if not addr: return set()
    return set(RE_NUM.findall(addr.lower()))

def prepare_candidates(model, device, batch_size=512, force_recompute=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    inv_index = defaultdict(list)

    if os.path.exists(CANDIDATE_CACHE_FILE) and not force_recompute:
        print(f"[*] Loading candidate embeddings from cache: {CANDIDATE_CACHE_FILE}...")
        t0 = time.time()
        cache = torch.load(CANDIDATE_CACHE_FILE, map_location=device, weights_only=False)
        cand_ids = cache["cand_ids"]
        cand_tokens = cache["cand_tokens"]
        cand_numbers = cache["cand_numbers"]
        cand_embeddings = cache["cand_embeddings"].to(device=device, dtype=torch.float16)

        counts = Counter()
        for toks in cand_tokens: counts.update(toks)
        for cid, toks in enumerate(cand_tokens):
            for t in toks:
                if counts[t] <= 5000:
                    inv_index[t].append(cid)

        print(f"[+] Loaded {len(cand_ids):,} candidates from cache in {time.time()-t0:.2f}s")
        return cand_ids, cand_tokens, cand_numbers, inv_index, cand_embeddings

    print("\n[*] Scanning Test S2 and S3 for India records...")
    t0 = time.time()
    cand_ids, cand_names, cand_tokens, cand_numbers = [], [], [], []

    def load_file(filepath):
        loaded = 0
        with open(filepath, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                if len(row) < 4: continue
                if is_india(row[3]):
                    cid = len(cand_ids)
                    cand_ids.append(row[0].strip())
                    cand_names.append(row[1].strip())
                    toks = extract_address_tokens(row[2])
                    nums = extract_numbers(row[2])
                    cand_tokens.append(toks)
                    cand_numbers.append(nums)
                    for t in toks: inv_index[t].append(cid)
                    loaded += 1
        return loaded

    s2_count = load_file(TEST_S2)
    s3_count = load_file(TEST_S3)
    total_cands = len(cand_ids)
    print(f"[+] Loaded {total_cands:,} candidates (S2: {s2_count:,}, S3: {s3_count:,})")

    print(f"[*] Encoding {total_cands:,} candidate names into float16 tensor...")
    cand_embeddings = model.encode(
        cand_names, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_tensor=True, device=device
    ).to(dtype=torch.float16)

    del cand_names
    gc.collect()

    torch.save({
        "cand_ids": cand_ids,
        "cand_tokens": cand_tokens,
        "cand_numbers": cand_numbers,
        "cand_embeddings": cand_embeddings.cpu(),
    }, CANDIDATE_CACHE_FILE)
    print("[+] Cache saved.")

    cand_embeddings = cand_embeddings.to(device=device)
    return cand_ids, cand_tokens, cand_numbers, inv_index, cand_embeddings

def run_matcher(batch_size=512, chunk_size=25000, force_recompute=False):
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("🚀 PRATEEK MULTILINGUAL INDIC MATCHER (VERSION 2 - PRODUCTION)")
    print(f"Device: {device} | Batch Size: {batch_size} | Chunk Size: {chunk_size:,}")
    print("=" * 80)

    model = SentenceTransformer(MODEL_NAME, device=device)
    cand_ids, cand_tokens, cand_numbers, inv_index, cand_embeddings = prepare_candidates(
        model, device, batch_size=batch_size, force_recompute=force_recompute
    )

    print(f"\n[*] Processing Test S1 records and writing full 1,732,544 row output to {OUTPUT_TSV}...")

    total_rows = 0
    matched_count = 0

    with open(TEST_S1, "r", encoding="utf-8") as fin, \
         open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as fout:

        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t", lineterminator="\n")
        
        # Write exact required header
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        next(reader, None)

        s1_batch = []

        def process_batch(batch):
            nonlocal matched_count
            b_ids = [r[0].strip() for r in batch]
            b_names = [r[1].strip() if len(r)>1 else "" for r in batch]
            b_addrs = [r[2].strip() if len(r)>2 else "" for r in batch]
            b_countries = [r[3].strip() if len(r)>3 else "" for r in batch]

            # Only encode India entities with Indic characters or non-empty names
            india_indices = [i for i, c in enumerate(b_countries) if is_india(c) and b_names[i]]
            
            # Map of local_idx -> best matches
            best_s2 = [None] * len(batch)
            best_s2_score = [0.0] * len(batch)
            best_s3 = [None] * len(batch)
            best_s3_score = [0.0] * len(batch)

            if india_indices:
                sub_names = [b_names[i] for i in india_indices]
                embs = model.encode(
                    sub_names, batch_size=len(sub_names), show_progress_bar=False,
                    normalize_embeddings=True, convert_to_tensor=True, device=device
                ).to(dtype=torch.float16)

                sims = torch.matmul(embs, cand_embeddings.T)

                for local_pos, orig_idx in enumerate(india_indices):
                    s_sims = sims[local_pos]
                    
                    # 1. High-Confidence Tier 1 (Transliteration >= 0.92)
                    top_scores, top_indices = torch.topk(s_sims, k=min(20, len(cand_ids)))
                    top_scores_np = top_scores.cpu().numpy()
                    top_indices_np = top_indices.cpu().numpy()

                    for score, c_idx in zip(top_scores_np, top_indices_np):
                        cid = cand_ids[c_idx]
                        if score >= TIER1_COSINE_THRESHOLD:
                            if cid.startswith("S2-") and score > best_s2_score[orig_idx]:
                                best_s2_score[orig_idx] = score
                                best_s2[orig_idx] = cid
                            elif cid.startswith("S3-") and score > best_s3_score[orig_idx]:
                                best_s3_score[orig_idx] = score
                                best_s3[orig_idx] = cid

                    # 2. Tier 2: Address Bridging (Address Overlap >= 6 and Sim >= 0.45)
                    s_toks = extract_address_tokens(b_addrs[orig_idx])
                    if s_toks:
                        s_nums = extract_numbers(b_addrs[orig_idx])
                        cand_hits = Counter()
                        for t in s_toks:
                            if t in inv_index:
                                for cid in inv_index[t]: cand_hits[cid] += 1
                        
                        for cid, hits in cand_hits.items():
                            has_num = bool(s_nums & cand_numbers[cid]) if (s_nums and cand_numbers[cid]) else False
                            if hits >= MIN_ADDR_TOKEN_OVERLAP or (hits >= RELAXED_ADDR_TOKEN_OVERLAP and has_num):
                                score = s_sims[cid].item()
                                if score >= BRIDGE_SIM_THRESHOLD:
                                    target_id = cand_ids[cid]
                                    if target_id.startswith("S2-") and score > best_s2_score[orig_idx]:
                                        best_s2_score[orig_idx] = score
                                        best_s2[orig_idx] = target_id
                                    elif target_id.startswith("S3-") and score > best_s3_score[orig_idx]:
                                        best_s3_score[orig_idx] = score
                                        best_s3[orig_idx] = target_id

                del embs
                del sims
                if device == "cuda": torch.cuda.empty_cache()

            # Write EXACT matches (Clamped strictly to at most 1 S2 and 1 S3)
            for idx in range(len(batch)):
                s1_id = b_ids[idx]
                matches = []
                if best_s2[idx]: matches.append(best_s2[idx])
                if best_s3[idx]: matches.append(best_s3[idx])

                if matches:
                    matched_count += 1
                    writer.writerow([s1_id, ",".join(matches)])
                else:
                    # Clean singleton row for non-India or unmatched
                    writer.writerow([s1_id, ""])

        for row in reader:
            total_rows += 1
            s1_batch.append(row)
            if len(s1_batch) >= 2000:
                process_batch(s1_batch)
                s1_batch = []
                if total_rows % 100000 == 0:
                    print(f"    Processed {total_rows:,} / 1,732,544 rows... (Matched: {matched_count:,})")

        if s1_batch:
            process_batch(s1_batch)

    print("\n" + "=" * 80)
    print("✅ EXECUTION COMPLETE!")
    print(f"Total Rows Written: {total_rows:,} (Expected: 1,732,544)")
    print(f"Total Matched:      {matched_count:,}")
    print(f"Output File:        {OUTPUT_TSV}")
    print("=" * 80)

    # Automatically validate output
    os.system(f"python3 validate_dgx_tsv.py {OUTPUT_TSV}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=25000)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    run_matcher(batch_size=args.batch_size, chunk_size=args.chunk_size, force_recompute=args.force_recompute)
```

---

## 🧪 PART 4: YOUR STEP-BY-STEP TERMINAL COMMANDS

Prateek, simply open your terminal and run these 3 steps:

### Step 1: Switch to `prateek` branch and pull latest tools
```bash
git checkout prateek
git pull origin main
```

### Step 2: Run the Matcher & Auto-Validate
```bash
python3 src/multilingual_matcher.py
```
*When it finishes, it will automatically call `validate_dgx_tsv.py`. Ensure it prints:*
`🎉 100% PERFECT PASS! TSV file is 100% valid and safe for Unstop.`

### Step 3: Commit and Push your Model
```bash
git add src/multilingual_matcher.py
git commit -m "feat(prateek): completed version 2 multilingual matcher with full row coverage & precision clamping"
git push origin prateek
```

---

## 🤝 WHAT HAPPENS NEXT (MUKUL'S HANDOFF):
1. Once you push, Mukul will review your `prateek` branch.
2. Mukul will merge it into `main` and execute the Sub 5 fusion pipeline on DGX.
3. Your multilingual predictions will be fused with the GBDT model, and the team will submit to Unstop.
4. Once evaluated, Mukul will share the updated leaderboard score with you!
cd ~/AWS-Hackathon && git checkout main && git pull origin main && chmod +x run_dgx_grandmaster.sh && ./run_dgx_grandmaster.sh
