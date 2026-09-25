"""
run_bge_reranker_precision_shield.py
====================================
Phase E: Dedicated BGE-Reranker-v2-m3 Precision Shield (Zero Generative LLMs)
Targets:
  1. Eliminate the ~225,000 false positives caused by cross-state / geographic clashes.
  2. Protect true multi-state corporate entities (e.g. Hyderabad TS vs AP, Siège Social vs Succursale).
  3. Re-verify low-similarity pairs with deep cross-attention.
  4. Maximize Unstop Leaderboard Macro F0.5 score by driving Precision to near 100%.

Model: BAAI/bge-reranker-v2-m3 (568M Params, Apache 2.0, Native FP16 on GPU).
"""

import os
import sys
import csv
import time
import re
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from FlagEmbedding import FlagReranker

csv.field_size_limit(sys.maxsize)

IN_MATCHING = "output_fused/matching_results.tsv"
IN_CANDIDATE = "output_fused/candidate_pairs.tsv"
OUT_DIR = "output_shielded"
OUT_MATCHING = os.path.join(OUT_DIR, "matching_results.tsv")
OUT_CANDIDATE = os.path.join(OUT_DIR, "candidate_pairs.tsv")

DATA_TEST = "student_resource/dataset/test"

US_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY', 'DC'
}

INDIAN_STATES = {
    'MH': {'maharashtra', 'mumbai', 'pune', 'nagpur', 'thane', 'nashik', 'aurangabad', 'solapur', 'kolhapur', 'amravati', 'navi mumbai', 'महाराष्ट्र'},
    'DL': {'delhi', 'new delhi', 'south delhi', 'north delhi', 'west delhi', 'east delhi', 'delhi ncr', 'दिल्‍ली', 'नई दिल्ली'},
    'KA': {'karnataka', 'bangalore', 'bengaluru', 'mysore', 'mysuru', 'hubli', 'dharwad', 'mangalore', 'mangaluru', 'belgaum', 'belagavi', 'ಕರ್ನಾಟಕ'},
    'TN': {'tamil nadu', 'tamilnadu', 'chennai', 'coimbatore', 'madurai', 'tiruchirappalli', 'trichy', 'salem', 'tirunelveli', 'erode', 'தமிழ்நாடு'},
    'GJ': {'gujarat', 'ahmedabad', 'surat', 'vadodara', 'baroda', 'rajkot', 'bhavnagar', 'jamnagar', 'ગુજરાત'},
    'UP': {'uttar pradesh', 'noida', 'greater noida', 'lucknow', 'kanpur', 'ghaziabad', 'agra', 'varanasi', 'meerut', 'allahabad', 'prayagraj', 'bareilly', 'aligarh', 'moradabad', 'saharanpur', 'gorakhpur', 'firozabad', 'jhansi', 'muzaffarnagar', 'mathura', 'उत्तर प्रदेश'},
    'WB': {'west bengal', 'bengal', 'kolkata', 'howrah', 'durgapur', 'asansol', 'siliguri', 'পশ্চিমবঙ্গ'},
    'RJ': {'rajasthan', 'jaipur', 'jodhpur', 'kota', 'bikaner', 'ajmer', 'udaipur', 'bhilwara', 'राजस्थान'},
    'TS': {'telangana', 'hyderabad', 'secunderabad', 'warangal'},
    'AP': {'andhra pradesh', 'visakhapatnam', 'vizag', 'vijayawada', 'guntur', 'nellore', 'kakinada', 'tirupati'},
    'KL': {'kerala', 'kochi', 'cochin', 'thiruvananthapuram', 'trivandrum', 'kozhikode', 'calicut', 'kollam', 'கேரளா', 'കേരളം'},
    'MP': {'madhya pradesh', 'indore', 'bhopal', 'jabalpur', 'gwalior', 'ujjain'},
    'HR': {'haryana', 'gurgaon', 'gurugram', 'faridabad', 'panipat', 'ambala', 'rohtak', 'hisar', 'karnal', 'sonipat'},
    'PB': {'punjab', 'ludhiana', 'amritsar', 'jalandhar', 'patiala', 'bathinda', 'ਪੰਜਾਬ'},
    'BR': {'bihar', 'patna', 'gaya', 'bhagalpur', 'muzaffarpur', 'बिहार'},
    'OD': {'odisha', 'orissa', 'bhubaneswar', 'cuttack', 'rourkela', 'berhampur', 'ଓଡ଼ିଶା'}
}

def get_state(addr, country):
    if not addr: return ''
    if country == 'us':
        for t in reversed(addr.replace(',', ' ').split()):
            if t in US_STATES: return t
    elif country == 'india':
        a_low = addr.lower()
        for code, kws in INDIAN_STATES.items():
            for kw in kws:
                if kw in a_low: return code
    elif country == 'france':
        m = re.search(r'\b(0[1-9]|[1-8]\d|9[0-5]|97|98)\d{3}\b', addr)
        if m: return m.group(1)
    return ''

def clean_toks(text):
    return set(re.findall(r'\b[a-zA-Z0-9]{3,}\b', text.lower()))

def main():
    print("=" * 75)
    print("🚀 BGE-RERANKER-V2-M3 PRECISION SHIELD & GEOGRAPHIC AUDIT")
    print("=" * 75)
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Collect all target IDs matched in input matching results
    print("\n[1/6] Reading matches to audit from", IN_MATCHING)
    matches_by_s1 = {}
    targets_needed = set()
    total_pairs_initial = 0

    with open(IN_MATCHING, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            s1_id = row[0].strip()
            if len(row) > 1 and row[1].strip():
                c_list = [x.strip() for x in row[1].split(',') if x.strip()]
                matches_by_s1[s1_id] = c_list
                for cid in c_list:
                    targets_needed.add(cid)
                total_pairs_initial += len(c_list)
            else:
                matches_by_s1[s1_id] = []

    print(f"[+] Loaded {len(matches_by_s1):,} S1 entities ({len(matches_by_s1)-sum(1 for v in matches_by_s1.values() if not v):,} non-empty).")
    print(f"[+] Total Matched Pairs: {total_pairs_initial:,} across {len(targets_needed):,} unique target IDs.")

    # 2. Load Metadata for S1
    print("\n[2/6] Loading Test Source 1 metadata...")
    s1_meta = {}
    with open(os.path.join(DATA_TEST, "test_source1.tsv"), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            s1_id = row[0].strip()
            name = row[1].strip() if len(row) > 1 else ""
            addr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip().lower() if len(row) > 3 else ""
            st = get_state(addr, country)
            s1_meta[s1_id] = (name, addr, country, st)

    # 3. Load Metadata for Targets
    print("\n[3/6] Loading Target metadata (S2 and S3)...")
    target_meta = {}
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(DATA_TEST, fname)
        print(f"    Reading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                cid = row[0].strip()
                if cid in targets_needed:
                    name = row[1].strip() if len(row) > 1 else ""
                    addr = row[2].strip() if len(row) > 2 else ""
                    country = row[3].strip().lower() if len(row) > 3 else ""
                    st = get_state(addr, country)
                    target_meta[cid] = (name, addr, country, st)

    print(f"[+] Successfully mapped metadata for {len(target_meta):,} matched targets.")

    # 4. Identification of Suspect / Conflicting Pairs
    print("\n[4/6] Auditing pairs for Geographic State Conflicts & Name Divergence...")
    pairs_to_rerank = []
    pair_metadata = [] # (s1_id, cid, reason)
    safe_pairs = set()

    for s1_id, c_list in matches_by_s1.items():
        if not c_list or s1_id not in s1_meta: continue
        s1_name, s1_addr, s1_country, s1_st = s1_meta[s1_id]
        s1_toks = clean_toks(s1_name)

        for cid in c_list:
            if cid not in target_meta:
                safe_pairs.add((s1_id, cid))
                continue
            t_name, t_addr, t_country, t_st = target_meta[cid]
            t_toks = clean_toks(t_name)

            # Check Country mismatch
            if s1_country and t_country and s1_country != t_country:
                # Instant rejection, 0% cross-country
                continue

            # Check State / Dept Conflict
            is_state_conflict = (s1_st and t_st and s1_st != t_st)
            
            # Check Token overlap
            tok_overlap = len(s1_toks & t_toks)
            is_low_overlap = (tok_overlap == 0 and len(s1_toks) > 0 and len(t_toks) > 0)

            if is_state_conflict or is_low_overlap:
                reason = "state_conflict" if is_state_conflict else "low_overlap"
                pairs_to_rerank.append([f"{s1_name}, {s1_addr}", f"{t_name}, {t_addr}"])
                pair_metadata.append((s1_id, cid, reason))
            else:
                safe_pairs.add((s1_id, cid))

    print(f"[+] Total Safe Pairs (Retained directly): {len(safe_pairs):,}")
    print(f"[+] Total Suspect Pairs needing BGE-Reranker evaluation: {len(pairs_to_rerank):,}")

    # 5. Execute BGE-Reranker on Suspect Pairs
    rejections = set()
    acceptances = 0

    if pairs_to_rerank:
        print("\n[5/6] Loading BAAI/bge-reranker-v2-m3 in FP16 on GB10 GPU...")
        reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=True)
        
        BATCH_SIZE = 512
        num_batches = (len(pairs_to_rerank) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"    Running inference across {num_batches:,} batches (batch_size={BATCH_SIZE})...")

        pbar = tqdm(total=len(pairs_to_rerank), desc="BGE Re-ranking")
        for i in range(0, len(pairs_to_rerank), BATCH_SIZE):
            batch_pairs = pairs_to_rerank[i : i + BATCH_SIZE]
            batch_meta = pair_metadata[i : i + BATCH_SIZE]
            scores = reranker.compute_score(batch_pairs, batch_size=BATCH_SIZE)

            for (s1_id, cid, reason), s in zip(batch_meta, scores):
                # Decision threshold:
                # For state conflict: must score >= 0.0 (strictly positive evidence required)
                # For low token overlap: must score >= -1.0 (some semantic connection)
                threshold = 0.0 if reason == "state_conflict" else -1.0

                if s < threshold:
                    rejections.add((s1_id, cid))
                else:
                    acceptances += 1
                    safe_pairs.add((s1_id, cid))

            pbar.update(len(batch_pairs))
        pbar.close()

    print(f"\n[+] BGE-Reranker Decision Complete:")
    print(f"    ❌ False Positives REJECTED: {len(rejections):,}")
    print(f"    ✅ Legitimate Pairs ACCEPTED: {acceptances:,}")
    print(f"    🛡️ Final High-Precision Valid Pairs: {len(safe_pairs):,}")

    # 6. Build Shielded Outputs
    print("\n[6/6] Writing Compliant Shielded TSV Files...")
    shielded_matches = defaultdict(list)
    for s1_id, cid in safe_pairs:
        shielded_matches[s1_id].append(cid)

    # Read original candidates to preserve 100% candidate superset compliance
    original_cands = {}
    print(f"    Reading original candidate pairs from {IN_CANDIDATE}...")
    with open(IN_CANDIDATE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            original_cands[row[0].strip()] = row[1].strip()

    total_s1 = 0
    non_empty_s1 = 0
    with open(os.path.join(DATA_TEST, "test_source1.tsv"), 'r', encoding='utf-8') as fin, \
         open(OUT_MATCHING, 'w', encoding='utf-8', newline='') as fout_m, \
         open(OUT_CANDIDATE, 'w', encoding='utf-8', newline='') as fout_c:

        wm = csv.writer(fout_m, delimiter='\t', lineterminator='\n')
        wc = csv.writer(fout_c, delimiter='\t', lineterminator='\n')

        wm.writerow(["source1_entity_id", "matched_entity_ids"])
        wc.writerow(["source1_entity_id", "candidate_entity_ids"])

        reader = csv.reader(fin, delimiter='\t')
        next(reader, None)
        for row in reader:
            total_s1 += 1
            s1_id = row[0].strip()
            matches = shielded_matches.get(s1_id, [])
            if matches:
                non_empty_s1 += 1

            # Candidate pairs must be a superset of matches
            cands_str = original_cands.get(s1_id, "")
            cands_set = set(x.strip() for x in cands_str.split(',') if x.strip())
            cands_set.update(matches)
            
            wm.writerow([s1_id, ",".join(matches)])
            wc.writerow([s1_id, ",".join(cands_set)])

    print(f"[+] Wrote {total_s1:,} rows to {OUT_MATCHING}")
    print(f"[+] Total Matched Entities: {non_empty_s1:,} ({non_empty_s1/total_s1*100:.2f}%)")
    print(f"[+] Total Singletons: {total_s1-non_empty_s1:,} ({(total_s1-non_empty_s1)/total_s1*100:.2f}%)")

    # Run official validation
    print("\nRunning Official Submission Validator...")
    os.system(f"python3 student_resource/utils/validate_submission.py -m {OUT_MATCHING} -c {OUT_CANDIDATE} -t {DATA_TEST} --check-ids")

    print(f"\n🎉 PRECISION SHIELD COMPLETE in {time.time() - t0:.1f}s!")
    print(f"Submission Ready at: {OUT_MATCHING}")

if __name__ == "__main__":
    main()
