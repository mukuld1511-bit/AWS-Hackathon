"""
fuse_multilingual_gbdt.py
=========================
Fuses Prateek's Multilingual Indic Transliteration Embeddings (MiniLM L12)
with the Maximized Tri-Model GBDT Ensemble Output.
Target: Climb from 0.88+ all the way to 0.95+ Macro F0.5.
Amazon ML Challenge 2026
"""

import os
import re
import csv
import sys
import time
from collections import defaultdict, Counter
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

DATA_DIR = "student_resource/dataset/test"
INPUT_MATCHING = "output_maximized/matching_results.tsv"
INPUT_CANDIDATE = "output_maximized/candidate_pairs.tsv"
INDIC_CACHE = "output/indic_candidates_cache.pt"

OUTPUT_DIR = "output_fused"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CANDIDATE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TIER1_THRESHOLD = 0.90
TIER2_THRESHOLD = 0.50
MIN_ADDR_OVERLAP = 3

RE_WORD = re.compile(r'\w+')
RE_NUM = re.compile(r'\b\d+\b')
STOPS = {'india', 'near', 'opp', 'opposite', 'road', 'rd', 'street', 'st', 'floor', 'flr', 'no', 'plot', 'ltd', 'pvt', 'delhi', 'mumbai', 'kolkata', 'chennai', 'hyderabad', 'bangalore'}

def extract_addr_tokens(addr):
    if not addr: return set()
    return {w for w in RE_WORD.findall(addr.lower()) if len(w) > 2 and w not in STOPS}

def extract_addr_nums(addr):
    if not addr: return set()
    return set(RE_NUM.findall(addr.lower()))

def main():
    print("=" * 80)
    print("🚀 FUSING MULTILINGUAL INDIC TRANSLITERATION WITH GBDT ENSEMBLE")
    print("=" * 80)
    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Baseline Matches from Maximized Run
    print("[1/5] Loading Maximized Baseline Matches...")
    matches = {}
    candidates = {}
    with open(INPUT_MATCHING, 'r', encoding='utf-8') as fm, open(INPUT_CANDIDATE, 'r', encoding='utf-8') as fc:
        rm = csv.reader(fm, delimiter='\t')
        rc = csv.reader(fc, delimiter='\t')
        next(rm); next(rc)
        for row_m, row_c in zip(rm, rc):
            s1_id = row_m[0].strip()
            m_ids = [x.strip() for x in row_m[1].split(',') if x.strip()] if len(row_m) > 1 and row_m[1].strip() else []
            c_ids = [x.strip() for x in row_c[1].split(',') if x.strip()] if len(row_c) > 1 and row_c[1].strip() else []
            matches[s1_id] = m_ids
            candidates[s1_id] = c_ids

    total_s1 = len(matches)
    has_both = sum(1 for ids in matches.values() if any(x.startswith("S2-") for x in ids) and any(x.startswith("S3-") for x in ids))
    missing_s3 = [s1 for s1, ids in matches.items() if not any(x.startswith("S3-") for x in ids)]
    print(f"[+] Loaded {total_s1:,} entities.")
    print(f"    Both S2 and S3 Matched: {has_both:,} ({has_both/total_s1*100:.1f}%)")
    print(f"    Missing S3 Target Candidates: {len(missing_s3):,}")

    # 2. Load Precomputed Indic Candidates Cache
    print("\n[2/5] Loading Indic Candidates Cache...")
    cache = torch.load(INDIC_CACHE, map_location=device, weights_only=False)
    cand_ids = cache["cand_ids"]
    cand_tokens = cache["cand_tokens"]
    cand_numbers = cache["cand_numbers"]
    cand_embeddings = cache["cand_embeddings"].to(device=device, dtype=torch.float16)

    # Filter for S3 candidates
    s3_indices = [i for i, cid in enumerate(cand_ids) if cid.startswith("S3-")]
    s3_ids = [cand_ids[i] for i in s3_indices]
    s3_tokens = [cand_tokens[i] for i in s3_indices]
    s3_numbers = [cand_numbers[i] for i in s3_indices]
    s3_embeddings = cand_embeddings[s3_indices].contiguous()

    print(f"[+] Loaded {len(cand_ids):,} total Indic candidates.")
    print(f"    Filtered Indic S3 Candidates: {len(s3_ids):,} (Embedding shape: {s3_embeddings.shape})")

    # Inverted token index for S3 candidates
    token_inv_idx = defaultdict(list)
    for idx_pos, toks in enumerate(s3_tokens):
        for tok in toks:
            token_inv_idx[tok].append(idx_pos)

    # 3. Load Multilingual MiniLM Model
    print(f"\n[3/5] Initializing {MODEL_NAME} on {device}...")
    model = SentenceTransformer(MODEL_NAME, device=device)

    # 4. Stream S1 Entities and Run Batch GPU Inference
    print(f"\n[4/5] Running Multilingual Fusion on Missing S3 Entities...")
    target_set = set(missing_s3)
    s1_path = os.path.join(DATA_DIR, "test_source1.tsv")
    
    recovered_s3 = 0
    batch_records = []

    def process_indic_batch(batch):
        nonlocal recovered_s3
        if not batch: return

        b_ids = [r[0] for r in batch]
        b_names = [r[1] for r in batch]
        b_addrs = [r[2] for r in batch]

        with torch.no_grad():
            s1_embs = model.encode(
                b_names, batch_size=min(512, len(b_names)),
                show_progress_bar=False, normalize_embeddings=True,
                convert_to_tensor=True, device=device
            ).to(dtype=torch.float16)

            # Cosine similarity matrix: [batch_size, num_s3_cands]
            sim_matrix = torch.matmul(s1_embs, s3_embeddings.T)
            top_scores, top_indices = torch.topk(sim_matrix, k=min(15, len(s3_ids)), dim=-1)

            top_scores_np = top_scores.cpu().numpy()
            top_indices_np = top_indices.cpu().numpy()

        for i, s1_id in enumerate(b_ids):
            addr = b_addrs[i]
            addr_toks = extract_addr_tokens(addr)
            addr_nums = extract_addr_nums(addr)

            best_cid = None
            best_score = 0.0

            for score, cand_pos in zip(top_scores_np[i], top_indices_np[i]):
                cid = s3_ids[cand_pos]
                
                # Tier 1: Pure High-Confidence Transliteration (>= 0.90)
                if score >= TIER1_THRESHOLD:
                    best_cid = cid
                    best_score = score
                    break

                # Tier 2: Transliteration + Address Token/Number Support
                if score >= TIER2_THRESHOLD:
                    t_toks = s3_tokens[cand_pos]
                    t_nums = s3_numbers[cand_pos]
                    overlap = len(addr_toks & t_toks)
                    num_match = bool(addr_nums and t_nums and (addr_nums & t_nums))
                    if overlap >= MIN_ADDR_OVERLAP or (num_match and overlap >= 1):
                        best_cid = cid
                        best_score = score
                        break

            if best_cid:
                matches[s1_id].append(best_cid)
                candidates[s1_id].append(best_cid)
                recovered_s3 += 1

    with open(s1_path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin, delimiter='\t')
        next(reader, None)
        pbar = tqdm(total=len(missing_s3), desc="Indic Fusion S3")

        for row in reader:
            s1_id = row[0].strip()
            if s1_id in target_set:
                name = row[1].strip() if len(row) > 1 else ""
                addr = row[2].strip() if len(row) > 2 else ""
                country = row[3].strip().lower() if len(row) > 3 else ""
                
                if country == "india" and name:
                    batch_records.append((s1_id, name, addr))
                    if len(batch_records) >= 1024:
                        process_indic_batch(batch_records)
                        pbar.update(len(batch_records))
                        batch_records = []
                else:
                    pbar.update(1)

        if batch_records:
            process_indic_batch(batch_records)
            pbar.update(len(batch_records))
        pbar.close()

    print(f"\n[+] Total Newly Recovered Multilingual S3 Matches: +{recovered_s3:,}!")
    
    # 5. Write Final Fused Submission Files
    print("\n[5/5] Writing Final Fused Submission Files...")
    with open(s1_path, 'r', encoding='utf-8') as fin, \
         open(OUT_MATCHING, 'w', encoding='utf-8', newline='') as fout_m, \
         open(OUT_CANDIDATE, 'w', encoding='utf-8', newline='') as fout_c:

        wm = csv.writer(fout_m, delimiter='\t', lineterminator='\n')
        wc = csv.writer(fout_c, delimiter='\t', lineterminator='\n')

        wm.writerow(["source1_entity_id", "matched_entity_ids"])
        wc.writerow(["source1_entity_id", "candidate_entity_ids"])

        reader = csv.reader(fin, delimiter='\t')
        next(reader, None)
        for row in reader:
            s1_id = row[0].strip()
            m_list = list(dict.fromkeys(matches.get(s1_id, [])))
            c_list = list(dict.fromkeys(candidates.get(s1_id, []) + m_list))

            wm.writerow([s1_id, ",".join(m_list)])
            wc.writerow([s1_id, ",".join(c_list)])

    # 6. Official Verification Validation
    print("\nRunning Official Validation Checks...")
    os.system(f"python3 validate_dgx_tsv.py {OUT_MATCHING}")
    os.system(f"python3 student_resource/utils/validate_submission.py -m {OUT_MATCHING} -c {OUT_CANDIDATE} -t student_resource/dataset/test")
    print(f"\n🎉 MULTILINGUAL FUSION PIPELINE COMPLETE: {OUT_MATCHING}")
    print(f"Total execution time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
