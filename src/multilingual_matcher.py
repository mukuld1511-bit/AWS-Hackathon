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

# Safeguard Hugging Face cache location to prevent root permission errors
if "HF_HOME" not in os.environ:
    local_cache = os.path.abspath(".venv/hf_cache")
    if os.path.exists(".venv"):
        os.environ["HF_HOME"] = local_cache

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

    print("\n[*] Scanning Test S2 and S3 for India records with Indic names...")
    t0 = time.time()
    cand_ids, cand_names, cand_tokens, cand_numbers = [], [], [], []

    def load_file(filepath):
        loaded = 0
        with open(filepath, "r", encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r, None)
            for row in r:
                if len(row) < 4: continue
                if is_india(row[3]) and RE_INDIC.search(row[1]):
                    cid = len(cand_ids)
                    cand_ids.append(row[0].strip())
                    cand_names.append(row[1].strip())
                    toks = extract_address_tokens(row[2])
                    nums = extract_numbers(row[2])
                    cand_tokens.append(toks)
                    cand_numbers.append(nums)
                    loaded += 1
        return loaded

    s2_count = load_file(TEST_S2)
    s3_count = load_file(TEST_S3)
    total_cands = len(cand_ids)
    print(f"[+] Loaded {total_cands:,} Indic candidates (S2: {s2_count:,}, S3: {s3_count:,}) in {time.time()-t0:.2f}s")

    counts = Counter()
    for toks in cand_tokens: counts.update(toks)
    for cid, toks in enumerate(cand_tokens):
        for t in toks:
            if counts[t] <= 5000:
                inv_index[t].append(cid)

    print(f"[*] Encoding {total_cands:,} candidate names into float16 tensor...")
    t1 = time.time()
    cand_embeddings = model.encode(
        cand_names, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_tensor=True, device=device
    ).to(dtype=torch.float16)
    print(f"[+] Candidate names encoded in {time.time()-t1:.2f}s. Shape: {cand_embeddings.shape}")

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

def run_matcher(batch_size=512, chunk_size=2000, force_recompute=False):
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

        t_loop = time.time()
        for row in reader:
            total_rows += 1
            s1_batch.append(row)
            if len(s1_batch) >= chunk_size:
                process_batch(s1_batch)
                s1_batch = []
                if total_rows % 100000 == 0:
                    speed = total_rows / (time.time() - t_loop)
                    print(f"    Processed {total_rows:>9,} / 1,732,544 rows... ({speed:>5.0f} rows/s, Matched: {matched_count:>7,})")

        if s1_batch:
            process_batch(s1_batch)

    print("\n" + "=" * 80)
    print("✅ EXECUTION COMPLETE!")
    print(f"Total Rows Written: {total_rows:,} (Expected: 1,732,544)")
    print(f"Total Matched:      {matched_count:,}")
    print(f"Output File:        {OUTPUT_TSV}")
    print(f"Execution Time:     {time.time()-start_time:.1f}s ({(time.time()-start_time)/60:.2f} mins)")
    print("=" * 80)

    # Automatically validate output using Python validator
    val_cmd = f'"{sys.executable}" validate_dgx_tsv.py "{OUTPUT_TSV}"'
    print(f"\n[*] Running validator: {val_cmd}")
    os.system(val_cmd)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    run_matcher(batch_size=args.batch_size, chunk_size=args.chunk_size, force_recompute=args.force_recompute)
