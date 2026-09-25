#!/usr/bin/env python3
"""
src/multilingual_matcher.py
Universal Multilingual & Cross-Country Entity Resolution Engine (Version 3 - Global Production)
Supports: ALL Countries (US, France, India) across ALL Sources (S1, S2, S3).
Combines:
  1. Exact Normalized Multi-Country Matching (Deterministic High-Precision)
  2. Postal / Zip / PIN Code & Address Inverted Index Blocking
  3. Deep Multilingual Semantic Embedding Re-Ranking (Indic, French, US Latin)
  4. Precision Clamping (Strict 1 S2, 1 S3 per S1) & Validation Integrity
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

# Ensure Hugging Face cache uses safe local directory
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
FINE_TUNED_MODEL = "output/fine_tuned_multilingual_entity_model"
DEFAULT_BASE_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

TEST_S1 = "student_resource/dataset/test/test_source1.tsv"
TEST_S2 = "student_resource/dataset/test/test_source2.tsv"
TEST_S3 = "student_resource/dataset/test/test_source3.tsv"

OUTPUT_DIR = "output"
OUTPUT_TSV = os.path.join(OUTPUT_DIR, "matching_results_prateek.tsv")
CANDIDATE_CACHE_FILE = os.path.join(OUTPUT_DIR, "global_candidates_cache.pt")

LEGAL_SUFFIXES_PATTERN = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|co|company|enterprises|enterprise|group|services|center|sarl|sas|eurl|sa|gmbh)\b',
    re.IGNORECASE
)
PUNCT_PATTERN = re.compile(r'[^\w\s]', re.UNICODE)
RE_INDIC = re.compile(r'[\u0900-\u0D7F]')
RE_WORD = re.compile(r'\w+')
RE_NUM = re.compile(r'\b\d+\b')
RE_ZIP = re.compile(r'\b\d{4,6}\b')
STOPS = {'india', 'usa', 'us', 'france', 'near', 'opp', 'opposite', 'road', 'rd', 'street', 'st', 'floor', 'flr', 'no', 'plot', 'ltd', 'pvt', 'rue', 'av', 'ave', 'boulevard', 'bd', 'blvd', 'cedex'}

def normalize_name(name: str) -> str:
    if not name:
        return ""
    text = PUNCT_PATTERN.sub(' ', name.lower())
    text = LEGAL_SUFFIXES_PATTERN.sub(' ', text)
    return " ".join(text.split())

def extract_address_tokens(addr: str):
    if not addr:
        return set()
    return {w for w in RE_WORD.findall(addr.lower()) if len(w) > 1 and w not in STOPS}

def extract_postal_code(addr: str):
    if not addr:
        return None
    codes = RE_ZIP.findall(addr)
    return codes[-1] if codes else None

def extract_numbers(addr: str):
    if not addr:
        return set()
    return set(RE_NUM.findall(addr.lower()))

# ---------------------------------------------------------------------------
# Global Candidate Indexing
# ---------------------------------------------------------------------------
def build_global_index(model, device, batch_size=512, force_recompute=False):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("🌍 BUILDING GLOBAL MULTI-COUNTRY CANDIDATE INDEX (US, FRANCE, INDIA)")
    print("=" * 70)
    
    exact_index = defaultdict(list)    # (country, norm_name) -> [cand_idx]
    postal_index = defaultdict(list)   # (country, postal_code) -> [cand_idx]
    token_index = defaultdict(list)    # token -> [cand_idx]
    
    cand_ids = []
    cand_names = []
    cand_addrs = []
    cand_countries = []
    cand_tokens_list = []
    cand_numbers_list = []

    t0 = time.time()
    for src_name, src_path in [("Source2", TEST_S2), ("Source3", TEST_S3)]:
        print(f"[*] Scanning {src_name} ({src_path})...")
        with open(src_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                eid, b_name, b_addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                cid = len(cand_ids)
                cand_ids.append(eid)
                cand_names.append(b_name)
                cand_addrs.append(b_addr)
                cand_countries.append(country)

                norm_n = normalize_name(b_name)
                if norm_n and country:
                    exact_index[(country.lower(), norm_n)].append(cid)

                post_code = extract_postal_code(b_addr)
                if post_code and country:
                    postal_index[(country.lower(), post_code)].append(cid)

                toks = extract_address_tokens(b_addr)
                cand_tokens_list.append(toks)
                cand_numbers_list.append(extract_numbers(b_addr))

    print(f"[+] Total Candidates Indexed across all countries: {len(cand_ids):,} in {time.time()-t0:.2f}s")
    print(f"[+] Exact Name Buckets: {len(exact_index):,}")
    print(f"[+] Postal Code Buckets: {len(postal_index):,}")

    # Build filtered token index for rare tokens to assist fuzzy matching
    print("[*] Building inverted token index...")
    counts = Counter()
    for toks in cand_tokens_list:
        counts.update(toks)
    
    for cid, toks in enumerate(cand_tokens_list):
        for t in toks:
            if 2 <= counts[t] <= 4000:
                token_index[t].append(cid)
    
    print(f"[+] Active Rare Inverted Tokens: {len(token_index):,}")

    return {
        "cand_ids": cand_ids,
        "cand_names": cand_names,
        "cand_addrs": cand_addrs,
        "cand_countries": cand_countries,
        "cand_tokens": cand_tokens_list,
        "cand_numbers": cand_numbers_list,
        "exact_index": exact_index,
        "postal_index": postal_index,
        "token_index": token_index
    }

# ---------------------------------------------------------------------------
# Global Multi-Country Matching Pipeline
# ---------------------------------------------------------------------------
def run_global_matching(batch_size=512, chunk_size=5000):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if fine-tuned model exists, else fallback to high-quality base multilingual model
    model_path = FINE_TUNED_MODEL if os.path.exists(FINE_TUNED_MODEL) else DEFAULT_BASE_MODEL
    print("\n" + "=" * 70)
    print(f"🚀 RUNNING GLOBAL MULTI-COUNTRY MATCHER ON DEVICE: {device}")
    print(f"[*] Model: {model_path}")
    print("=" * 70)

    model = SentenceTransformer(model_path, device=device)

    # 1. Build Global Candidate Index
    cand_data = build_global_index(model, device, batch_size=batch_size)
    cand_ids = cand_data["cand_ids"]
    cand_names = cand_data["cand_names"]
    cand_addrs = cand_data["cand_addrs"]
    cand_countries = cand_data["cand_countries"]
    cand_tokens = cand_data["cand_tokens"]
    cand_numbers = cand_data["cand_numbers"]
    exact_index = cand_data["exact_index"]
    postal_index = cand_data["postal_index"]
    token_index = cand_data["token_index"]

    # 2. Process Test S1 in Memory-Safe Streaming Chunks
    print(f"\n[*] Processing Test S1 records and writing output to {OUTPUT_TSV}...")
    t_start = time.time()
    total_processed = 0
    total_matched = 0

    with open(TEST_S1, "r", encoding="utf-8") as in_f, \
         open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as out_f:

        reader = csv.reader(in_f, delimiter="\t")
        next(reader, None)  # Skip header

        out_f.write("source1_entity_id\tmatched_entity_ids\n")

        chunk_rows = []
        for row in reader:
            chunk_rows.append(row)
            if len(chunk_rows) >= chunk_size:
                matched_in_chunk = process_chunk(
                    chunk_rows, out_f, model, device,
                    cand_ids, cand_names, cand_addrs, cand_countries,
                    cand_tokens, cand_numbers, exact_index, postal_index, token_index
                )
                total_processed += len(chunk_rows)
                total_matched += matched_in_chunk
                chunk_rows = []

                speed = total_processed / (time.time() - t_start)
                print(f"    Processed {total_processed:>9,} / 1,732,544 rows... ({speed:>5.0f} rows/s, Total Matched: {total_matched:>9,})")

        if chunk_rows:
            matched_in_chunk = process_chunk(
                chunk_rows, out_f, model, device,
                cand_ids, cand_names, cand_addrs, cand_countries,
                cand_tokens, cand_numbers, exact_index, postal_index, token_index
            )
            total_processed += len(chunk_rows)
            total_matched += matched_in_chunk

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"🎉 GLOBAL MULTI-COUNTRY MATCHING COMPLETE IN {elapsed/60:.2f} MINUTES!")
    print(f"[+] Total S1 Rows Processed: {total_processed:,}")
    print(f"[+] Total Matches Produced:  {total_matched:,} ({total_matched/total_processed*100:.2f}%)")
    print(f"[+] Output File:             {OUTPUT_TSV} ({os.path.getsize(OUTPUT_TSV)/(1024*1024):.2f} MB)")
    print("=" * 70)

def process_chunk(
    chunk_rows, out_f, model, device,
    cand_ids, cand_names, cand_addrs, cand_countries,
    cand_tokens, cand_numbers, exact_index, postal_index, token_index
):
    matched_count = 0

    # Step 1: High-Speed Exact & Postal Matching
    unresolved_queries = []  # (row_index, s1_id, name, addr, country, candidate_cids)
    results = [None] * len(chunk_rows)

    for i, row in enumerate(chunk_rows):
        if len(row) < 4:
            s1_id = row[0].strip() if row else ""
            results[i] = (s1_id, "")
            continue

        s1_id, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
        c_lower = country.lower()
        norm_n = normalize_name(name)
        post_code = extract_postal_code(addr)
        s1_toks = extract_address_tokens(addr)
        s1_nums = extract_numbers(addr)

        # Tier 1: Exact Name + Country Match
        exact_cids = exact_index.get((c_lower, norm_n), [])
        if exact_cids:
            # Group by S2 / S3
            best_s2, best_s3 = None, None
            for cid in exact_cids:
                cid_target = cand_ids[cid]
                is_s2 = cid_target.startswith(("S2-", "S2_"))
                is_s3 = cid_target.startswith(("S3-", "S3_"))

                # Disambiguate using address token and number overlap
                t_overlap = len(s1_toks.intersection(cand_tokens[cid]))
                num_overlap = len(s1_nums.intersection(cand_numbers[cid]))
                score = t_overlap * 2 + num_overlap * 3

                if is_s2 and (best_s2 is None or score > best_s2[1]):
                    best_s2 = (cid_target, score)
                elif is_s3 and (best_s3 is None or score > best_s3[1]):
                    best_s3 = (cid_target, score)

            matched_list = []
            if best_s2: matched_list.append(best_s2[0])
            if best_s3: matched_list.append(best_s3[0])

            if matched_list:
                results[i] = (s1_id, ",".join(matched_list))
                matched_count += 1
                continue

        # Collect candidate pool for semantic / fuzzy matching
        cand_pool = set()
        
        # Postal code candidates
        if post_code:
            for cid in postal_index.get((c_lower, post_code), [])[:15]:
                cand_pool.add(cid)

        # Rare address/name token candidates
        for tok in list(s1_toks)[:3]:
            for cid in token_index.get(tok, [])[:10]:
                if cand_countries[cid].lower() == c_lower:
                    cand_pool.add(cid)

        # Cap cand_pool to top 20 candidates
        if cand_pool and len(cand_pool) <= 20:
            unresolved_queries.append((i, s1_id, name, addr, country, list(cand_pool)))
        else:
            results[i] = (s1_id, "")

    # Step 2: Batch Multilingual Semantic Re-ranking on Unresolved Queries
    if unresolved_queries:
        # 1. Batch encode all queries in chunk
        query_texts = [f"{q[2]} | {q[3]}" if q[3] else q[2] for q in unresolved_queries]
        q_embeddings = model.encode(
            query_texts,
            batch_size=256,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True
        )

        # 2. Gather all unique candidate CIDs and encode in ONE batch pass
        unique_cids = list({cid for q in unresolved_queries for cid in q[5]})
        cid_to_pos = {cid: idx for idx, cid in enumerate(unique_cids)}
        cand_texts = [f"{cand_names[c]} | {cand_addrs[c]}" if cand_addrs[c] else cand_names[c] for c in unique_cids]
        
        c_embeddings_all = model.encode(
            cand_texts,
            batch_size=512,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=device,
            normalize_embeddings=True
        )

        # 3. Fast Vectorized Similarity Scoring
        for q_idx, (orig_i, s1_id, name, addr, country, cand_list) in enumerate(unresolved_queries):
            c_lower = country.lower()
            s1_toks = extract_address_tokens(addr)
            s1_nums = extract_numbers(addr)

            pos_indices = [cid_to_pos[c] for c in cand_list]
            sub_c_embs = c_embeddings_all[pos_indices]
            q_emb = q_embeddings[q_idx]
            sims = torch.mv(sub_c_embs, q_emb).cpu().numpy()

            best_s2, best_s3 = None, None
            for c_pos, cid in enumerate(cand_list):
                sim = float(sims[c_pos])
                cid_target = cand_ids[cid]
                is_s2 = cid_target.startswith(("S2-", "S2_"))
                is_s3 = cid_target.startswith(("S3-", "S3_"))

                t_overlap = len(s1_toks.intersection(cand_tokens[cid]))
                num_overlap = len(s1_nums.intersection(cand_numbers[cid]))

                # High confidence semantic match with minimum geographic / token overlap
                if (sim >= 0.84 and (t_overlap >= 1 or num_overlap >= 1)) or (sim >= 0.70 and t_overlap >= 3):
                    combined_score = sim + (0.05 * t_overlap) + (0.1 * num_overlap)
                    if is_s2 and (best_s2 is None or combined_score > best_s2[1]):
                        best_s2 = (cid_target, combined_score)
                    elif is_s3 and (best_s3 is None or combined_score > best_s3[1]):
                        best_s3 = (cid_target, combined_score)

            matched_list = []
            if best_s2: matched_list.append(best_s2[0])
            if best_s3: matched_list.append(best_s3[0])

            if matched_list:
                results[orig_i] = (s1_id, ",".join(matched_list))
                matched_count += 1
            else:
                results[orig_i] = (s1_id, "")

    # Step 3: Write Chunk Output
    for s1_id, match_str in results:
        out_f.write(f"{s1_id}\t{match_str}\n")

    return matched_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal Multi-Country Multilingual Matcher")
    parser.add_argument("--batch_size", type=int, default=512, help="Embedding batch size")
    parser.add_argument("--chunk_size", type=int, default=5000, help="Chunk size for streaming processing")
    args = parser.parse_args()

    run_global_matching(batch_size=args.batch_size, chunk_size=args.chunk_size)
