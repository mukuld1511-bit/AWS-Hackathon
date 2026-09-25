#!/usr/bin/env python3
"""
src/multilingual_matcher.py

Standalone Multilingual Semantic & Indic Transliteration Matcher
Module for Amazon ML Challenge 2026 — Business Entity Resolution

Role: Prateek (Branch: prateek)
Target Output: output/multilingual_matches.tsv

Architecture & Safety Enhancements:
1. Environment & HF Cache: Configured to local cache to avoid permission errors.
2. Candidate Caching: Encodes 867k Indic candidates and caches to output/indic_candidates_cache.pt.
3. Inverted Index Address Blocking: Uses token inverted indexing to evaluate candidate address overlap in O(1) time.
4. Two-Tier Precision Decision Rule (Calibrated from Phase 2):
   - Tier 1: Direct High-Confidence Transliteration (Cosine >= 0.95)
   - Tier 2: Address Bridge (Address Overlap >= 7 or Overlap >= 5 with shared number, AND Cosine >= 0.45)
5. Dataset Chunking & Checkpointing:
   - S1 dataset divided into manageable chunks (default 25,000 records).
   - Results flushed directly to output/multilingual_matches.tsv at chunk boundaries.
   - Resumes seamlessly if interrupted without restarting from scratch.
   - Explicit memory garbage collection and torch.cuda.empty_cache() prevent system freezes.
"""

import os
import re
import csv
import sys
import json
import time
import gc
import argparse
from collections import defaultdict

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
OUTPUT_TSV = os.path.join(OUTPUT_DIR, "multilingual_matches.tsv")
CHECKPOINT_FILE = os.path.join(OUTPUT_DIR, "matcher_checkpoint.json")
CANDIDATE_CACHE_FILE = os.path.join(OUTPUT_DIR, "indic_candidates_cache.pt")

# Calibrated Decision Rule Parameters (from Phase 2 empirical validation)
# Tier 1 (Pure Cosine) is disabled as it produces 76% false positives.
HIGH_CONF_SIM_THRESHOLD = 1.01 # Disabled
BRIDGE_SIM_THRESHOLD = 0.50
MIN_ADDR_TOKEN_OVERLAP = 7
RELAXED_ADDR_TOKEN_OVERLAP = 999 # Disabled
MAX_MATCHES_PER_ENTITY = 15

# Indic script Unicode block: U+0900 (Devanagari) to U+0D7F (Malayalam)
RE_INDIC = re.compile(r'[\u0900-\u0D7F]')
RE_WORD = re.compile(r'\w+')
RE_NUM = re.compile(r'\b\d+\b')
STOPS = {'india', 'near', 'opp', 'opposite', 'road', 'rd', 'street', 'st', 'floor', 'flr', 'no', 'plot'}


def is_india(country_str):
    if not country_str:
        return False
    return country_str.strip().lower() == "india"


def extract_address_tokens(addr):
    if not addr:
        return set()
    return {w for w in RE_WORD.findall(addr.lower()) if len(w) > 1 and w not in STOPS}


def extract_numbers(addr):
    if not addr:
        return set()
    return set(RE_NUM.findall(addr.lower()))


def get_vram_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0.0


# ---------------------------------------------------------------------------
# Step 1 & 2: Candidate Loading, Indexing & Caching
# ---------------------------------------------------------------------------
def prepare_candidates(model, device, batch_size=512, force_recompute=False):
    """
    Loads Indic candidates from S2 and S3, computes embeddings (or loads cache),
    and constructs the inverted index for address bridging.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cand_ids = []
    cand_tokens = []
    cand_numbers = []
    inv_index = defaultdict(list)

    # Check if candidate cache exists
    if os.path.exists(CANDIDATE_CACHE_FILE) and not force_recompute:
        print(f"[*] Loading candidate embeddings from cache: {CANDIDATE_CACHE_FILE}...")
        t0 = time.time()
        cache_data = torch.load(CANDIDATE_CACHE_FILE, map_location=device, weights_only=False)
        cand_ids = cache_data["cand_ids"]
        cand_tokens = cache_data["cand_tokens"]
        cand_numbers = cache_data["cand_numbers"]
        cand_embeddings = cache_data["cand_embeddings"].to(device=device, dtype=torch.float16)

        # Count token frequencies to prune non-discriminative ubiquitous words (e.g. city/state names)
        from collections import Counter
        counts = Counter()
        for toks in cand_tokens:
            counts.update(toks)

        MAX_TOKEN_FREQ = 5000
        for cid, toks in enumerate(cand_tokens):
            for t in toks:
                if counts[t] <= MAX_TOKEN_FREQ:
                    inv_index[t].append(cid)

        print(f"[+] Loaded {len(cand_ids):,} candidates from cache in {time.time()-t0:.2f}s "
              f"(VRAM: {get_vram_mb():.1f} MB, Index tokens: {len(inv_index):,})")
        return cand_ids, cand_tokens, cand_numbers, inv_index, cand_embeddings

    # Otherwise stream and load candidate records
    print("\n[*] Scanning Test S2 and S3 for India records with Indic names...")
    t0 = time.time()
    cand_names = []

    def load_candidates_from_file(filepath):
        loaded = 0
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                if is_india(row[3]) and RE_INDIC.search(row[1]):
                    cid = len(cand_ids)
                    cand_ids.append(row[0].strip())
                    cand_names.append(row[1].strip())
                    addr_toks = extract_address_tokens(row[2])
                    addr_nums = extract_numbers(row[2])
                    cand_tokens.append(addr_toks)
                    cand_numbers.append(addr_nums)

                    for t in addr_toks:
                        inv_index[t].append(cid)
                    loaded += 1
        return loaded

    s2_cands = load_candidates_from_file(TEST_S2)
    s3_cands = load_candidates_from_file(TEST_S3)
    total_cands = len(cand_ids)
    print(f"[+] Loaded {total_cands:,} Indic candidates (S2: {s2_cands:,}, S3: {s3_cands:,}) in {time.time()-t0:.2f}s")
    print(f"[+] Address inverted index contains {len(inv_index):,} unique tokens")

    # Encode candidate names on GPU
    print(f"[*] Encoding {total_cands:,} candidate names into float16 tensor...")
    t1 = time.time()
    cand_embeddings = model.encode(
        cand_names,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_tensor=True,
        device=device,
    ).to(dtype=torch.float16)
    print(f"[+] Candidate names encoded in {time.time()-t1:.2f}s. Shape: {cand_embeddings.shape}")

    # Free candidate names string list from RAM
    del cand_names
    gc.collect()

    # Save to candidate cache
    print(f"[*] Saving candidate cache to {CANDIDATE_CACHE_FILE}...")
    torch.save({
        "cand_ids": cand_ids,
        "cand_tokens": cand_tokens,
        "cand_numbers": cand_numbers,
        "cand_embeddings": cand_embeddings.cpu(),
    }, CANDIDATE_CACHE_FILE)
    print(f"[+] Cache saved successfully.")

    # Ensure embeddings are on active compute device
    cand_embeddings = cand_embeddings.to(device=device)
    return cand_ids, cand_tokens, cand_numbers, inv_index, cand_embeddings


# ---------------------------------------------------------------------------
# Step 3: Chunked S1 Matching Engine
# ---------------------------------------------------------------------------
def run_matcher(
    limit_s1=None,
    chunk_size=25000,
    batch_size=512,
    force_recompute=False,
    resume=True
):
    start_time = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("MULTILINGUAL INDIC ENTITY MATCHER (Chunked & Memory-Safe)")
    print(f"Device:      {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"Model:       {MODEL_NAME}")
    print(f"Chunk Size:  {chunk_size:,} S1 records per chunk")
    print(f"Batch Size:  {batch_size} queries per GPU forward pass")
    print(f"Output File: {OUTPUT_TSV}")
    print("=" * 80)

    # Load Model
    print(f"\n[*] Initializing SentenceTransformer ({MODEL_NAME}) on {device}...")
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"[+] Model ready in {time.time()-t0:.2f}s")

    # Prepare Candidates
    cand_ids, cand_tokens, cand_numbers, inv_index, cand_embeddings = prepare_candidates(
        model, device, batch_size=batch_size, force_recompute=force_recompute
    )

    # Checkpoint recovery
    start_chunk = 0
    total_matched_s1 = 0
    total_matched_pairs = 0

    if resume and os.path.exists(CHECKPOINT_FILE) and os.path.exists(OUTPUT_TSV):
        try:
            with open(CHECKPOINT_FILE, "r") as f:
                ckpt = json.load(f)
                start_chunk = ckpt.get("completed_chunks", 0)
                total_matched_s1 = ckpt.get("total_matched_s1", 0)
                total_matched_pairs = ckpt.get("total_matched_pairs", 0)
                print(f"[+] Resuming from Checkpoint: Chunk #{start_chunk} already done. "
                      f"(Matched S1: {total_matched_s1:,}, Pairs: {total_matched_pairs:,})")
        except Exception as e:
            print(f"[!] Checkpoint read error: {e}. Starting fresh.")
            start_chunk = 0

    # Initialize output file if starting fresh
    if start_chunk == 0:
        with open(OUTPUT_TSV, "w", encoding="utf-8", newline="") as f:
            f.write("source1_entity_id\tmatched_entity_ids\n")

    # Read S1 in Chunks
    print(f"\n[*] Streaming India S1 anchors from {TEST_S1}...")
    current_chunk_idx = 0
    current_chunk_s1 = []
    global_s1_processed = 0

    def process_chunk(chunk_records, chunk_num):
        nonlocal total_matched_s1, total_matched_pairs
        chunk_t0 = time.time()
        chunk_len = len(chunk_records)
        chunk_matches = defaultdict(list)
        tier1_chunk_count = 0
        tier2_chunk_count = 0

        # Sub-batch processing for GPU memory safety
        for b_start in range(0, chunk_len, batch_size):
            b_end = min(b_start + batch_size, chunk_len)
            batch = chunk_records[b_start:b_end]
            b_ids = [r[0] for r in batch]
            b_names = [r[1] for r in batch]
            b_addrs = [r[2] for r in batch]

            # Encode S1 names in float16
            b_embs = model.encode(
                b_names,
                batch_size=len(b_names),
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_tensor=True,
                device=device,
            ).to(dtype=torch.float16)

            # GPU matrix multiplication: (len(batch), total_cands)
            sims = torch.matmul(b_embs, cand_embeddings.T)

            # --- TIER 1: High Confidence Transliteration (Cosine >= 0.95) ---
            high_mask = sims >= HIGH_CONF_SIM_THRESHOLD
            h_rows, h_cols = torch.nonzero(high_mask, as_tuple=True)
            if len(h_rows) > 0:
                h_rows_np = h_rows.cpu().numpy()
                h_cols_np = h_cols.cpu().numpy()
                for r_idx, c_idx in zip(h_rows_np, h_cols_np):
                    s1_id = b_ids[r_idx]
                    cand_id = cand_ids[c_idx]
                    chunk_matches[s1_id].append(cand_id)
                    tier1_chunk_count += 1

            # --- TIER 2: Address Bridge (Inverted Index + Cosine >= 0.45) ---
            candidate_pairs = []
            for local_idx, addr_str in enumerate(b_addrs):
                s_toks = extract_address_tokens(addr_str)
                if not s_toks:
                    continue
                s_nums = extract_numbers(addr_str)

                # Collect candidate hits from inverted index
                cand_hits = defaultdict(int)
                for t in s_toks:
                    if t in inv_index:
                        for cid in inv_index[t]:
                            cand_hits[cid] += 1

                # Check candidates meeting address overlap threshold
                for cid, hits in cand_hits.items():
                    # Rule: Overlap >= 7 OR (Overlap >= 5 with matching address number)
                    has_shared_num = bool(s_nums & cand_numbers[cid]) if (s_nums and cand_numbers[cid]) else False
                    if hits >= MIN_ADDR_TOKEN_OVERLAP or (hits >= RELAXED_ADDR_TOKEN_OVERLAP and has_shared_num):
                        candidate_pairs.append((local_idx, cid))

            # Vectorized GPU similarity check for all candidate pairs in batch
            if candidate_pairs:
                p_rows = torch.tensor([p[0] for p in candidate_pairs], device=device, dtype=torch.long)
                p_cols = torch.tensor([p[1] for p in candidate_pairs], device=device, dtype=torch.long)
                pair_sims = sims[p_rows, p_cols]
                valid_mask = pair_sims >= BRIDGE_SIM_THRESHOLD
                valid_indices = torch.nonzero(valid_mask).squeeze(1).cpu().numpy()
                for v_idx in valid_indices:
                    loc_r, c_idx = candidate_pairs[v_idx]
                    s1_id = b_ids[loc_r]
                    cand_id = cand_ids[c_idx]
                    chunk_matches[s1_id].append(cand_id)
                    tier2_chunk_count += 1

            del b_embs
            del sims

        # Write chunk results to disk immediately
        new_matched_entities = len(chunk_matches)
        new_matched_pairs = 0
        with open(OUTPUT_TSV, "a", encoding="utf-8", newline="") as f:
            for s1_id, match_list in chunk_matches.items():
                unique_matches = list(dict.fromkeys(match_list))[:MAX_MATCHES_PER_ENTITY]
                f.write(f"{s1_id}\t{','.join(unique_matches)}\n")
                new_matched_pairs += len(unique_matches)

        total_matched_s1 += new_matched_entities
        total_matched_pairs += new_matched_pairs

        # Save checkpoint
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({
                "completed_chunks": chunk_num + 1,
                "total_matched_s1": total_matched_s1,
                "total_matched_pairs": total_matched_pairs,
                "timestamp": time.time(),
            }, f, indent=2)

        # Clear PyTorch caching and run garbage collection
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        chunk_time = time.time() - chunk_t0
        speed = chunk_len / chunk_time if chunk_time > 0 else 0
        print(f"[Chunk #{chunk_num + 1:>2}] Processed {chunk_len:>6,} S1 rows in {chunk_time:>5.1f}s ({speed:>5.0f} rows/s) | "
              f"Tier1: {tier1_chunk_count:>4,} | Tier2: {tier2_chunk_count:>5,} | "
              f"Cumulative S1 Matched: {total_matched_s1:>6,} | VRAM: {get_vram_mb():.1f} MB")

    # Stream through S1 dataset
    with open(TEST_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if len(row) < 4:
                continue
            if is_india(row[3]):
                global_s1_processed += 1

                # If resuming, skip rows belonging to already completed chunks
                chunk_id_for_row = (global_s1_processed - 1) // chunk_size
                if chunk_id_for_row < start_chunk:
                    continue

                current_chunk_s1.append((row[0].strip(), row[1].strip(), row[2].strip() if row[2] else ""))

                if len(current_chunk_s1) >= chunk_size:
                    process_chunk(current_chunk_s1, current_chunk_idx if start_chunk == 0 else chunk_id_for_row)
                    current_chunk_idx += 1
                    current_chunk_s1.clear()

                if limit_s1 and global_s1_processed >= limit_s1:
                    break

    # Process final residual chunk
    if current_chunk_s1:
        chunk_id = (global_s1_processed - 1) // chunk_size
        process_chunk(current_chunk_s1, chunk_id)
        current_chunk_s1.clear()

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION COMPLETED")
    print("=" * 80)
    print(f"Total S1 Entities Processed:      {global_s1_processed:>10,}")
    print(f"Total S1 Entities with Matches:   {total_matched_s1:>10,}")
    print(f"Total Match Pairs Identified:     {total_matched_pairs:>10,}")
    print(f"Output File:                      {OUTPUT_TSV}")
    print(f"Total Execution Time:             {total_time:.1f}s ({total_time/60:.2f} mins)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Multilingual Indic Entity Matcher for Amazon ML Challenge 2026")
    parser.add_argument("--limit-s1", type=int, default=None, help="Limit number of S1 entities (for quick test runs)")
    parser.add_argument("--chunk-size", type=int, default=25000, help="Number of S1 records per chunk (default: 25,000)")
    parser.add_argument("--batch-size", type=int, default=512, help="GPU forward pass batch size (default: 512)")
    parser.add_argument("--force-recompute", action="store_true", help="Force recomputation of candidate embeddings cache")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume from checkpoint, restart fresh")
    args = parser.parse_args()

    run_matcher(
        limit_s1=args.limit_s1,
        chunk_size=args.chunk_size,
        batch_size=args.batch_size,
        force_recompute=args.force_recompute,
        resume=not args.no_resume
    )


if __name__ == "__main__":
    main()
