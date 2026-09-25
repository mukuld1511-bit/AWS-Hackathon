"""
Diagnostic Script: Analyze Missed True Matches in Baseline Blocking.

Inspects the 1,812,526 missed true matches from Stage 4 baseline evaluation
to identify the exact root cause patterns (name token overlap, address digit presence,
script differences, postal codes) that cause recall loss.
"""

import os
import sys
import csv
import time
import unicodedata
from collections import Counter, defaultdict

sys.path.insert(0, "/home/piet/Desktop/aws model 2")
from src.normalization import normalize_name, extract_address_numbers, clean_punctuation
from src.blocking_keys import generate_key_a, generate_key_b, generate_key_c
from src.blocking_index import BlockingIndex
from src.evaluate_candidate_recall import load_ground_truth

train_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
s1_path = os.path.join(train_dir, "train_source1.tsv")
s2_path = os.path.join(train_dir, "train_source2.tsv")
s3_path = os.path.join(train_dir, "train_source3.tsv")
gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

def is_latin(text: str) -> bool:
    """Check if string contains only ASCII/Latin characters."""
    try:
        text.encode('latin-1')
        return True
    except UnicodeEncodeError:
        return False

def extract_postal_code(address: str) -> str:
    """Extract candidate 5-6 digit postal code if present."""
    if not address:
        return ""
    tokens = clean_punctuation(address).split()
    for t in tokens:
        if t.isdigit() and len(t) in (5, 6):
            return t
    return ""

def run_failure_analysis(sample_limit: int = 50000):
    print("=" * 70)
    print(f"=== DIAGNOSTIC FAILURE ANALYSIS (Sampling up to {sample_limit:,} S1 records) ===")
    print("=" * 70)

    # 1. Load Ground Truth
    print(f"[*] Loading ground truth from {gt_path}...")
    gt_map = load_ground_truth(gt_path)

    # 2. Build Index over S2 & S3 (store raw records for target lookup)
    index = BlockingIndex(prefix_len=4)
    target_records = {}  # target_id -> (name, addr, country)

    print("[*] Indexing target sources (S2 & S3)...")
    for file_path in [s2_path, s3_path]:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    e_id, b_name, b_addr, country = row[0].strip(), row[1], row[2], row[3].strip()
                    index.add_record(e_id, b_name, b_addr, country)
                    target_records[e_id] = (b_name, b_addr, country)

    print(f"[+] Total target records stored for lookup: {len(target_records):,}")

    # 3. Analyze Missed True Matches
    print(f"[*] Streaming S1 queries and inspecting missed true match pairs...")
    
    total_missed_analyzed = 0
    total_true_analyzed = 0
    total_s1_analyzed = 0

    # Diagnostic Counters
    patterns = Counter()
    sample_missed_pairs = []

    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)

        for row in reader:
            if total_s1_analyzed >= sample_limit:
                break
            if len(row) < 4:
                continue

            s1_id = row[0].strip()
            s1_name = row[1]
            s1_addr = row[2]
            s1_country = row[3].strip()

            total_s1_analyzed += 1
            true_matches = gt_map.get(s1_id, set())

            if not true_matches:
                continue

            total_true_analyzed += len(true_matches)

            # Query baseline blocking index
            evidence_map = index.query_record(s1_name, s1_addr, s1_country)
            cands_found = set(evidence_map.keys())

            # Find missed true matches
            missed_matches = true_matches - cands_found

            for target_id in missed_matches:
                total_missed_analyzed += 1
                t_name, t_addr, t_country = target_records.get(target_id, ("", "", ""))

                # --- 1. Script / Multilingual check ---
                s1_is_latin = is_latin(s1_name)
                t_is_latin = is_latin(t_name)
                if s1_is_latin != t_is_latin or (not s1_is_latin and not t_is_latin):
                    patterns["non_latin_or_script_mismatch"] += 1

                # --- 2. Name Pattern checks ---
                s1_norm = normalize_name(s1_name)
                t_norm = normalize_name(t_name)

                s1_tokens = set(s1_norm.split())
                t_tokens = set(t_norm.split())

                token_intersect = s1_tokens & t_tokens
                if token_intersect:
                    patterns["has_token_overlap"] += 1
                    if len(token_intersect) == 1 and list(token_intersect)[0] in s1_norm.split()[0:1]:
                        patterns["first_token_matches"] += 1
                else:
                    patterns["zero_name_token_overlap"] += 1

                # Check Prefix 5 & Prefix 6
                if s1_norm and t_norm and len(s1_norm) >= 5 and len(t_norm) >= 5 and s1_norm[:5] == t_norm[:5]:
                    patterns["prefix_5_matches"] += 1

                # --- 3. Address Pattern checks ---
                s1_nums = extract_address_numbers(s1_addr)
                t_nums = extract_address_numbers(t_addr)

                if not s1_nums or not t_nums:
                    patterns["missing_address_digits"] += 1
                elif set(s1_nums.split("_")) & set(t_nums.split("_")):
                    patterns["address_digits_overlap_but_name_failed"] += 1
                else:
                    patterns["address_digits_mismatch"] += 1

                # Postal Code check
                s1_zip = extract_postal_code(s1_addr)
                t_zip = extract_postal_code(t_addr)
                if s1_zip and t_zip and s1_zip == t_zip:
                    patterns["postal_code_matches"] += 1

                if len(sample_missed_pairs) < 10:
                    sample_missed_pairs.append((
                        s1_id, s1_name, s1_addr, s1_country,
                        target_id, t_name, t_addr, t_country
                    ))

    print("\n" + "=" * 70)
    print(f"=== FAILURE ANALYSIS RESULTS ({total_s1_analyzed:,} S1 Records Sampled) ===")
    print("=" * 70)
    print(f"Total True Matches Evaluated: {total_true_analyzed:,}")
    print(f"Total Missed Matches:         {total_missed_analyzed:,} ({total_missed_analyzed/total_true_analyzed*100:.2f}%)")

    print("\n--- Root Cause Pattern Breakdown (among Missed Matches) ---")
    for key, count in patterns.most_common():
        pct = (count / total_missed_analyzed * 100) if total_missed_analyzed > 0 else 0.0
        print(f"  {key:42s}: {count:6,d} ({pct:5.2f}%)")

    print("\n--- Sample Missed Pairs (S1 vs Target) ---")
    for i, p in enumerate(sample_missed_pairs[:5], 1):
        print(f"\nSample #{i}:")
        print(f"  S1:     [{p[0]}] '{p[1]}' | Address: '{p[2]}' | Country: {p[3]}")
        print(f"  Target: [{p[4]}] '{p[5]}' | Address: '{p[6]}' | Country: {p[7]}")

if __name__ == "__main__":
    run_failure_analysis()
