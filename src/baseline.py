"""
Baseline Entity Resolution Pipeline for Amazon ML Challenge 2026.
Generates:
1. output/matching_results.tsv (High precision matches)
2. output/candidate_pairs.tsv (Candidate pool from country & normalized name blocking)
"""
import os
import sys
import re
import csv
import time
from collections import defaultdict

# Regex to clean legal suffixes and punctuation
LEGAL_SUFFIXES_PATTERN = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|co|company|enterprises|enterprise|group|services|center)\b',
    re.IGNORECASE
)
PUNCT_PATTERN = re.compile(r'[^\w\s]', re.UNICODE)

def normalize_name(name: str) -> str:
    """Normalize business name for high-precision blocking and matching."""
    if not name:
        return ""
    # Lowercase & remove punctuation
    text = PUNCT_PATTERN.sub(' ', name.lower())
    # Remove common legal suffixes
    text = LEGAL_SUFFIXES_PATTERN.sub(' ', text)
    # Collapse multiple whitespaces
    return " ".join(text.split())

def normalize_address(addr: str) -> str:
    """Normalize business address."""
    if not addr:
        return ""
    text = PUNCT_PATTERN.sub(' ', addr.lower())
    return " ".join(text.split())

def run_baseline(
    data_dir: str = "student_resource/dataset/test",
    output_dir: str = "output"
):
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")

    print("=" * 70)
    print(f"[*] Starting Baseline Pipeline on {data_dir}...")
    print("=" * 70)

    # Step 1: Build Inverted Index on Source 2 and Source 3
    # Index key: (country, normalized_name) -> list of entity_ids
    index = defaultdict(list)
    total_target_records = 0

    for src_name in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(data_dir, src_name)
        print(f"[*] Indexing {src_name}...")
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None) # skip header
            for row in reader:
                total_target_records += 1
                if total_target_records % 2_000_000 == 0:
                    print(f"    Indexed {total_target_records:,} records...")
                if len(row) < 4:
                    continue
                e_id, b_name, b_addr, country = row[0].strip(), row[1], row[2], row[3].strip()
                norm_n = normalize_name(b_name)
                if norm_n and country:
                    index[(country, norm_n)].append(e_id)

    print(f"[+] Total target records indexed: {total_target_records:,}")
    print(f"[+] Unique (country, name) blocking buckets: {len(index):,}")

    # Step 2: Match Test Source 1 entities against the index
    s1_path = os.path.join(data_dir, "test_source1.tsv")
    print(f"[*] Processing {s1_path} and writing outputs...")

    matched_count = 0
    singleton_count = 0
    total_s1 = 0

    with open(s1_path, encoding="utf-8") as s1_f, \
         open(matching_file, "w", encoding="utf-8", newline="") as m_f, \
         open(candidate_file, "w", encoding="utf-8", newline="") as c_f:

        s1_reader = csv.reader(s1_f, delimiter="\t")
        next(s1_reader, None) # skip header

        # Write required headers
        m_f.write("source1_entity_id\tmatched_entity_ids\n")
        c_f.write("source1_entity_id\tcandidate_entity_ids\n")

        for row in s1_reader:
            total_s1 += 1
            if total_s1 % 500_000 == 0:
                print(f"    Processed {total_s1:,} Source 1 entities...")

            s1_id = row[0].strip()
            b_name = row[1] if len(row) > 1 else ""
            b_addr = row[2] if len(row) > 2 else ""
            country = row[3].strip() if len(row) > 3 else ""

            norm_n = normalize_name(b_name)
            candidates = index.get((country, norm_n), [])

            if candidates:
                # Deduplicate while preserving order
                unique_cands = list(dict.fromkeys(candidates))
                cand_str = ",".join(unique_cands)
                # In baseline: candidates with exact normalized name match are our matches
                m_f.write(f"{s1_id}\t{cand_str}\n")
                c_f.write(f"{s1_id}\t{cand_str}\n")
                matched_count += 1
            else:
                # Singleton (no match)
                m_f.write(f"{s1_id}\t\n")
                c_f.write(f"{s1_id}\t\n")
                singleton_count += 1

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[+] FINISHED IN {elapsed:.2f} seconds!")
    print(f"[+] Total S1 entities: {total_s1:,}")
    print(f"[+] Matched entities:  {matched_count:,} ({matched_count/total_s1*100:.2f}%)")
    print(f"[+] Singletons:        {singleton_count:,} ({singleton_count/total_s1*100:.2f}%)")
    print(f"[+] Matching file:     {matching_file} ({os.path.getsize(matching_file)/(1024*1024):.1f} MB)")
    print(f"[+] Candidate file:    {candidate_file} ({os.path.getsize(candidate_file)/(1024*1024):.1f} MB)")
    print("=" * 70)

if __name__ == "__main__":
    run_baseline()
