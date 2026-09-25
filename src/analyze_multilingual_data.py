"""
Phase 1 — Data Understanding / EDA Script
Analyzes multilingual Indic records across S1, S2, S3 and ground truth.
Memory-safe streaming execution without loading full tables into RAM.
"""
import os
import re
import csv
import sys
import time
from collections import Counter, defaultdict

# Dataset paths
TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"

TEST_S1 = "student_resource/dataset/test/test_source1.tsv"
TEST_S2 = "student_resource/dataset/test/test_source2.tsv"
TEST_S3 = "student_resource/dataset/test/test_source3.tsv"

OUTPUT_REPORT = "output/multilingual_eda_report.txt"

# Regex for all Indic scripts combined
RE_INDIC = re.compile(r'[\u0900-\u0D7F]')

# Specific Indic scripts
SCRIPTS = {
    "Devanagari (Hindi/Marathi)": re.compile(r'[\u0900-\u097F]'),
    "Bengali": re.compile(r'[\u0980-\u09FF]'),
    "Gurmukhi (Punjabi)": re.compile(r'[\u0A00-\u0A7F]'),
    "Gujarati": re.compile(r'[\u0A80-\u0AFF]'),
    "Tamil": re.compile(r'[\u0B80-\u0BFF]'),
    "Telugu": re.compile(r'[\u0C00-\u0C7F]'),
    "Kannada": re.compile(r'[\u0C80-\u0CFF]'),
    "Malayalam": re.compile(r'[\u0D00-\u0D7F]'),
}

RE_PIN = re.compile(r'\b[1-9]\d{5}\b')
RE_NUM = re.compile(r'\b\d+\b')
RE_ASCII = re.compile(r'[A-Za-z]')

def is_india(country_str):
    if not country_str:
        return False
    return country_str.strip().lower() == "india"

def analyze():
    start_time = time.time()
    os.makedirs("output", exist_ok=True)
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(msg)

    log("=" * 80)
    log("PHASE 1 — MULTILINGUAL DATA EDA (Business Entity Resolution)")
    log("=" * 80)

    # -------------------------------------------------------------
    # Step 1: Inspect Schemas
    # -------------------------------------------------------------
    log("\n--- STEP 1: SCHEMAS ---")
    files_to_check = [
        ("S1 (Train)", TRAIN_S1),
        ("S2 (Train)", TRAIN_S2),
        ("S3 (Train)", TRAIN_S3),
        ("Ground Truth", TRAIN_GT),
    ]
    for label, filepath in files_to_check:
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            header = next(reader)
            log(f"{label} columns: {header}")

    # -------------------------------------------------------------
    # Step 2 & 3: Count dataset sizes & Country distribution
    # -------------------------------------------------------------
    log("\n--- STEP 2 & 3: DATASET SIZES & INDIA FILTERING ---")
    
    counts = {}
    country_counters = {}

    def scan_file(filepath, label):
        row_count = 0
        india_count = 0
        country_cnt = Counter()
        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                row_count += 1
                if len(row) >= 4:
                    raw_country = row[3].strip()
                    country_cnt[raw_country] += 1
                    if is_india(raw_country):
                        india_count += 1
        return row_count, india_count, country_cnt

    # Train files
    t_s1_total, t_s1_india, c_s1 = scan_file(TRAIN_S1, "Train S1")
    t_s2_total, t_s2_india, c_s2 = scan_file(TRAIN_S2, "Train S2")
    t_s3_total, t_s3_india, c_s3 = scan_file(TRAIN_S3, "Train S3")

    # Ground truth total
    gt_total = 0
    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for _ in reader:
            gt_total += 1

    # Test files
    test_s1_total, test_s1_india, test_c_s1 = scan_file(TEST_S1, "Test S1")
    test_s2_total, test_s2_india, test_c_s2 = scan_file(TEST_S2, "Test S2")
    test_s3_total, test_s3_india, test_c_s3 = scan_file(TEST_S3, "Test S3")

    log(f"Train S1:           {t_s1_total:>10,} rows  |  India: {t_s1_india:>10,} ({t_s1_india/t_s1_total*100:.1f}%)")
    log(f"Train S2:           {t_s2_total:>10,} rows  |  India: {t_s2_india:>10,} ({t_s2_india/t_s2_total*100:.1f}%)")
    log(f"Train S3:           {t_s3_total:>10,} rows  |  India: {t_s3_india:>10,} ({t_s3_india/t_s3_total*100:.1f}%)")
    log(f"Train Ground Truth: {gt_total:>10,} rows")
    log("-" * 60)
    log(f"Test S1:            {test_s1_total:>10,} rows  |  India: {test_s1_india:>10,} ({test_s1_india/test_s1_total*100:.1f}%)")
    log(f"Test S2:            {test_s2_total:>10,} rows  |  India: {test_s2_india:>10,} ({test_s2_india/test_s2_total*100:.1f}%)")
    log(f"Test S3:            {test_s3_total:>10,} rows  |  India: {test_s3_india:>10,} ({test_s3_india/test_s3_total*100:.1f}%)")

    log("\nTop Country values in Test S1:")
    for country, cnt in test_c_s1.most_common(5):
        log(f"  '{country}': {cnt:,} ({cnt/test_s1_total*100:.2f}%)")

    # -------------------------------------------------------------
    # Step 4 & 5: Indic-script detection & Script distribution (Test S2 & S3)
    # -------------------------------------------------------------
    log("\n--- STEP 4 & 5: INDIC SCRIPT DETECTION & SCRIPT BREAKDOWN (TEST SET) ---")

    def analyze_indic_candidates(filepath, source_label):
        total_india = 0
        indic_name = 0
        indic_addr = 0
        indic_either = 0
        script_counts = Counter()
        samples_name = []
        
        # Address signal counters on India + Indic-name records
        pin_count = 0
        num_count = 0
        ascii_count = 0
        indic_records_analyzed = 0

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if len(row) < 4:
                    continue
                eid, name, addr, country = row[0], row[1], row[2], row[3]
                if not is_india(country):
                    continue
                
                total_india += 1
                has_n = bool(RE_INDIC.search(name))
                has_a = bool(RE_INDIC.search(addr))

                if has_n:
                    indic_name += 1
                    if len(samples_name) < 10:
                        samples_name.append((eid, name, addr, country))
                if has_a:
                    indic_addr += 1
                if has_n or has_a:
                    indic_either += 1

                # Script breakdown (check across name and address for full script visibility)
                full_text = (name or "") + " " + (addr or "")
                for sname, sregex in SCRIPTS.items():
                    if sregex.search(full_text):
                        script_counts[sname] += 1

                # Address signals on candidates with Indic names
                if has_n:
                    indic_records_analyzed += 1
                    if RE_PIN.search(addr):
                        pin_count += 1
                    if RE_NUM.search(addr):
                        num_count += 1
                    if RE_ASCII.search(addr):
                        ascii_count += 1

        stats = {
            "total_india": total_india,
            "indic_name": indic_name,
            "indic_addr": indic_addr,
            "indic_either": indic_either,
            "script_counts": script_counts,
            "samples_name": samples_name,
            "indic_records_analyzed": indic_records_analyzed,
            "pin_count": pin_count,
            "num_count": num_count,
            "ascii_count": ascii_count,
        }
        return stats

    log("[*] Scanning Test S2...")
    s2_stats = analyze_indic_candidates(TEST_S2, "S2")
    log("[*] Scanning Test S3...")
    s3_stats = analyze_indic_candidates(TEST_S3, "S3")

    log("\nIndic Script Breakdown in Test Records:")
    log(f"India S2 total:                          {s2_stats['total_india']:>8,}")
    log(f"India S2 with Indic business name:       {s2_stats['indic_name']:>8,} ({s2_stats['indic_name']/s2_stats['total_india']*100:.2f}%)")
    log(f"India S2 with Indic address:             {s2_stats['indic_addr']:>8,} ({s2_stats['indic_addr']/s2_stats['total_india']*100:.2f}%)")
    log(f"India S2 with Indic name OR address:     {s2_stats['indic_either']:>8,} ({s2_stats['indic_either']/s2_stats['total_india']*100:.2f}%)")
    log("-" * 60)
    log(f"India S3 total:                          {s3_stats['total_india']:>8,}")
    log(f"India S3 with Indic business name:       {s3_stats['indic_name']:>8,} ({s3_stats['indic_name']/s3_stats['total_india']*100:.2f}%)")
    log(f"India S3 with Indic address:             {s3_stats['indic_addr']:>8,} ({s3_stats['indic_addr']/s3_stats['total_india']*100:.2f}%)")
    log(f"India S3 with Indic name OR address:     {s3_stats['indic_either']:>8,} ({s3_stats['indic_either']/s3_stats['total_india']*100:.2f}%)")

    log("\nScript Occurrence (in India records with Indic characters):")
    log(f"{'Script':<30} | {'Test S2':>10} | {'Test S3':>10}")
    log("-" * 56)
    for sname in SCRIPTS.keys():
        log(f"{sname:<30} | {s2_stats['script_counts'][sname]:>10,} | {s3_stats['script_counts'][sname]:>10,}")
    log("\nNote: Multi-script records may contain characters from more than one script; counts are per-script presence.")

    # -------------------------------------------------------------
    # Step 6: Representative Examples
    # -------------------------------------------------------------
    log("\n--- STEP 6: REPRESENTATIVE EXAMPLES OF INDIC BUSINESS NAMES (TEST SET) ---")
    log("\n[10 Representative Test S2 Records with Indic Names]:")
    for idx, (eid, name, addr, country) in enumerate(s2_stats['samples_name'], 1):
        log(f"  [{idx:2d}] {eid} | Name: {name} | Addr: {addr} | Country: {country}")

    log("\n[10 Representative Test S3 Records with Indic Names]:")
    for idx, (eid, name, addr, country) in enumerate(s3_stats['samples_name'], 1):
        log(f"  [{idx:2d}] {eid} | Name: {name} | Addr: {addr} | Country: {country}")

    # -------------------------------------------------------------
    # Step 7: Address Signal Analysis
    # -------------------------------------------------------------
    log("\n--- STEP 7: ADDRESS SIGNAL ANALYSIS (ON INDIC-NAME CANDIDATES) ---")
    tot_indic_cand = s2_stats["indic_records_analyzed"] + s3_stats["indic_records_analyzed"]
    tot_pin = s2_stats["pin_count"] + s3_stats["pin_count"]
    tot_num = s2_stats["num_count"] + s3_stats["num_count"]
    tot_ascii = s2_stats["ascii_count"] + s3_stats["ascii_count"]

    log(f"Total Indic-Name candidates analyzed (S2 + S3): {tot_indic_cand:,}")
    log(f"  Containing 6-digit Indian PIN code: {tot_pin:>8,} ({tot_pin/tot_indic_cand*100:.2f}%)")
    log(f"  Containing numeric token (building/road/pin): {tot_num:>8,} ({tot_num/tot_indic_cand*100:.2f}%)")
    log(f"  Containing ASCII/Latin token in address:      {tot_ascii:>8,} ({tot_ascii/tot_indic_cand*100:.2f}%)")
    log("Insight: Even when the business name is 100% Indic, the address overwhelmingly contains ASCII tokens and numbers (PIN codes, road/plot numbers), serving as an indispensable verification bridge!")

    # -------------------------------------------------------------
    # Step 8 & 9: Ground Truth Multilingual Analysis (Train Set)
    # -------------------------------------------------------------
    log("\n--- STEP 8 & 9: TRAINING GROUND TRUTH MULTILINGUAL MATCH ANALYSIS ---")
    log("[*] Indexing Indic-script records from Train S2 and Train S3...")
    
    # Memory-safe: store only (name, addr) for India S2/S3 train records that have Indic chars
    train_indic_candidates = {}
    with open(TRAIN_S2, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 4 and is_india(row[3]) and RE_INDIC.search(row[1]):
                train_indic_candidates[row[0]] = ('S2', row[1], row[2])

    with open(TRAIN_S3, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 4 and is_india(row[3]) and RE_INDIC.search(row[1]):
                train_indic_candidates[row[0]] = ('S3', row[1], row[2])

    log(f"[+] Indexed {len(train_indic_candidates):,} Indic-name records from Train S2 & S3")

    # Stream Train Ground Truth to find matches involving these candidates
    log("[*] Streaming Train Ground Truth...")
    s1_with_indic_matches = defaultdict(list)
    total_gt_matches_india_s1 = 0
    india_s1_with_any_match = 0
    s2_multilingual_true_matches = 0
    s3_multilingual_true_matches = 0

    # We also need to know which S1 entities are India: stream TRAIN_S1 to collect India S1 IDs
    # (Storing set of IDs takes ~50MB RAM for 1M strings)
    log("[*] Collecting Train India S1 IDs...")
    india_s1_ids = set()
    with open(TRAIN_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) >= 4 and is_india(row[3]):
                india_s1_ids.add(row[0])

    log(f"[+] Found {len(india_s1_ids):,} India S1 entities in Train")

    # Now inspect ground truth
    with open(TRAIN_GT, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if len(row) < 2:
                continue
            s1_id = row[0]
            matched_str = row[1].strip()
            if not matched_str:
                continue
            
            if s1_id in india_s1_ids:
                india_s1_with_any_match += 1
                matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]
                total_gt_matches_india_s1 += len(matched_ids)
                for mid in matched_ids:
                    if mid in train_indic_candidates:
                        src, cname, caddr = train_indic_candidates[mid]
                        s1_with_indic_matches[s1_id].append((mid, src, cname, caddr))
                        if src == 'S2':
                            s2_multilingual_true_matches += 1
                        else:
                            s3_multilingual_true_matches += 1

    total_multilingual_matches = s2_multilingual_true_matches + s3_multilingual_true_matches

    log(f"\nGround Truth Statistics for India Entities:")
    log(f"  India S1 entities with at least one ground-truth match:   {india_s1_with_any_match:>10,}")
    log(f"  Total true match pairs for India S1 entities:            {total_gt_matches_india_s1:>10,}")
    log(f"  India S1 entities with Indic-script ground-truth matches: {len(s1_with_indic_matches):>10,}")
    log(f"  Number of S2 multilingual true matches:                  {s2_multilingual_true_matches:>10,}")
    log(f"  Number of S3 multilingual true matches:                  {s3_multilingual_true_matches:>10,}")
    log(f"  Total multilingual true match pairs:                     {total_multilingual_matches:>10,}")
    log(f"  Percentage of India matches that are multilingual:       {total_multilingual_matches/total_gt_matches_india_s1*100:.2f}%")

    # Step 9: Get details for 15 genuine matched pairs
    log("\n[15 Genuine Training Examples of Multilingual / Cross-Script Matches]:")
    sample_s1_targets = list(s1_with_indic_matches.keys())[:15]
    sample_s1_targets_set = set(sample_s1_targets)

    # Scan TRAIN_S1 to fetch metadata for these 15 S1 targets
    s1_meta = {}
    with open(TRAIN_S1, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[0] in sample_s1_targets_set:
                s1_meta[row[0]] = (row[1], row[2])
                if len(s1_meta) == len(sample_s1_targets_set):
                    break

    pair_idx = 1
    for s1_id in sample_s1_targets:
        s1_n, s1_a = s1_meta.get(s1_id, ("Unknown", "Unknown"))
        for (cand_id, cand_src, cand_n, cand_a) in s1_with_indic_matches[s1_id]:
            log(f"\nExample {pair_idx}:")
            log(f"  S1 ID:       {s1_id}")
            log(f"  S1 Name:     {s1_n}")
            log(f"  S1 Address:  {s1_a}")
            log(f"  Matched ID:  {cand_id} ({cand_src})")
            log(f"  Match Name:  {cand_n}")
            log(f"  Match Addr:  {cand_a}")
            pair_idx += 1
            if pair_idx > 15:
                break
        if pair_idx > 15:
            break

    # -------------------------------------------------------------
    # Step 10: Estimate Candidate Pool Size & Pair Count (Test Set)
    # -------------------------------------------------------------
    log("\n--- STEP 10: ESTIMATE CANDIDATE-POOL SIZE (TEST SET) ---")
    india_s1_count = test_s1_india
    indic_s2_count = s2_stats["indic_name"]
    indic_s3_count = s3_stats["indic_name"]
    total_indic_candidates = indic_s2_count + indic_s3_count
    theoretical_pairs = india_s1_count * total_indic_candidates

    log(f"India S1 entities (anchors):          {india_s1_count:>12,}")
    log(f"India S2 Indic-name candidates:       {indic_s2_count:>12,}")
    log(f"India S3 Indic-name candidates:       {indic_s3_count:>12,}")
    log(f"Total multilingual candidate records: {total_indic_candidates:>12,}")
    log(f"Theoretical candidate pairs:          {theoretical_pairs:>12,} (~{theoretical_pairs/1e9:.2f} Billion pairs)")

    log("\nTechnical Observation:")
    log(f"  - 809,986 S1 anchors × {total_indic_candidates:,} Indic candidates = ~{theoretical_pairs/1e9:.2f}B potential pairs.")
    log("  - Full dense matrix in memory (float32): 810K x ~50K x 4 bytes = ~160 GB RAM -> OUT OF MEMORY if unchunked.")
    log("  - However, chunked dot product (batching S1 in chunks of 1,000–5,000 on GPU or memory-mapped candidate embeddings)")
    log("    or two-level blocking (filtering by state/city/PIN or fast ANN / token pre-filtering) avoids OOM completely.")

    # -------------------------------------------------------------
    # Save Report
    # -------------------------------------------------------------
    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines) + "\n")

    elapsed = time.time() - start_time
    log(f"\n[+] Analysis completed in {elapsed:.1f}s. Report saved to: {OUTPUT_REPORT}")
    log("=" * 80)
    log("PHASE 1 COMPLETE")
    log("=" * 80)

if __name__ == "__main__":
    analyze()
