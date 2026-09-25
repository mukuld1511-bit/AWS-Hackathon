"""
Stage 5 Production Test Candidate Generation & Validation Script.

Executes the production candidate generator on the real test dataset:
- S1 Test: ~1.73M records (test_source1.tsv)
- S2 Test: ~4.89M records (test_source2.tsv)
- S3 Test: ~5.08M records (test_source3.tsv)

Generates official output/candidate_pairs.tsv and performs complete output integrity validation.
"""

import os
import sys
import time
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.candidate_generator import generate_candidate_pairs, get_frozen_r3_3_ranking_config


def validate_test_candidate_output(output_path: str, s1_test_path: str):
    """Perform rigorous output integrity checks on candidate_pairs.tsv."""
    print("\n[*] Step 3/3: Running Output Integrity & Invariant Checks...", flush=True)
    
    assert os.path.exists(output_path), f"Output file not found: {output_path}"
    file_size_mb = round(os.path.getsize(output_path) / (1024 * 1024), 2)
    print(f"  [+] Output file exists: '{output_path}' ({file_size_mb} MB)")

    # Count test S1 records
    expected_s1_ids = []
    with open(s1_test_path, "r", encoding="utf-8") as f:
        import csv
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                expected_s1_ids.append(row[0].strip())

    total_expected_s1 = len(expected_s1_ids)
    print(f"  [+] Expected Test S1 count: {total_expected_s1:,}")

    # Read output file
    out_s1_count = 0
    total_pairs = 0
    duplicate_s1_rows = 0
    max_cands = 0
    over_25_count = 0
    null_s1_count = 0
    seen_s1_ids = set()

    with open(output_path, "r", encoding="utf-8") as f:
        import csv
        reader = csv.reader(f, delimiter="\t")
        header = next(reader, None)
        
        assert header == ["source1_entity_id", "candidate_entity_ids"], f"Invalid TSV header: {header}"
        print("  [+] Header verified: ['source1_entity_id', 'candidate_entity_ids']")

        for i, row in enumerate(reader):
            if len(row) < 1:
                null_s1_count += 1
                continue

            s1_id = row[0].strip()
            if not s1_id:
                null_s1_count += 1
                continue

            if s1_id in seen_s1_ids:
                duplicate_s1_rows += 1
            seen_s1_ids.add(s1_id)

            cand_str = row[1].strip() if len(row) > 1 else ""
            if cand_str:
                cand_list = [c.strip() for c in cand_str.split(",") if c.strip()]
                num_cands = len(cand_list)
                total_pairs += num_cands
                
                # Check deduplication per S1
                assert len(cand_list) == len(set(cand_list)), f"Duplicate candidate IDs found for S1 entity {s1_id}!"
                
                if num_cands > 25:
                    over_25_count += 1
                if num_cands > max_cands:
                    max_cands = num_cands

            out_s1_count += 1

    print(f"  [+] Total S1 rows in output: {out_s1_count:,}")
    print(f"  [+] Total candidate pairs in output: {total_pairs:,}")
    print(f"  [+] Maximum candidates for any S1: {max_cands}")

    # Assertions
    assert out_s1_count == total_expected_s1, f"Row count mismatch! Expected {total_expected_s1}, found {out_s1_count}"
    assert duplicate_s1_rows == 0, f"Duplicate S1 rows detected: {duplicate_s1_rows}"
    assert over_25_count == 0, f"S1 records exceeding 25 candidates detected: {over_25_count}"
    assert null_s1_count == 0, f"Null or empty S1 rows detected: {null_s1_count}"
    assert set(expected_s1_ids) == seen_s1_ids, "S1 ID sequence mismatch between test set and output!"

    print("  [+] ALL OUTPUT INTEGRITY CHECKS PASSED PERFECTLY!")


def main():
    print("=" * 80, flush=True)
    print("=== STAGE 5: PRODUCTION TEST CANDIDATE GENERATION ===", flush=True)
    print("=" * 80 + "\n", flush=True)

    test_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/test"
    test_s1_path = os.path.join(test_dir, "test_source1.tsv")
    test_s2_path = os.path.join(test_dir, "test_source2.tsv")
    test_s3_path = os.path.join(test_dir, "test_source3.tsv")
    
    output_path = "output/candidate_pairs.tsv"

    print("[*] Target Output Path: 'output/candidate_pairs.tsv'", flush=True)

    # Execute Candidate Generator on Test Dataset
    report = generate_candidate_pairs(
        s1_path=test_s1_path,
        s2_path=test_s2_path,
        s3_path=test_s3_path,
        output_path=output_path
    )

    print("\n" + "=" * 75, flush=True)
    print("=== STAGE 5 PRODUCTION TEST CANDIDATE GENERATION REPORT ===", flush=True)
    print("=" * 75, flush=True)
    print(f"Test S1 Entities Processed  : {report.total_s1_records:,}")
    print(f"Total Candidate Pairs      : {report.total_candidate_pairs:,}")
    print(f"Average Candidates per S1  : {report.mean_candidates}")
    print(f"P50 Candidates (Median)    : {report.p50_candidates}")
    print(f"P95 Candidates             : {report.p95_candidates}")
    print(f"P99 Candidates             : {report.p99_candidates}")
    print(f"Maximum Candidates         : {report.max_candidates}")

    print("\nDistribution Breakdown:")
    print(f"  S1 with 0 candidates     : {report.zero_candidate_s1:,} ({report.zero_candidate_s1/report.total_s1_records*100:.2f}%)")
    print(f"  S1 with 1-24 candidates  : {report.fewer_than_25_cands:,} ({report.fewer_than_25_cands/report.total_s1_records*100:.2f}%)")
    print(f"  S1 with 25 candidates    : {report.exactly_25_cands:,} ({report.exactly_25_cands/report.total_s1_records*100:.2f}%)")

    print("\nSource Target Breakdown:")
    print(f"  S2 Candidate Pairs       : {report.s2_candidates_count:,}")
    print(f"  S3 Candidate Pairs       : {report.s3_candidates_count:,}")

    print("\nPerformance & File Statistics:")
    print(f"  Indexing Time (10M test) : {report.indexing_time_sec}s ({report.indexing_time_sec/60:.2f} min)")
    print(f"  Query / Scoring Time     : {report.query_eval_time_sec}s ({report.query_eval_time_sec/60:.2f} min)")
    print(f"  Total End-to-End Time    : {report.total_runtime_sec}s ({report.total_runtime_sec/60:.2f} min)")
    print(f"  Peak Memory (RAM)        : {report.peak_ram_gb} GB")
    print(f"  Output File Size         : {report.output_file_size_mb} MB")

    # Perform Validation
    validate_test_candidate_output(output_path, test_s1_path)

    print("\n" + "=" * 80, flush=True)
    print(f"=== STAGE 5 PRODUCTION TEST GENERATION COMPLETED IN {report.total_runtime_sec}s ===", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
