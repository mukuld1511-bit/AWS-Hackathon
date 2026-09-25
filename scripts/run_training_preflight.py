"""
Stage 5 Training Preflight & Determinism Verification Script.

Executes the production candidate generator on the full training dataset:
- S1: ~2.2M records
- S2: ~5.03M records
- S3: ~5.28M records

Verifies all production invariants, determinism, and alignment with Stage 4.8.5 metrics.
"""

import os
import sys
import time
import hashlib
import psutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.candidate_generator import generate_candidate_pairs, get_frozen_r3_3_ranking_config
from src.evaluate_candidate_recall import load_ground_truth


def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of a file for determinism verification."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def main():
    print("=" * 80, flush=True)
    print("=== STAGE 5: PRODUCTION CANDIDATE GENERATOR — TRAINING PREFLIGHT ===", flush=True)
    print("=" * 80 + "\n", flush=True)

    train_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
    s1_path = os.path.join(train_dir, "train_source1.tsv")
    s2_path = os.path.join(train_dir, "train_source2.tsv")
    s3_path = os.path.join(train_dir, "train_source3.tsv")
    gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

    preflight_output_path = "data/preflight_candidate_pairs.tsv"
    os.makedirs("data", exist_ok=True)

    # 1. Load Ground Truth for Recall Validation
    print("[*] Loading training ground truth mapping...", flush=True)
    gt_map = load_ground_truth(gt_path)

    # 2. Execute Production Candidate Generator Run 1
    print("\n[*] Executing Production Generator (Run 1)...", flush=True)
    report_run1 = generate_candidate_pairs(
        s1_path=s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_path=preflight_output_path,
        gt_map=gt_map
    )

    hash_run1 = compute_file_hash(preflight_output_path)

    print("\n" + "=" * 75, flush=True)
    print("=== PREFLIGHT VALIDATION METRICS REPORT (RUN 1) ===", flush=True)
    print("=" * 75, flush=True)
    print(f"S1 Records Processed       : {report_run1.total_s1_records:,}")
    print(f"Non-singletons             : {report_run1.non_singletons_count:,}")
    print(f"Singletons                 : {report_run1.singletons_count:,}")
    print(f"Total True Matches         : {report_run1.total_true_matches:,}")
    print(f"Uncapped Recovered Matches : {report_run1.uncapped_recovered_matches:,}")
    print(f"Uncapped Blocking Recall   : {report_run1.uncapped_blocking_recall:.4f}%")
    print(f"Recovered @ K=25 Matches   : {report_run1.recov_k25_matches:,}")
    print(f"Recall@25 (Primary)        : {report_run1.recall_at_25:.4f}%")

    print("\nRetained Candidate Statistics (Capped at K=25):")
    print(f"  Mean Candidates per S1   : {report_run1.mean_candidates}")
    print(f"  P50 Candidates           : {report_run1.p50_candidates}")
    print(f"  P95 Candidates           : {report_run1.p95_candidates}")
    print(f"  P99 Candidates           : {report_run1.p99_candidates}")
    print(f"  Max Candidates           : {report_run1.max_candidates}")

    print("\nS1 Distribution Breakdown:")
    print(f"  Zero Candidates          : {report_run1.zero_candidate_s1:,}")
    print(f"  Fewer than 25 Candidates : {report_run1.fewer_than_25_cands:,}")
    print(f"  Exactly 25 Candidates    : {report_run1.exactly_25_cands:,}")

    print("\nSource Breakdown:")
    print(f"  S2 Candidate Instances   : {report_run1.s2_candidates_count:,}")
    print(f"  S3 Candidate Instances   : {report_run1.s3_candidates_count:,}")

    print("\nProduction Invariant Checks:")
    print(f"  Dropped S1 Records       : {report_run1.dropped_s1_records} (EXPECTED: 0)")
    print(f"  Cross-Country Violations : {report_run1.cross_country_violations} (EXPECTED: 0)")
    print(f"  Duplicate Candidate Pairs: {report_run1.duplicate_pairs_count} (EXPECTED: 0)")
    print(f"  Invalid Target Entity IDs: {report_run1.invalid_target_ids_count} (EXPECTED: 0)")
    print(f"  Output File SHA256       : {hash_run1}")

    # Assert Production Invariants
    assert report_run1.dropped_s1_records == 0, "INVARIANT VIOLATED: Dropped S1 records > 0!"
    assert report_run1.cross_country_violations == 0, "INVARIANT VIOLATED: Cross-country pairs > 0!"
    assert report_run1.duplicate_pairs_count == 0, "INVARIANT VIOLATED: Duplicate candidate pairs detected!"
    assert report_run1.invalid_target_ids_count == 0, "INVARIANT VIOLATED: Invalid target IDs detected!"
    assert abs(report_run1.uncapped_blocking_recall - 92.7934) < 0.05, f"Uncapped recall drift detected: {report_run1.uncapped_blocking_recall}%"
    assert abs(report_run1.recall_at_25 - 79.6692) < 0.05, f"Recall@25 drift detected: {report_run1.recall_at_25}%"

    print("\n[+] ALL TRAINING PREFLIGHT INVARIANTS PASSED PERFECTLY!", flush=True)

    # 3. Determinism Test (Run 2 on 50,000 S1 subset)
    print("\n[*] Running Determinism & Hash Verification Check (50,000 S1 subset)...", flush=True)
    sample_s1_path = "data/s1_50k_sample.tsv"
    with open(s1_path, "r", encoding="utf-8") as fin, open(sample_s1_path, "w", encoding="utf-8", newline="") as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.writer(fout, delimiter="\t")
        writer.writerow(next(reader))
        for i, row in enumerate(reader):
            if i >= 50000:
                break
            writer.writerow(row)

    report_det1 = generate_candidate_pairs(
        s1_path=sample_s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_path="data/det_candidate_pairs_run1.tsv"
    )
    hash_det1 = compute_file_hash("data/det_candidate_pairs_run1.tsv")

    report_det2 = generate_candidate_pairs(
        s1_path=sample_s1_path,
        s2_path=s2_path,
        s3_path=s3_path,
        output_path="data/det_candidate_pairs_run2.tsv"
    )
    hash_det2 = compute_file_hash("data/det_candidate_pairs_run2.tsv")

    print(f"  Run 1 SHA256 (50k sample): {hash_det1}")
    print(f"  Run 2 SHA256 (50k sample): {hash_det2}")
    assert hash_det1 == hash_det2, "DETERMINISM FAILED: File hashes differ between runs!"
    print("  [+] DETERMINISM TEST 100% IDENTICAL (PASSED)")

    print("\n" + "=" * 80)
    print(f"=== TRAINING PREFLIGHT COMPLETED SUCCESSFULLY IN {report_run1.total_runtime_sec}s ===")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
