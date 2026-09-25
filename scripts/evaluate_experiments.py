"""
Experimental Evaluation Script (Stage 4.5).

Evaluates baseline A+B+C vs experimental blocking strategies (+D, +E, +F)
one at a time on 100,000 S1 training records to measure:
- Incremental recall improvement
- Candidate volume explosion
- Posting list frequency caps
- Memory & Runtime impact
"""

import os
import sys
import csv
import time
import psutil
sys.path.insert(0, "/home/piet/Desktop/aws model 2")

from src.experimental_blocking import ExperimentalBlockingIndex
from src.evaluate_candidate_recall import load_ground_truth
from src.ranking import RankingConfig, rank_candidates

train_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
s1_path = os.path.join(train_dir, "train_source1.tsv")
s2_path = os.path.join(train_dir, "train_source2.tsv")
s3_path = os.path.join(train_dir, "train_source3.tsv")
gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

def run_experiment(
    exp_name: str,
    enable_exp_d: bool = False,
    enable_exp_e: bool = False,
    enable_exp_f: bool = False,
    max_s1_eval: int = 100000,
    gt_map: dict = None
):
    print(f"\n" + "=" * 70)
    print(f"=== RUNNING EXPERIMENT: {exp_name} ===")
    print("=" * 70)

    process = psutil.Process()
    rss_start = process.memory_info().rss / (1024 * 1024)

    # 1. Build Index
    index = ExperimentalBlockingIndex(
        prefix_len=4,
        enable_exp_d=enable_exp_d,
        enable_exp_e=enable_exp_e,
        enable_exp_f=enable_exp_f,
        max_posting_threshold=500
    )

    index_start = time.time()
    total_target = 0
    for file_path in [s2_path, s3_path]:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    index.add_record(row[0].strip(), row[1], row[2], row[3].strip())
                    total_target += 1

    index_time = time.time() - index_start
    rss_after_index = process.memory_info().rss / (1024 * 1024)

    # 2. Evaluate S1 Queries
    eval_start = time.time()
    total_s1 = 0
    total_true_matches = 0
    blocking_recovered = 0

    cand_counts = []

    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if total_s1 >= max_s1_eval:
                break
            if len(row) < 4:
                continue

            s1_id = row[0].strip()
            name, addr, country = row[1], row[2], row[3].strip()

            total_s1 += 1
            true_matches = gt_map.get(s1_id, set())
            total_true_matches += len(true_matches)

            evidence_map = index.query_record(name, addr, country)
            cands_set = set(evidence_map.keys())

            cand_counts.append(len(cands_set))

            if true_matches:
                blocking_recovered += len(true_matches & cands_set)

    eval_time = time.time() - eval_start
    total_time = index_time + eval_time

    recall = (blocking_recovered / total_true_matches) if total_true_matches > 0 else 0.0
    avg_cands = (sum(cand_counts) / len(cand_counts)) if cand_counts else 0.0
    max_cands = max(cand_counts) if cand_counts else 0

    print(f"Indexed {total_target:,} records in {index_time:.2f}s")
    print(f"Evaluated {total_s1:,} S1 queries in {eval_time:.2f}s")
    print(f"Total Runtime: {total_time:.2f}s | RAM RSS: {rss_after_index:.2f} MB")
    print(f"Total True Matches: {total_true_matches:,}")
    print(f"Blocking Recovered: {blocking_recovered:,} ({recall * 100:.2f}%)")
    print(f"Avg Candidates / S1 (Uncapped): {avg_cands:.2f}")
    print(f"Max Candidates / S1 (Uncapped): {max_cands}")

    return {
        "exp_name": exp_name,
        "recall": recall,
        "recovered": blocking_recovered,
        "total_true": total_true_matches,
        "avg_cands": avg_cands,
        "max_cands": max_cands,
        "total_time": total_time,
        "ram_mb": rss_after_index
    }

def main():
    print("[*] Loading ground truth...")
    gt_map = load_ground_truth(gt_path)

    # 1. Baseline A+B+C
    base_res = run_experiment("Baseline A+B+C", False, False, False, 100000, gt_map)

    # 2. + Exp D (Token Anchor)
    exp_d_res = run_experiment("+ Exp D (Name Token Anchor)", True, False, False, 100000, gt_map)

    # 3. + Exp E (5-char Prefix)
    exp_e_res = run_experiment("+ Exp E (5-char Prefix)", False, True, False, 100000, gt_map)

    # 4. + Exp F (Street Pair Anchor)
    exp_f_res = run_experiment("+ Exp F (Street Pair Anchor)", False, False, True, 100000, gt_map)

    print("\n" + "=" * 80)
    print("=== STAGE 4.5 EXPERIMENTAL COMPARISON TABLE (100,000 S1 Sample) ===")
    print("=" * 80)
    print(f"| {'Strategy':30s} | {'Blocking Recall':15s} | {'Incremental Recovered':21s} | {'Avg Cands/S1':12s} | {'Max Cands':9s} | {'Runtime':9s} | {'RAM (MB)':9s} |")
    print("-" * 115)

    results = [base_res, exp_d_res, exp_e_res, exp_f_res]
    base_rec = base_res["recovered"]

    for r in results:
        incr = (r["recovered"] - base_rec) if r != base_res else 0
        incr_str = f"+{incr:,}" if incr > 0 else "—"
        print(f"| {r['exp_name']:30s} | {r['recall']*100:14.2f}% | {incr_str:21s} | {r['avg_cands']:12.2f} | {r['max_cands']:9d} | {r['total_time']:8.2f}s | {r['ram_mb']:9.2f} |")

if __name__ == "__main__":
    main()
