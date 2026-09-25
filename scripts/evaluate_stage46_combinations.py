"""
Stage 4.6 Experimental Combination & Threshold Optimization Evaluator.

Evaluates combinations of baseline A+B+C with D (Token Anchor) and F (Street Pair Anchor)
across configurable thresholds (100, 250, 500) and token selection strategies (first vs longest)
on 100,000 S1 training records to measure:
- Marginal recall gains beyond baseline, D, and F
- D vs F true-match overlap analysis (D-only, F-only, D+F, Neither)
- Candidate volume percentiles (p50, p95, p99, max, counts >25, >50, >100, >250, >500, >1000)
- Posting list size statistics
- Runtime & RAM footprint
"""

import os
import sys
import csv
import time
import psutil
from collections import Counter, defaultdict

sys.path.insert(0, "/home/piet/Desktop/aws model 2")
from src.experimental_blocking import Stage46ExperimentalIndex
from src.evaluate_candidate_recall import load_ground_truth

train_dir = "6ab10eb3b23ba_student_resource/student_resource/dataset/train"
s1_path = os.path.join(train_dir, "train_source1.tsv")
s2_path = os.path.join(train_dir, "train_source2.tsv")
s3_path = os.path.join(train_dir, "train_source3.tsv")
gt_path = os.path.join(train_dir, "train_ground_truth.tsv")

def calculate_percentiles(values: list) -> dict:
    if not values:
        return {"avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
    s_val = sorted(values)
    n = len(s_val)
    def pct(p):
        return s_val[int(p * (n - 1))]
    return {
        "avg": round(sum(s_val) / n, 2),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": s_val[-1],
        "gt_25_pct": round(sum(1 for v in s_val if v > 25) / n * 100, 2),
        "gt_50_pct": round(sum(1 for v in s_val if v > 50) / n * 100, 2),
        "gt_100_pct": round(sum(1 for v in s_val if v > 100) / n * 100, 2),
        "gt_250_pct": round(sum(1 for v in s_val if v > 250) / n * 100, 2),
        "gt_500_pct": round(sum(1 for v in s_val if v > 500) / n * 100, 2),
        "gt_1000_pct": round(sum(1 for v in s_val if v > 1000) / n * 100, 2),
    }

def run_stage46_experiment(
    exp_name: str,
    enable_exp_d: bool = False,
    enable_exp_f: bool = False,
    d_token_strategy: str = "first",
    d_max_posting_threshold: int = 500,
    max_s1_eval: int = 100000,
    gt_map: dict = None
):
    print(f"\n" + "=" * 75)
    print(f"=== EVALUATING EXPERIMENT: {exp_name} ===")
    print("=" * 75)

    process = psutil.Process()
    rss_start = process.memory_info().rss / (1024 * 1024)

    # 1. Build Index
    index = Stage46ExperimentalIndex(
        prefix_len=4,
        enable_exp_d=enable_exp_d,
        enable_exp_f=enable_exp_f,
        d_token_strategy=d_token_strategy,
        d_max_posting_threshold=d_max_posting_threshold,
        f_max_posting_threshold=500
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

    # Posting stats for Exp D & F
    d_post_stats = index.get_posting_list_stats(index.exp_d_index) if enable_exp_d else {}
    f_post_stats = index.get_posting_list_stats(index.exp_f_index) if enable_exp_f else {}

    # 2. Evaluate S1 Queries
    eval_start = time.time()
    total_s1 = 0
    total_true_matches = 0
    blocking_recovered = 0

    cand_counts = []

    # Overlap diagnostics
    recovered_set = set()  # set of (s1_id, target_id)

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
                found_matches = true_matches & cands_set
                blocking_recovered += len(found_matches)
                for tm in found_matches:
                    recovered_set.add((s1_id, tm))

    eval_time = time.time() - eval_start
    total_time = index_time + eval_time

    recall = (blocking_recovered / total_true_matches) if total_true_matches > 0 else 0.0
    vol_stats = calculate_percentiles(cand_counts)

    print(f"Total Target Indexed:     {total_target:,} records in {index_time:.2f}s")
    print(f"Total S1 Evaluated:       {total_s1:,} queries in {eval_time:.2f}s")
    print(f"Total Runtime / RAM:      {total_time:.2f}s | RAM RSS: {rss_after_index:.2f} MB")
    print(f"Total True Matches:       {total_true_matches:,}")
    print(f"Blocking Recovered:       {blocking_recovered:,} ({recall * 100:.2f}%)")
    print(f"Candidates Volume:        Avg={vol_stats['avg']}, P50={vol_stats['p50']}, P95={vol_stats['p95']}, P99={vol_stats['p99']}, Max={vol_stats['max']}")
    print(f"Candidate Caps Exceeded:  >25: {vol_stats['gt_25_pct']}%, >50: {vol_stats['gt_50_pct']}%, >100: {vol_stats['gt_100_pct']}%, >500: {vol_stats['gt_500_pct']}%")

    if enable_exp_d:
        print(f"Exp D Posting Stats:     Keys={d_post_stats['unique_keys']:,}, Avg={d_post_stats['avg']}, P95={d_post_stats['p95']}, Max={d_post_stats['max']}")

    return {
        "exp_name": exp_name,
        "recall": recall,
        "recovered": blocking_recovered,
        "total_true": total_true_matches,
        "recovered_set": recovered_set,
        "vol_stats": vol_stats,
        "d_post_stats": d_post_stats,
        "f_post_stats": f_post_stats,
        "total_time": total_time,
        "ram_mb": rss_after_index
    }

def main():
    print("[*] Loading ground truth...")
    gt_map = load_ground_truth(gt_path)

    # 1. Baseline A+B+C
    base = run_stage46_experiment("Baseline A+B+C", False, False, "first", 500, 100000, gt_map)

    # 2. Baseline + F (Street Pair Anchor)
    exp_f = run_stage46_experiment("Baseline + F", False, True, "first", 500, 100000, gt_map)

    # 3. Baseline + D (Name Token Anchor, cap 500)
    exp_d_500 = run_stage46_experiment("Baseline + D (Cap 500)", True, False, "first", 500, 100000, gt_map)

    # 4. Baseline + D + F (Combined Union, Cap 500)
    exp_df_500 = run_stage46_experiment("Baseline + D + F (Cap 500)", True, True, "first", 500, 100000, gt_map)

    # 5. Baseline + D + F (Cap 250)
    exp_df_250 = run_stage46_experiment("Baseline + D + F (Cap 250)", True, True, "first", 250, 100000, gt_map)

    # 6. Baseline + D + F (Cap 100)
    exp_df_100 = run_stage46_experiment("Baseline + D + F (Cap 100)", True, True, "first", 100, 100000, gt_map)

    # 7. Baseline + D + F (Longest Token Strategy, Cap 250)
    exp_df_longest = run_stage46_experiment("Baseline + D + F (Longest Token, Cap 250)", True, True, "longest", 250, 100000, gt_map)

    # --- TRUE MATCH OVERLAP ANALYSIS BETWEEN D AND F ---
    print("\n" + "=" * 80)
    print("=== BASELINE-MISSED TRUE MATCH OVERLAP ANALYSIS (D vs F) ===")
    print("=" * 80)

    base_set = base["recovered_set"]
    d_set = exp_d_500["recovered_set"]
    f_set = exp_f["recovered_set"]

    # Focus on matches missed by baseline
    all_true_pairs = set()
    total_s1 = 0
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if total_s1 >= 100000: break
            if len(row) >= 4:
                s1_id = row[0].strip()
                total_s1 += 1
                for tm in gt_map.get(s1_id, set()):
                    all_true_pairs.add((s1_id, tm))

    missed_by_base = all_true_pairs - base_set
    total_missed = len(missed_by_base)

    d_recovered_missed = d_set & missed_by_base
    f_recovered_missed = f_set & missed_by_base

    d_only = d_recovered_missed - f_recovered_missed
    f_only = f_recovered_missed - d_recovered_missed
    both_df = d_recovered_missed & f_recovered_missed
    neither = missed_by_base - (d_recovered_missed | f_recovered_missed)

    print(f"Total True Match Pairs in 100k Sample: {len(all_true_pairs):,}")
    print(f"Baseline Recovered:                    {len(base_set):,} ({len(base_set)/len(all_true_pairs)*100:.2f}%)")
    print(f"Baseline Missed:                       {total_missed:,} ({total_missed/len(all_true_pairs)*100:.2f}%)\n")

    print(f"Breakdown of {total_missed:,} Baseline-Missed True Matches:")
    print(f"  ├── Recovered by D Only:      {len(d_only):,d} ({len(d_only)/total_missed*100:.2f}%)")
    print(f"  ├── Recovered by F Only:      {len(f_only):,d} ({len(f_only)/total_missed*100:.2f}%)")
    print(f"  ├── Recovered by Both D & F:  {len(both_df):,d} ({len(both_df)/total_missed*100:.2f}%)")
    print(f"  └── Recovered by Neither:     {len(neither):,d} ({len(neither)/total_missed*100:.2f}%)\n")

    # --- COMPARATIVE SUMMARY TABLE ---
    print("\n" + "=" * 115)
    print("=== STAGE 4.6 EXPERIMENTAL COMBINATIONS SUMMARY TABLE (100,000 S1 Sample) ===")
    print("=" * 115)
    print(f"| {'Strategy':36s} | {'Recall':8s} | {'Incr/Base':9s} | {'Incr/F':8s} | {'Incr/D':8s} | {'AvgCand':7s} | {'P95':5s} | {'P99':5s} | {'Max':5s} | {'Runtime':7s} |")
    print("-" * 115)

    experiments = [base, exp_f, exp_d_500, exp_df_500, exp_df_250, exp_df_100, exp_df_longest]

    for exp in experiments:
        rec = exp["recovered"]
        b_rec = base["recovered"]
        f_rec = exp_f["recovered"]
        d_rec = exp_d_500["recovered"]

        inc_base = f"+{rec - b_rec:,}" if rec > b_rec else "—"
        inc_f = f"+{rec - f_rec:,}" if rec > f_rec else ("0" if rec == f_rec else f"-{f_rec - rec:,}")
        inc_d = f"+{rec - d_rec:,}" if rec > d_rec else ("0" if rec == d_rec else f"-{d_rec - rec:,}")

        v = exp["vol_stats"]
        print(f"| {exp['exp_name']:36s} | {exp['recall']*100:7.2f}% | {inc_base:9s} | {inc_f:8s} | {inc_d:8s} | {v['avg']:7.1f} | {v['p95']:5d} | {v['p99']:5d} | {v['max']:5d} | {exp['total_time']:6.1f}s |")

if __name__ == "__main__":
    main()
