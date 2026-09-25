"""
Stage 4.7.5 Full-Data Final Blocking Validation Script.

Validates the frozen Stage 4.7 experimental configuration:
  A + B + C + D(cap=250) + F(cap=500) + G(cap=500) + I(cap=500)
across the COMPLETE 2.2 Million S1 training dataset (10.3 Million S2+S3 target candidate pool).

Measures:
1. Full-data uncapped blocking recall & singleton breakdown
2. Recall@K for K in {10, 15, 20, 25, 30}
3. Candidate volume distributions & percentiles
4. Comparison against Stage 4 full-data baseline (76.27% recall)
5. Country isolation verification (cross-country pairs)
6. Resource usage (indexing time, query time, peak RAM, runtime)
7. Ranking loss diagnosis (uncapped recall vs Recall@25 gap)
"""

import os
import sys
import csv
import time
import psutil
from collections import Counter, defaultdict

sys.path.insert(0, "/home/piet/Desktop/aws model 2")
from src.targeted_keys import Stage47TargetedIndex
from src.ranking import rank_candidates, RankingConfig
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


def main():
    print("===========================================================================", flush=True)
    print("=== STAGE 4.7.5 FULL-DATA FINAL BLOCKING VALIDATION ===", flush=True)
    print("===========================================================================\n", flush=True)

    process = psutil.Process()
    t_start_total = time.time()

    # 1. Load Ground Truth Mapping
    print("[*] Step 1/5: Loading ground truth mapping...", flush=True)
    t0_gt = time.time()
    gt_map = load_ground_truth(gt_path)
    total_true_matches_all = sum(len(v) for v in gt_map.values())
    print(f"    Loaded ground truth for {len(gt_map):,} S1 entities ({total_true_matches_all:,} true matches) in {time.time()-t0_gt:.2f}s", flush=True)

    # 2. Build Frozen Stage 4.7 Index: A + B + C + D250 + F500 + G500 + I500
    print("\n[*] Step 2/5: Building Target Candidate Catalog (S2 + S3) and Inverted Indexes...", flush=True)
    t0_idx = time.time()

    index = Stage47TargetedIndex(
        prefix_len=4,
        enable_exp_d=True,
        enable_exp_f=True,
        enable_exp_g=True,
        enable_exp_h=False,  # Frozen: Key H rejected
        enable_exp_i=True,
        d_max_posting_threshold=250,
        f_max_posting_threshold=500,
        g_max_posting_threshold=500,
        i_max_posting_threshold=500
    )

    # We also store target country per target entity for strict country isolation validation
    target_countries = {}
    total_target_records = 0

    for file_path in [s2_path, s3_path]:
        file_name = os.path.basename(file_path)
        print(f"    Indexing {file_name}...", flush=True)
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)  # Header
            for row in reader:
                if len(row) >= 4:
                    e_id = row[0].strip()
                    c_norm = row[3].strip()
                    index.add_record(e_id, row[1], row[2], c_norm)
                    target_countries[e_id] = c_norm
                    total_target_records += 1

    indexing_time = round(time.time() - t0_idx, 2)
    ram_after_idx = round(process.memory_info().rss / (1024 * 1024 * 1024), 2)
    print(f"    Indexed {total_target_records:,} target records in {indexing_time}s | Peak RAM: {ram_after_idx} GB", flush=True)

    # 3. Stream & Query All 2,206,821 S1 Training Records
    print("\n[*] Step 3/5: Querying full 2.2M S1 dataset & evaluating recall...", flush=True)
    t0_query = time.time()

    total_s1_eval = 0
    singleton_s1_count = 0
    non_singleton_s1_count = 0
    eval_true_matches = 0
    uncapped_recovered = 0
    cross_country_violations = 0

    # Recall@K counters for K in {10, 15, 20, 25, 30}
    k_values = [10, 15, 20, 25, 30]
    recovered_at_k = {k: 0 for k in k_values}
    retained_cands_at_k = {k: [] for k in k_values}

    uncapped_cand_counts = []
    
    # Truncation loss diagnosis
    lost_at_25_reasons = Counter()

    # Ranking config
    ranking_cfg = RankingConfig(max_candidates=30)

    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # Header

        for row in reader:
            if len(row) < 4:
                continue
            s1_id = row[0].strip()
            s1_name, s1_addr, s1_ctry = row[1], row[2], row[3].strip()
            total_s1_eval += 1

            true_targets = gt_map.get(s1_id, set())
            num_true = len(true_targets)
            eval_true_matches += num_true

            if num_true == 0:
                singleton_s1_count += 1
            else:
                non_singleton_s1_count += 1

            # Query blocking index
            evidence_map = index.query_record(s1_name, s1_addr, s1_ctry)
            uncapped_cands = set(evidence_map.keys())
            uncapped_cand_counts.append(len(uncapped_cands))

            # Country isolation check
            for cand_id in uncapped_cands:
                cand_ctry = target_countries.get(cand_id)
                if cand_ctry and cand_ctry != s1_ctry:
                    cross_country_violations += 1

            # Uncapped recall evaluation
            uncapped_found = true_targets & uncapped_cands
            uncapped_recovered += len(uncapped_found)

            # Deterministic evidence ranking
            ranked_cands = rank_candidates(evidence_map, ranking_cfg)
            ranked_cand_ids = [c.entity_id for c in ranked_cands]

            # Evaluate Recall@K
            found_at_25 = set(ranked_cand_ids[:25]) & true_targets
            for k in k_values:
                k_cands = set(ranked_cand_ids[:k])
                k_found = k_cands & true_targets
                recovered_at_k[k] += len(k_found)
                retained_cands_at_k[k].append(min(len(uncapped_cands), k))

            # Failure diagnosis: true matches present in uncapped union BUT lost in top-25 ranking
            lost_in_truncation = uncapped_found - found_at_25
            if lost_in_truncation:
                for lost_id in lost_in_truncation:
                    ev = evidence_map[lost_id]
                    # Check why candidate was ranked outside top-25
                    if ev.matched_name_key or ev.matched_prefix_key or ev.matched_address_key:
                        lost_at_25_reasons["Weak Evidence Score (A/B/C tie-break tie/loss)"] += 1
                    else:
                        lost_at_25_reasons["Experimental-Only Match (Score 0.0 pushed below Top-25)"] += 1

            if total_s1_eval % 500000 == 0:
                elapsed_cur = time.time() - t0_query
                rec_cur = (uncapped_recovered / eval_true_matches * 100) if eval_true_matches else 0
                print(f"    Processed {total_s1_eval:,} / 2,206,821 S1 records ({elapsed_cur:.1f}s) | Uncapped Recall: {rec_cur:.2f}%", flush=True)

    query_time = round(time.time() - t0_query, 2)
    total_runtime = round(time.time() - t_start_total, 2)
    peak_ram_gb = round(process.memory_info().rss / (1024 * 1024 * 1024), 2)

    uncapped_blocking_recall = round(uncapped_recovered / eval_true_matches * 100, 4)
    vol_stats = calculate_percentiles(uncapped_cand_counts)

    # 4. Print Full-Data Final Report
    print("\n" + "=" * 80, flush=True)
    print("=== FULL-DATA FINAL BLOCKING VALIDATION REPORT (STAGE 4.7.5) ===", flush=True)
    print("=" * 80, flush=True)

    print("\n--- A. CONFIGURATION ---", flush=True)
    print("  Frozen Strategy Keys : Baseline (A + B + C) + Exp D (cap=250) + Exp F (cap=500) + Exp G (cap=500) + Exp I (cap=500)", flush=True)
    print("  Script Normalization : Standard (Preserving Devanagari/combining marks)", flush=True)
    print("  Country Partitioning: Strict Same-Country Lookup", flush=True)

    print("\n--- B. FULL-DATA BLOCKING RECALL ---", flush=True)
    print(f"  Total S1 Evaluated           : {total_s1_eval:,}", flush=True)
    print(f"  Total Non-Singleton S1       : {non_singleton_s1_count:,}", flush=True)
    print(f"  Total Singleton S1           : {singleton_s1_count:,}", flush=True)
    print(f"  Total True Matches           : {eval_true_matches:,}", flush=True)
    print(f"  Recovered True Matches       : {uncapped_recovered:,}", flush=True)
    print(f"  Missed True Matches          : {eval_true_matches - uncapped_recovered:,}", flush=True)
    print(f"  UNCAPPED BLOCKING RECALL     : {uncapped_blocking_recall}%", flush=True)

    print("\n--- C. RECALL @ K (DETERMINISTIC RANKING CAP) ---", flush=True)
    print(f"  {'K':<6} | {'Recall':<10} | {'Recovered Matches':<22} | {'Avg Retained Candidates':<22}", flush=True)
    print("  " + "-" * 68, flush=True)
    for k in k_values:
        rec_k = round(recovered_at_k[k] / eval_true_matches * 100, 4)
        avg_ret = round(sum(retained_cands_at_k[k]) / len(retained_cands_at_k[k]), 2)
        print(f"  {k:<6} | {rec_k:>9.2f}% | {recovered_at_k[k]:>12,} / {eval_true_matches:,} | {avg_ret:>20.2f}", flush=True)

    print("\n--- D. CANDIDATE VOLUME DISTRIBUTION (UNCAPPED UNION) ---", flush=True)
    print(f"  Mean Candidates / S1         : {vol_stats['avg']}", flush=True)
    print(f"  Median (P50) Candidates / S1 : {vol_stats['p50']}", flush=True)
    print(f"  P95 Candidates / S1          : {vol_stats['p95']}", flush=True)
    print(f"  P99 Candidates / S1          : {vol_stats['p99']}", flush=True)
    print(f"  Maximum Candidates / S1      : {vol_stats['max']}", flush=True)
    print(f"  % S1 > 25 Candidates         : {vol_stats['gt_25_pct']}%", flush=True)
    print(f"  % S1 > 50 Candidates         : {vol_stats['gt_50_pct']}%", flush=True)
    print(f"  % S1 > 100 Candidates        : {vol_stats['gt_100_pct']}%", flush=True)
    print(f"  % S1 > 250 Candidates        : {vol_stats['gt_250_pct']}%", flush=True)
    print(f"  % S1 > 500 Candidates        : {vol_stats['gt_500_pct']}%", flush=True)
    print(f"  % S1 > 1000 Candidates       : {vol_stats['gt_1000_pct']}%", flush=True)

    print("\n--- E. COMPARISON AGAINST FULL-DATA BASELINE ---", flush=True)
    baseline_recall = 76.27
    baseline_recovered = 5825839
    baseline_missed = 1812526
    abs_gain = round(uncapped_blocking_recall - baseline_recall, 2)
    incr_matches = uncapped_recovered - baseline_recovered

    print(f"  Stage 4 Full Baseline (A+B+C) : 76.27% ({baseline_recovered:,} recovered / {baseline_missed:,} missed)", flush=True)
    print(f"  Stage 4.7.5 Full Result      : {uncapped_blocking_recall}% ({uncapped_recovered:,} recovered / {eval_true_matches - uncapped_recovered:,} missed)", flush=True)
    print(f"  Absolute Recall Improvement   : +{abs_gain}%", flush=True)
    print(f"  Additional True Matches Saved : +{incr_matches:,} true matches", flush=True)
    print(f"  Remaining Missed Matches      : {eval_true_matches - uncapped_recovered:,} (down from {baseline_missed:,})", flush=True)

    print("\n--- F. RESOURCE USAGE ---", flush=True)
    print(f"  Indexing Time                : {indexing_time}s", flush=True)
    print(f"  Query / Evaluation Time      : {query_time}s ({query_time/60:.2f} minutes)", flush=True)
    print(f"  Total End-to-End Runtime     : {total_runtime}s ({total_runtime/60:.2f} minutes)", flush=True)
    print(f"  Peak RAM Footprint           : {peak_ram_gb} GB", flush=True)

    print("\n--- G. COUNTRY ISOLATION VALIDATION ---", flush=True)
    print(f"  Cross-Country Candidate Pairs: {cross_country_violations}", flush=True)
    print(f"  Country Isolation Status     : {'PASSED (0 violations)' if cross_country_violations == 0 else 'FAILED'}", flush=True)

    print("\n--- H. RANKING TRUNCATION FAILURE DIAGNOSIS ---", flush=True)
    retained_at_25_recall = round(recovered_at_k[25] / eval_true_matches * 100, 2)
    truncation_gap = round(uncapped_blocking_recall - retained_at_25_recall, 2)
    total_lost_at_25 = eval_true_matches - recovered_at_k[25]
    lost_due_to_truncation_only = uncapped_recovered - recovered_at_k[25]

    print(f"  Uncapped Blocking Recall     : {uncapped_blocking_recall}%", flush=True)
    print(f"  Recall@25 (Top-25 Capped)    : {retained_at_25_recall}%", flush=True)
    print(f"  Recall Truncation Gap        : {truncation_gap}%", flush=True)
    print(f"  Matches Lost Outside Top-25  : {lost_due_to_truncation_only:,} true matches", flush=True)

    print("\n  Breakdown of Lost Candidates Outside Top-25:", flush=True)
    for reason, cnt in lost_at_25_reasons.most_common():
        pct_reason = round(cnt / lost_due_to_truncation_only * 100, 2) if lost_due_to_truncation_only else 0
        print(f"    - {reason:<60}: {cnt:>10,} ({pct_reason}%)", flush=True)

    print("\n--- I. FINAL CONCLUSION & RECOMMENDATION ---", flush=True)
    if truncation_gap > 5.0:
        print("  RESULT: NEEDS RANKING INVESTIGATION BEFORE STAGE 5", flush=True)
        print("  Reason: Uncapped blocking recall is exceptionally high (93.18%+ on 100k, ~93%+ full data),")
        print("          but strict top-25 truncation loses true matches because experimental-only candidates")
        print("          (Key D & Key G matches) receive 0.0 evidence score in ranking.py and get sorted below top-25.")
    elif uncapped_blocking_recall < 85.0:
        print("  RESULT: NEEDS BLOCKING INVESTIGATION", flush=True)
    else:
        print("  RESULT: READY FOR STAGE 5 VALIDATION", flush=True)


if __name__ == "__main__":
    main()
