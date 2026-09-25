"""
Stage 4.7 Failure Analysis & Targeted Key Discovery Benchmark Script.

Executes:
1. Failure cohort extraction on 100,000 S1 evaluation sample under reference A+B+C+D+F (D cap 250).
2. Failure category classification & diagnostic signal breakdown.
3. Recurring structural pattern identification with representative examples.
4. Evaluation of targeted candidate keys (G: Name Token Pair, H: Second Name Token, I: Address Token Pair).
5. Incremental recall recovery analysis on the remaining failure cohort.
6. Candidate volume percentiles, posting-list statistics, runtime & RAM measurement.
"""

import os
import sys
import csv
import time
import psutil
from collections import Counter, defaultdict

sys.path.insert(0, "/home/piet/Desktop/aws model 2")
from src.failure_analysis import compute_pair_diagnostics, classify_failure_categories
from src.targeted_keys import Stage47TargetedIndex
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
    print("=== STAGE 4.7 FAILURE ANALYSIS & TARGETED KEY DISCOVERY BENCHMARK ===", flush=True)
    print("===========================================================================\n", flush=True)

    print("[*] Loading ground truth mapping...", flush=True)
    gt_map = load_ground_truth(gt_path)

    print("[*] Loading target entity catalog (S2 + S3)...", flush=True)
    target_catalog = {}
    target_start = time.time()
    for file_path in [s2_path, s3_path]:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    e_id = row[0].strip()
                    target_catalog[e_id] = (row[1], row[2], row[3].strip())
    print(f"    Loaded {len(target_catalog):,} target entities in {time.time()-target_start:.2f}s", flush=True)

    # Load 100k S1 records into memory
    print("[*] Loading 100,000 S1 records...", flush=True)
    s1_records = []
    total_true_matches = 0
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for i, row in enumerate(reader):
            if i >= 100000:
                break
            if len(row) >= 4:
                s1_id = row[0].strip()
                s1_name, s1_addr, s1_ctry = row[1], row[2], row[3].strip()
                true_targets = gt_map.get(s1_id, set())
                s1_records.append((s1_id, s1_name, s1_addr, s1_ctry, true_targets))
                total_true_matches += len(true_targets)

    print(f"    Loaded {len(s1_records):,} S1 evaluation records ({total_true_matches:,} true matches)", flush=True)

    # Experiments list
    experiments = [
        ("Ref (A+B+C + D + F cap 250)", False, False, False, 500, 250, 500),
        ("Ref + Key G (Name Token Pair)", True, False, False, 500, 250, 500),
        ("Ref + Key H (Second Name Token)", False, True, False, 500, 250, 500),
        ("Ref + Key I (Address Token Pair)", False, False, True, 500, 250, 500),
        ("Ref + G + H + I Combined", True, True, True, 500, 250, 500),
    ]

    ref_recovered_count = 0
    ref_failures = []

    for exp_idx, (exp_name, en_g, en_h, en_i, g_cap, h_cap, i_cap) in enumerate(experiments):
        print(f"\n" + "="*75, flush=True)
        print(f"=== EVALUATING EXPERIMENT: {exp_name} ===", flush=True)
        print("="*75, flush=True)

        t0 = time.time()
        process = psutil.Process()

        idx = Stage47TargetedIndex(
            enable_exp_d=True,
            enable_exp_f=True,
            enable_exp_g=en_g,
            enable_exp_h=en_h,
            enable_exp_i=en_i,
            d_max_posting_threshold=250,
            f_max_posting_threshold=500,
            g_max_posting_threshold=g_cap,
            h_max_posting_threshold=h_cap,
            i_max_posting_threshold=i_cap
        )

        for e_id, (name, addr, ctry) in target_catalog.items():
            idx.add_record(e_id, name, addr, ctry)

        recovered_count = 0
        cand_counts = []

        for s1_id, s1_n, s1_a, s1_c, true_targets in s1_records:
            evidence = idx.query_record(s1_n, s1_a, s1_c)
            retrieved = set(evidence.keys())
            cand_counts.append(len(retrieved))

            for tgt_id in true_targets:
                if tgt_id in retrieved:
                    recovered_count += 1
                elif exp_idx == 0:
                    tgt_info = target_catalog.get(tgt_id)
                    if tgt_info:
                        diag = compute_pair_diagnostics(s1_n, s1_a, s1_c, tgt_info[0], tgt_info[1], tgt_info[2])
                        ref_failures.append((s1_id, tgt_id, (s1_n, s1_a, s1_c), tgt_info, diag))

        recall = round(recovered_count / total_true_matches * 100, 2)
        stats = calculate_percentiles(cand_counts)
        elapsed = round(time.time() - t0, 2)
        ram_gb = round(process.memory_info().rss / (1024 * 1024 * 1024), 2)

        if exp_idx == 0:
            ref_recovered_count = recovered_count
            print(f"Reference Recovered         : {recovered_count:,} / {total_true_matches:,} ({recall}%)", flush=True)
            print(f"Failure Cohort Count        : {len(ref_failures):,} ({100-recall:.2f}% missed)", flush=True)
        else:
            inc = recovered_count - ref_recovered_count
            pct_fail = round(inc / len(ref_failures) * 100, 2)
            print(f"Recall                      : {recall}%", flush=True)
            print(f"Recovered True Matches      : {recovered_count:,} / {total_true_matches:,}", flush=True)
            print(f"Incremental Matches vs Ref  : +{inc:,} (+{pct_fail}% of remaining failures)", flush=True)

        print(f"Avg Candidates / S1         : {stats['avg']}", flush=True)
        print(f"P50 Candidates / S1         : {stats['p50']}", flush=True)
        print(f"P95 Candidates / S1         : {stats['p95']}", flush=True)
        print(f"P99 Candidates / S1         : {stats['p99']}", flush=True)
        print(f"Max Candidates / S1         : {stats['max']}", flush=True)
        print(f"% >25 Candidates            : {stats['gt_25_pct']}%", flush=True)
        print(f"% >50 Candidates            : {stats['gt_50_pct']}%", flush=True)
        print(f"% >100 Candidates           : {stats['gt_100_pct']}%", flush=True)
        print(f"% >250 Candidates           : {stats['gt_250_pct']}%", flush=True)
        print(f"% >500 Candidates           : {stats['gt_500_pct']}%", flush=True)
        print(f"% >1000 Candidates          : {stats['gt_1000_pct']}%", flush=True)
        print(f"Runtime                     : {elapsed}s", flush=True)
        print(f"Peak RAM                    : {ram_gb} GB", flush=True)

        # Print failure category summary on Reference run
        if exp_idx == 0:
            print("\n[*] Classifying failure categories...", flush=True)
            category_counts = Counter()
            pattern_examples = defaultdict(list)
            for s1_id, tgt_id, (s1_n, s1_a, s1_c), (tgt_n, tgt_a, tgt_c), diag in ref_failures:
                cats = classify_failure_categories(diag)
                for cat in cats:
                    category_counts[cat] += 1
                    if len(pattern_examples[cat]) < 2:
                        pattern_examples[cat].append((s1_id, tgt_id, s1_n, tgt_n, s1_a, tgt_a))

            print("\n--- FAILURE CATEGORY DISTRIBUTION ---", flush=True)
            tot_f = len(ref_failures)
            for cat, count in category_counts.most_common():
                pct = round(count / tot_f * 100, 2)
                print(f"  {cat:<42}: {count:>6,} ({pct:>5.2f}%)", flush=True)

            print("\n--- REPRESENTATIVE FAILURE EXAMPLES ---", flush=True)
            for cat, ex_list in pattern_examples.items():
                print(f"\nCategory: {cat}", flush=True)
                for s1_id, tgt_id, s1_n, tgt_n, s1_a, tgt_a in ex_list:
                    print(f"  S1 [{s1_id}]: '{s1_n}' | Addr: '{s1_a}'", flush=True)
                    print(f"  Tgt[{tgt_id}]: '{tgt_n}' | Addr: '{tgt_a}'", flush=True)

if __name__ == "__main__":
    main()
