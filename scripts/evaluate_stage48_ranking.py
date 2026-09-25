"""
Stage 4.8 Ranking-Only Investigation & Validation Script.

Evaluates controlled ranking configurations on the exact frozen blocking candidate pool:
  A + B + C + D(cap=250) + F(cap=500) + G(cap=500) + I(cap=500)

Evaluates:
- Baseline A/B/C-only ranking (D=F=G=I=0)
- Experiment R1: Equal experimental weights (D=F=G=I in {0.5, 1.0, 1.5, 2.0})
- Experiment R2: Multi-key evidence bonus (bonus in {0.25, 0.50, 1.00})
- Experiment R3: Key-specific weighting based on key precision (D=1.0, F=1.5, G=2.0, I=2.0)
- Recovery of the ~510,513 experimental-only true matches at K=25
- Full-data Recall@10, 15, 20, 25, 30 and remaining failure diagnosis
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


def main():
    print("===========================================================================", flush=True)
    print("=== STAGE 4.8 RANKING-ONLY INVESTIGATION & VALIDATION BENCHMARK ===", flush=True)
    print("===========================================================================\n", flush=True)

    process = psutil.Process()
    t0_start = time.time()

    # 1. Load Ground Truth Mapping
    print("[*] Step 1/4: Loading ground truth mapping...", flush=True)
    gt_map = load_ground_truth(gt_path)
    total_true_matches_all = sum(len(v) for v in gt_map.values())
    print(f"    Loaded ground truth for {len(gt_map):,} S1 entities ({total_true_matches_all:,} true matches)", flush=True)

    # 2. Build Frozen Stage 4.7 Index (A+B+C + D250 + F500 + G500 + I500)
    print("\n[*] Step 2/4: Building Target Candidate Catalog & Frozen Inverted Index...", flush=True)
    t0_idx = time.time()
    index = Stage47TargetedIndex(
        prefix_len=4,
        enable_exp_d=True,
        enable_exp_f=True,
        enable_exp_g=True,
        enable_exp_h=False,
        enable_exp_i=True,
        d_max_posting_threshold=250,
        f_max_posting_threshold=500,
        g_max_posting_threshold=500,
        i_max_posting_threshold=500
    )

    total_target_records = 0
    for file_path in [s2_path, s3_path]:
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for row in reader:
                if len(row) >= 4:
                    index.add_record(row[0].strip(), row[1], row[2], row[3].strip())
                    total_target_records += 1

    idx_time = round(time.time() - t0_idx, 2)
    ram_idx = round(process.memory_info().rss / (1024 * 1024 * 1024), 2)
    print(f"    Indexed {total_target_records:,} target records in {idx_time}s | Peak RAM: {ram_idx} GB", flush=True)

    # 3. Define Ranking Configurations to Evaluate
    ranking_configs = [
        ("Baseline (A=3.0, B=2.0, C=1.5, Exp=0.0)", RankingConfig(3.0, 2.0, 1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 30)),
        ("R1.1 Equal (Exp=0.5)", RankingConfig(3.0, 2.0, 1.5, 0.5, 0.5, 0.5, 0.5, 0.0, 30)),
        ("R1.2 Equal (Exp=1.0)", RankingConfig(3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 0.0, 30)),
        ("R1.3 Equal (Exp=1.5)", RankingConfig(3.0, 2.0, 1.5, 1.5, 1.5, 1.5, 1.5, 0.0, 30)),
        ("R1.4 Equal (Exp=2.0)", RankingConfig(3.0, 2.0, 1.5, 2.0, 2.0, 2.0, 2.0, 0.0, 30)),
        ("R2.1 Multi Bonus 0.25", RankingConfig(3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 0.25, 30)),
        ("R2.2 Multi Bonus 0.50", RankingConfig(3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 0.50, 30)),
        ("R2.3 Multi Bonus 1.00", RankingConfig(3.0, 2.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.00, 30)),
        ("R3.1 Key-Specific (D=1.0, F=1.5, G=2.0, I=2.0)", RankingConfig(3.0, 2.0, 1.5, 1.0, 1.5, 2.0, 2.0, 0.0, 30)),
        ("R3.2 High Pair Anchor (D=1.0, F=1.5, G=2.5, I=2.5)", RankingConfig(3.0, 2.0, 1.5, 1.0, 1.5, 2.5, 2.5, 0.0, 30)),
        ("R3.3 Key-Specific + Multi Bonus 0.5", RankingConfig(3.0, 2.0, 1.5, 1.0, 1.5, 2.0, 2.0, 0.50, 30)),
    ]

    # 4. First Pass: Evaluate on 100,000 S1 Sample for Fast Ranking Sensitivity & Experimental-Only Match Recovery
    print("\n[*] Step 3/4: Evaluating Ranking Configurations on 100,000 S1 Sample...", flush=True)
    
    # Pre-query 100k sample into memory to isolate ranking runtime from index querying
    sample_s1_data = []
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for i, row in enumerate(reader):
            if i >= 100000:
                break
            if len(row) >= 4:
                s1_id = row[0].strip()
                s1_name, s1_addr, s1_ctry = row[1], row[2], row[3].strip()
                true_tgts = gt_map.get(s1_id, set())
                ev_map = index.query_record(s1_name, s1_addr, s1_ctry)
                sample_s1_data.append((s1_id, true_tgts, ev_map))

    sample_total_matches = sum(len(tgts) for _, tgts, _ in sample_s1_data)
    
    # Identify experimental-only true matches in sample (true matches matched ONLY by D, F, G, I)
    sample_exp_only_matches = 0
    sample_uncapped_matches = 0
    for _, tgts, ev_map in sample_s1_data:
        uncapped = set(ev_map.keys()) & tgts
        sample_uncapped_matches += len(uncapped)
        for tgt in uncapped:
            ev = ev_map[tgt]
            if not (ev.matched_name_key or ev.matched_prefix_key or ev.matched_address_key):
                sample_exp_only_matches += 1

    sample_uncapped_recall = round(sample_uncapped_matches / sample_total_matches * 100, 2)
    print(f"    Sample Evaluated          : 100,000 S1 records ({sample_total_matches:,} true matches)", flush=True)
    print(f"    Sample Uncapped Recall    : {sample_uncapped_recall}% ({sample_uncapped_matches:,} recovered)", flush=True)
    print(f"    Experimental-Only Matches : {sample_exp_only_matches:,} true matches", flush=True)

    print("\n--- SAMPLE RANKING EXPERIMENTS (100K S1) ---", flush=True)
    print(f"{'Config Name':<50} | {'Recall@25':<10} | {'Exp-Only Recov':<16} | {'Avg Cands@25':<14}", flush=True)
    print("-" * 96, flush=True)

    sample_results = []
    for cfg_name, cfg in ranking_configs:
        t0_cfg = time.time()
        recov_25 = 0
        exp_only_recov_25 = 0
        retained_counts_25 = []

        for _, tgts, ev_map in sample_s1_data:
            ranked = rank_candidates(ev_map, cfg)
            cands_25 = set(c.entity_id for c in ranked[:25])
            retained_counts_25.append(len(cands_25))

            found = cands_25 & tgts
            recov_25 += len(found)

            for tgt in found:
                ev = ev_map[tgt]
                if not (ev.matched_name_key or ev.matched_prefix_key or ev.matched_address_key):
                    exp_only_recov_25 += 1

        rec_25_pct = round(recov_25 / sample_total_matches * 100, 2)
        exp_rec_pct = round(exp_only_recov_25 / sample_exp_only_matches * 100, 2) if sample_exp_only_matches else 0
        avg_cand_25 = round(sum(retained_counts_25) / len(retained_counts_25), 2)

        sample_results.append((cfg_name, cfg, rec_25_pct, exp_rec_pct, avg_cand_25))
        print(f"{cfg_name:<50} | {rec_25_pct:>9.2f}% | {exp_only_recov_25:>6,} ({exp_rec_pct:>5.1f}%) | {avg_cand_25:>12.2f}", flush=True)

    # Sort configs by sample Recall@25 to select top 3 for Full 2.2M Data Validation
    top_configs = sorted(sample_results, key=lambda x: -x[2])[:3]
    baseline_sample_rec = sample_results[0][2]

    print("\n" + "=" * 80, flush=True)
    print("[*] Step 4/4: Running Full 2.2M Data Validation on Top Ranking Configurations...", flush=True)
    print("=" * 80 + "\n", flush=True)

    full_eval_configs = [
        ("Baseline (A=3.0, B=2.0, C=1.5, Exp=0.0)", ranking_configs[0][1]),
        (top_configs[0][0], top_configs[0][1]),
        (top_configs[1][0], top_configs[1][1]),
    ]

    t0_full = time.time()
    k_vals = [10, 15, 20, 25, 30]

    # Structure per config
    cfg_stats = {}
    for name, cfg in full_eval_configs:
        cfg_stats[name] = {
            "cfg": cfg,
            "recov_k": {k: 0 for k in k_vals},
            "retained_k": {k: [] for k in k_vals},
            "exp_only_recov_25": 0,
            "lost_at_25_reasons": Counter(),
        }

    s1_eval_count = 0
    eval_true_matches = 0
    uncapped_recovered_full = 0

    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)

        for row in reader:
            if len(row) < 4:
                continue
            s1_id = row[0].strip()
            s1_name, s1_addr, s1_ctry = row[1], row[2], row[3].strip()
            s1_eval_count += 1

            true_targets = gt_map.get(s1_id, set())
            eval_true_matches += len(true_targets)

            ev_map = index.query_record(s1_name, s1_addr, s1_ctry)
            uncapped_cands = set(ev_map.keys())
            uncapped_found = true_targets & uncapped_cands
            uncapped_recovered_full += len(uncapped_found)

            for name, st in cfg_stats.items():
                cfg = st["cfg"]
                ranked = rank_candidates(ev_map, cfg)
                ranked_ids = [c.entity_id for c in ranked]

                found_25 = set(ranked_ids[:25]) & true_targets

                for tgt in found_25:
                    ev = ev_map[tgt]
                    if not (ev.matched_name_key or ev.matched_prefix_key or ev.matched_address_key):
                        st["exp_only_recov_25"] += 1

                for k in k_vals:
                    k_found = set(ranked_ids[:k]) & true_targets
                    st["recov_k"][k] += len(k_found)
                    st["retained_k"][k].append(min(len(uncapped_cands), k))

                lost_in_trunc = uncapped_found - found_25
                if lost_in_trunc:
                    for lost_id in lost_in_trunc:
                        ev = ev_map[lost_id]
                        if ev.matched_name_key or ev.matched_prefix_key or ev.matched_address_key:
                            st["lost_at_25_reasons"]["Weak Evidence Score (A/B/C tie-break loss)"] += 1
                        else:
                            st["lost_at_25_reasons"]["Experimental Match Truncated (Lower score/tie-break)"] += 1

            if s1_eval_count % 500000 == 0:
                base_rec = round(cfg_stats[full_eval_configs[0][0]]["recov_k"][25] / eval_true_matches * 100, 2)
                top1_rec = round(cfg_stats[full_eval_configs[1][0]]["recov_k"][25] / eval_true_matches * 100, 2)
                print(f"    Evaluated {s1_eval_count:,} / 2,206,821 S1 records | Baseline R@25: {base_rec}% | {full_eval_configs[1][0][:20]} R@25: {top1_rec}%", flush=True)

    full_runtime = round(time.time() - t0_full, 2)
    full_ram_gb = round(process.memory_info().rss / (1024 * 1024 * 1024), 2)
    uncapped_rec_full_pct = round(uncapped_recovered_full / eval_true_matches * 100, 4)

    for full_cfg_name, _ in full_eval_configs:
        st = cfg_stats[full_cfg_name]
        print(f"\n" + "="*75, flush=True)
        print(f"=== FULL 2.2M DATA VALIDATION SUMMARY: {full_cfg_name} ===", flush=True)
        print("="*75, flush=True)
        print(f"Uncapped Blocking Recall       : {uncapped_rec_full_pct}% ({uncapped_recovered_full:,} / {eval_true_matches:,})", flush=True)

        print(f"\nRecall@K Breakdown:", flush=True)
        for k in k_vals:
            rec_k_pct = round(st["recov_k"][k] / eval_true_matches * 100, 4)
            avg_ret_k = round(sum(st["retained_k"][k]) / len(st["retained_k"][k]), 2)
            print(f"  Recall@{k:<2}                      : {rec_k_pct:>7.4f}% ({st['recov_k'][k]:>12,} matches) | Avg Candidates: {avg_ret_k}", flush=True)

        rec_25_full_pct = round(st["recov_k"][25] / eval_true_matches * 100, 4)
        print(f"\nKey Metric (Recall@25)          : {rec_25_full_pct}% ({st['recov_k'][25]:,} matches)", flush=True)
        print(f"Experimental-Only Matches @25   : {st['exp_only_recov_25']:,} matches recovered", flush=True)

        print(f"\nRemaining Failure Diagnosis at K=25:", flush=True)
        blocking_missed = eval_true_matches - uncapped_recovered_full
        ranking_truncated = uncapped_recovered_full - st["recov_k"][25]
        print(f"  1. Blocking Misses (Not in Pool): {blocking_missed:,} ({round(blocking_missed/eval_true_matches*100, 4)}%)", flush=True)
        print(f"  2. Ranking Truncations (Lost @25): {ranking_truncated:,} ({round(ranking_truncated/eval_true_matches*100, 4)}%)", flush=True)
        for rsn, cnt in st["lost_at_25_reasons"].most_common():
            print(f"     - {rsn:<55}: {cnt:>10,} ({round(cnt/ranking_truncated*100, 2)}%)", flush=True)

    print(f"\nFull 2.2M Evaluation Runtime     : {full_runtime}s ({full_runtime/60:.2f} minutes)", flush=True)
    print(f"Peak RAM                         : {full_ram_gb} GB", flush=True)

    total_pipeline_time = round(time.time() - t0_start, 2)
    print("\n" + "=" * 80, flush=True)
    print(f"=== STAGE 4.8 BENCHMARK COMPLETED IN {total_pipeline_time}s ({total_pipeline_time/60:.2f} min) ===", flush=True)
    print("=" * 80 + "\n", flush=True)


if __name__ == "__main__":
    main()
