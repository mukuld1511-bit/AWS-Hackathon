"""
simulate_f05_score.py
=====================
Simulate Macro F_0.5 Score on Training Ground Truth Holdout (N=20,000 S1 entities)
Amazon ML Challenge 2026

Simulates:
1. Tight Blocker + Geo Extraction
2. XGBoost v2 scoring (10 features) + Geo-Conflict Filter
3. Threshold 0.65 + Strict Cardinatlity (Max 1 S2, Max 1 S3)
4. Graph Transitivity (Phase 4)
5. Official Macro F_0.5 Metric Calculation
"""
import os
import csv
import sys
import numpy as np
import xgboost as xgb
from collections import defaultdict
from rapidfuzz import fuzz, distance

# Import pipeline components
from run_final_pipeline import extract_features, N_FEATURES
import src.tight_blocker as tb

DATA_DIR = "student_resource/dataset/train"
SAMPLE_SIZE = 20_000
OUTPUT_MODEL = "xgb_reranker_v2.json"
XGB_THRESHOLD = 0.65

def calc_f05(true_set, pred_set):
    if not true_set and not pred_set:
        return 1.0, 1.0, 1.0
    if not true_set or not pred_set:
        return 0.0, 0.0, 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0, 0.0, 0.0
    prec = tp / len(pred_set)
    rec = tp / len(true_set)
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
    return f05, prec, rec

def main():
    print("=" * 70)
    print(f"📊 SIMULATING MACRO F_0.5 SCORE (N={SAMPLE_SIZE:,} Holdout Entities)")
    print("=" * 70)

    # 1. Load Ground Truth for first N entities
    print("[1/5] Loading Ground Truth holdout...")
    gt = {}
    with open(os.path.join(DATA_DIR, "train_ground_truth.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, row in enumerate(reader):
            if i >= SAMPLE_SIZE: break
            if len(row) >= 2:
                gt[row[0].strip()] = set(row[1].split(',')) if row[1].strip() else set()

    print(f"[+] Loaded {len(gt):,} Ground Truth S1 entities.")
    matched_gt = sum(1 for v in gt.values() if v)
    print(f"    GT Distribution: {matched_gt:,} with matches ({matched_gt/len(gt)*100:.1f}%), {len(gt)-matched_gt:,} singletons ({ (len(gt)-matched_gt)/len(gt)*100:.1f}%)")

    # 2. Build blocking indices over target records
    print("\n[2/5] Indexing target records (train_source2.tsv + train_source3.tsv)...")
    idx, records = tb.build_indices([
        os.path.join(DATA_DIR, "train_source2.tsv"),
        os.path.join(DATA_DIR, "train_source3.tsv")
    ])
    print(f"[+] Indexed {len(records):,} targets.")

    # 3. Load XGBoost Model
    print(f"\n[3/5] Loading XGBoost Model ({OUTPUT_MODEL})...")
    model = xgb.Booster()
    model.load_model(OUTPUT_MODEL)

    # 4. Run Pipeline Inference on Holdout
    print("\n[4/5] Running Pipeline Inference on Holdout...")
    preds_before_trans = {}
    s1_rows = {}
    
    with open(os.path.join(DATA_DIR, "train_source1.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id not in gt: continue
            
            s1_rows[s1_id] = row
            s1_name = tb.clean_name(row[1].strip() if len(row) > 1 else "")
            s1_addr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip().lower() if len(row) > 3 else ""
            s1_geo = tb.extract_geo(s1_addr, country)

            cands = tb.get_candidates(row, idx, max_candidates=10)
            
            feature_matrix = []
            cand_refs = []
            for cid in cands:
                if cid not in records: continue
                t_cn, t_addr, _, t_geo = records[cid]
                feats = extract_features(s1_name, s1_addr, t_cn, t_addr, 10)
                feature_matrix.append(feats)
                is_conflict = (s1_geo and t_geo and s1_geo != t_geo)
                cand_refs.append((cid, is_conflict))

            best_s2, best_s2_p = None, 0.0
            best_s3, best_s3_p = None, 0.0

            if feature_matrix:
                probs = model.predict(xgb.DMatrix(np.array(feature_matrix, dtype=np.float32)))
                for (cid, is_conflict), prob in zip(cand_refs, probs):
                    if is_conflict: prob = 0.0
                    if cid.startswith("S2-") and prob > best_s2_p and prob >= XGB_THRESHOLD:
                        best_s2, best_s2_p = cid, prob
                    elif cid.startswith("S3-") and prob > best_s3_p and prob >= XGB_THRESHOLD:
                        best_s3, best_s3_p = cid, prob

            m_set = set()
            if best_s2: m_set.add(best_s2)
            if best_s3: m_set.add(best_s3)
            preds_before_trans[s1_id] = m_set

    # 5. Apply Graph Transitivity on holdout
    print("\n[5/5] Testing Graph Transitivity on Holdout...")
    # Index S2 and S3 by (country, clean_name)
    s2_by_name = defaultdict(list)
    s3_by_name = defaultdict(list)
    for eid, (cn, addr, country, geo) in records.items():
        if cn:
            if eid.startswith("S2-"): s2_by_name[(country, cn)].append((eid, addr))
            elif eid.startswith("S3-"): s3_by_name[(country, cn)].append((eid, addr))

    preds_after_trans = {}
    trans_recovered = 0

    for s1_id, p_set in preds_before_trans.items():
        final_set = set(p_set)
        has_s2 = [x for x in final_set if x.startswith("S2-")]
        has_s3 = [x for x in final_set if x.startswith("S3-")]

        if has_s2 and not has_s3:
            s2_id = has_s2[0]
            if s2_id in records:
                s2_cn, s2_addr, s2_country, _ = records[s2_id]
                s2_toks = set(s2_addr.lower().split()) if s2_addr else set()
                cands_s3 = s3_by_name.get((s2_country, s2_cn), [])
                for cand_s3, s3_addr in cands_s3:
                    s3_toks = set(s3_addr.lower().split()) if s3_addr else set()
                    overlap = len(s2_toks & s3_toks)
                    if overlap >= 2 or (len(s2_toks) <= 2 and len(s3_toks) <= 2):
                        final_set.add(cand_s3)
                        trans_recovered += 1
                        break

        elif has_s3 and not has_s2:
            s3_id = has_s3[0]
            if s3_id in records:
                s3_cn, s3_addr, s3_country, _ = records[s3_id]
                s3_toks = set(s3_addr.lower().split()) if s3_addr else set()
                cands_s2 = s2_by_name.get((s3_country, s3_cn), [])
                for cand_s2, s2_addr in cands_s2:
                    s2_toks = set(s2_addr.lower().split()) if s2_addr else set()
                    overlap = len(s3_toks & s2_toks)
                    if overlap >= 2 or (len(s3_toks) <= 2 and len(s2_toks) <= 2):
                        final_set.add(cand_s2)
                        trans_recovered += 1
                        break

        preds_after_trans[s1_id] = final_set

    # === Metric Evaluation ===
    def score_dict(p_dict):
        f05_list, prec_list, rec_list = [], [], []
        country_f05 = defaultdict(list)
        for s1_id, true_set in gt.items():
            pred_set = p_dict.get(s1_id, set())
            f, p, r = calc_f05(true_set, pred_set)
            f05_list.append(f)
            prec_list.append(p)
            rec_list.append(r)
            country = s1_rows[s1_id][3].strip().lower() if s1_id in s1_rows and len(s1_rows[s1_id]) > 3 else "unknown"
            country_f05[country].append(f)
        return np.mean(f05_list), np.mean(prec_list), np.mean(rec_list), {c: np.mean(vals) for c, vals in country_f05.items()}

    f05_raw, p_raw, r_raw, c_raw = score_dict(preds_before_trans)
    f05_trans, p_trans, r_trans, c_trans = score_dict(preds_after_trans)

    print("\n" + "=" * 70)
    print("🏆 FINAL SIMULATED METRICS BREAKDOWN (N=20,000 Entities)")
    print("=" * 70)
    print(f"\n1. Before Graph Transitivity (Phases 1-3):")
    print(f"   • Macro F_0.5 Score:  {f05_raw:.4f}")
    print(f"   • Macro Precision:    {p_raw:.4f}")
    print(f"   • Macro Recall:       {r_raw:.4f}")
    for c, val in c_raw.items():
        print(f"     └ {c.upper()} F_0.5:       {val:.4f}")

    print(f"\n2. After Graph Transitivity (Phase 4 - Current Output):")
    print(f"   • Macro F_0.5 Score:  {f05_trans:.4f}")
    print(f"   • Macro Precision:    {p_trans:.4f}")
    print(f"   • Macro Recall:       {r_trans:.4f}")
    for c, val in c_trans.items():
        print(f"     └ {c.upper()} F_0.5:       {val:.4f}")

    delta = f05_trans - f05_raw
    print(f"\n🚀 Graph Transitivity Boost: +{delta:.4f} (+{trans_recovered:,} recovered matches)")
    print("=" * 70)


if __name__ == "__main__":
    main()
