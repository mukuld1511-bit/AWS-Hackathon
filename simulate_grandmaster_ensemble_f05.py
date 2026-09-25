"""
simulate_grandmaster_ensemble_f05.py
====================================
Simulate Macro F_0.5 Score on Holdout Ground Truth using:
  1. Ultra-Tight Blocker v2 (Multi-Key Hierarchical Indexing + super_clean_name)
  2. 18-Dimensional Feature Extraction (Levenshtein, Jaro-Winkler, LCS, Jaccard, Geo, State, Numbers)
  3. Heavy Tri-Model GBDT Ensemble (XGBoost GPU + LightGBM + CatBoost GPU)
  4. Tripartite Graph Transitivity & Cycle Completion
  5. Exact Official Competition Macro F_0.5 Metric
"""

import os
import csv
import time
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from collections import defaultdict

from src.features_18d import (
    super_clean_name,
    extract_features_18d,
    US_STATES,
    RE_US_STATE
)
import src.tight_blocker_v2 as tb2

DATA_DIR = "student_resource/dataset/train"
SAMPLE_SIZE = 10_000

def extract_us_state(addr: str) -> str:
    if not addr: return ""
    for m in RE_US_STATE.finditer(addr):
        if m.group(1) in US_STATES:
            return m.group(1)
    return ""

def calc_entity_f05(true_set, pred_set):
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

def evaluate_predictions(gt_dict, preds_dict, s1_countries):
    f05_list, prec_list, rec_list = [], [], []
    by_country = defaultdict(lambda: {'f05': [], 'prec': [], 'rec': []})

    for s1_id, true_set in gt_dict.items():
        pred_set = preds_dict.get(s1_id, set())
        f05, prec, rec = calc_entity_f05(true_set, pred_set)
        f05_list.append(f05)
        prec_list.append(prec)
        rec_list.append(rec)

        c = s1_countries.get(s1_id, 'unknown')
        by_country[c]['f05'].append(f05)
        by_country[c]['prec'].append(prec)
        by_country[c]['rec'].append(rec)

    macro_f05 = float(np.mean(f05_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))

    return macro_f05, macro_prec, macro_rec, by_country

def main():
    print("=" * 75)
    print(f"🚀 GRANDMASTER TRI-MODEL ENSEMBLE F_0.5 SIMULATOR (N={SAMPLE_SIZE:,} Holdout)")
    print("=" * 75)
    t0 = time.time()

    # 1. Load Ground Truth
    print("[1/5] Loading Ground Truth holdout...")
    gt = {}
    with open(os.path.join(DATA_DIR, "train_ground_truth.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, row in enumerate(reader):
            if i >= SAMPLE_SIZE: break
            if len(row) >= 2:
                s1_id = row[0].strip()
                matches = set(x.strip() for x in row[1].split(',') if x.strip())
                gt[s1_id] = matches

    print(f"[+] Loaded {len(gt):,} Ground Truth entities.")
    matched_gt = sum(1 for v in gt.values() if v)
    print(f"    GT Distribution: {matched_gt:,} with matches ({matched_gt/len(gt)*100:.1f}%), {len(gt)-matched_gt:,} singletons ({(len(gt)-matched_gt)/len(gt)*100:.1f}%)")

    # 2. Build Inverted Index using Blocker v2
    print("\n[2/5] Indexing target records (train_source2.tsv + train_source3.tsv)...")
    idx, records = tb2.build_indices([
        os.path.join(DATA_DIR, "train_source2.tsv"),
        os.path.join(DATA_DIR, "train_source3.tsv")
    ])
    print(f"[+] Total target records indexed: {len(records):,}")

    # 3. Load Tri-Model GBDT Ensemble
    print("\n[3/5] Loading Heavy Tri-Model GBDT Ensemble...")
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_heavy.json")

    lgb_model = lgb.Booster(model_file="lgb_heavy.txt")

    cat_model = CatBoostClassifier()
    cat_model.load_model("cat_heavy.cbm")
    print("[+] XGBoost (GPU), LightGBM, and CatBoost (GPU) models loaded.")

    # 4. Stream S1 Entities and Run Inference
    print("\n[4/5] Running Blocker v2 + 18-D Feature Extraction + Ensemble Scoring...")
    s1_rows = {}
    s1_countries = {}
    candidate_features = {} # s1_id -> list of (cid, feats, is_conflict)

    with open(os.path.join(DATA_DIR, "train_source1.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id not in gt: continue
            
            s1_rows[s1_id] = row
            s1_name = row[1].strip() if len(row) > 1 else ""
            s1_addr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip().lower() if len(row) > 3 else ""
            s1_countries[s1_id] = country

            s1_cn = super_clean_name(s1_name)
            s1_geo = tb2.extract_geo(s1_addr, country)
            s1_state = extract_us_state(s1_addr) if country == "us" else ""

            cands = tb2.get_candidates_v2(row, idx, max_candidates=10)
            
            cand_info = []
            for cid in cands:
                if cid not in records: continue
                t_cn, t_addr, t_country, t_geo = records[cid]
                t_state = extract_us_state(t_addr) if t_country == "us" else ""

                feats = extract_features_18d(
                    s1_cn, s1_addr, t_cn, t_addr,
                    s1_geo=s1_geo, s2_geo=t_geo,
                    s1_state=s1_state, s2_state=t_state
                )
                
                # Strong conflict shield
                is_conflict = (s1_geo and t_geo and s1_geo != t_geo) or (s1_state and t_state and s1_state != t_state)
                cand_info.append((cid, feats, is_conflict))

            candidate_features[s1_id] = cand_info

    # Batch ensemble prediction
    all_feats = []
    entity_slice = [] # (s1_id, start_idx, end_idx)
    curr = 0
    for s1_id in gt:
        c_list = candidate_features.get(s1_id, [])
        for cid, f, conf in c_list:
            all_feats.append(f)
        entity_slice.append((s1_id, curr, curr + len(c_list)))
        curr += len(c_list)

    print(f"[+] Total candidate pairs extracted: {len(all_feats):,} (Avg: {len(all_feats)/len(gt):.2f}/entity)")

    if all_feats:
        X_all = np.array(all_feats, dtype=np.float32)
        print("    Running XGBoost prediction...")
        p_xgb = xgb_model.predict(xgb.DMatrix(X_all))
        print("    Running LightGBM prediction...")
        p_lgb = lgb_model.predict(X_all)
        print("    Running CatBoost prediction...")
        p_cat = cat_model.predict_proba(X_all)[:, 1]

        # Weighted Tri-Model Ensemble
        all_probs = 0.40 * p_xgb + 0.35 * p_lgb + 0.25 * p_cat
    else:
        all_probs = np.array([])

    # Map predictions back
    scored_candidates = defaultdict(list)
    for s1_id, start, end in entity_slice:
        c_list = candidate_features.get(s1_id, [])
        for i, (cid, _, is_conflict) in enumerate(c_list):
            prob = float(all_probs[start + i])
            if is_conflict:
                prob = 0.0 # Strict geo/state conflict rejection
            scored_candidates[s1_id].append((cid, prob))

    # Evaluate multiple thresholds
    print("\n" + "=" * 75)
    print("🎯 THRESHOLD SWEEP ON TRI-MODEL ENSEMBLE (Before & After Transitivity)")
    print("=" * 75)

    # Pre-index S2 and S3 by (country, super_clean_name) for Transitivity
    s2_by_name = defaultdict(list)
    s3_by_name = defaultdict(list)
    for eid, (cn, addr, country, geo) in records.items():
        if cn:
            if eid.startswith("S2-"): s2_by_name[(country, cn)].append((eid, addr))
            elif eid.startswith("S3-"): s3_by_name[(country, cn)].append((eid, addr))

    best_thresh = 0.65
    best_final_f05 = 0.0

    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        preds_before = {}
        for s1_id in gt:
            best_s2, best_s2_p = None, 0.0
            best_s3, best_s3_p = None, 0.0
            for cid, prob in scored_candidates.get(s1_id, []):
                if prob >= thresh:
                    if cid.startswith("S2-") and prob > best_s2_p:
                        best_s2, best_s2_p = cid, prob
                    elif cid.startswith("S3-") and prob > best_s3_p:
                        best_s3, best_s3_p = cid, prob
            p_set = set()
            if best_s2: p_set.add(best_s2)
            if best_s3: p_set.add(best_s3)
            preds_before[s1_id] = p_set

        f05_b, prec_b, rec_b, _ = evaluate_predictions(gt, preds_before, s1_countries)

        # Graph Transitivity
        preds_after = {}
        trans_recovered = 0
        for s1_id, p_set in preds_before.items():
            final_set = set(p_set)
            has_s2 = [x for x in final_set if x.startswith("S2-")]
            has_s3 = [x for x in final_set if x.startswith("S3-")]

            if has_s2 and not has_s3:
                s2_id = has_s2[0]
                if s2_id in records:
                    cn, addr, country, geo = records[s2_id]
                    hits = s3_by_name.get((country, cn), [])
                    if hits:
                        final_set.add(hits[0][0])
                        trans_recovered += 1

            elif has_s3 and not has_s2:
                s3_id = has_s3[0]
                if s3_id in records:
                    cn, addr, country, geo = records[s3_id]
                    hits = s2_by_name.get((country, cn), [])
                    if hits:
                        final_set.add(hits[0][0])
                        trans_recovered += 1

            preds_after[s1_id] = final_set

        f05_a, prec_a, rec_a, country_breakdown = evaluate_predictions(gt, preds_after, s1_countries)

        print(f"Threshold = {thresh:.2f}:")
        print(f"   Before Transitivity -> F0.5: {f05_b:.4f} | Prec: {prec_b:.4f} | Rec: {rec_b:.4f}")
        print(f"   After Transitivity  -> F0.5: {f05_a:.4f} | Prec: {prec_a:.4f} | Rec: {rec_a:.4f}  (+{trans_recovered:,} recovered matches)")

        if f05_a > best_final_f05:
            best_final_f05 = f05_a
            best_thresh = thresh

    print("\n" + "=" * 75)
    print(f"🏆 BEST MACRO F_0.5 SCORE: {best_final_f05:.4f} at Threshold {best_thresh:.2f}")
    print("=" * 75)

    # Detailed breakdown of best
    print(f"\nBreakdown by Country (at Threshold {best_thresh:.2f}):")
    for country in ['us', 'india', 'france']:
        if country in country_breakdown:
            c_f05 = float(np.mean(country_breakdown[country]['f05']))
            c_p = float(np.mean(country_breakdown[country]['prec']))
            c_r = float(np.mean(country_breakdown[country]['rec']))
            print(f"   • {country.upper():6s} -> F0.5: {c_f05:.4f} | Precision: {c_p:.4f} | Recall: {c_r:.4f}")

    print(f"\n[+] Total simulation completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
