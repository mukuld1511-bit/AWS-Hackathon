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
SAMPLE_SIZE = 5000

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

def evaluate(gt_dict, preds_dict):
    f05_list, prec_list, rec_list = [], [], []
    for s1_id, true_set in gt_dict.items():
        pred_set = preds_dict.get(s1_id, set())
        f05, prec, rec = calc_entity_f05(true_set, pred_set)
        f05_list.append(f05)
        prec_list.append(prec)
        rec_list.append(rec)
    return float(np.mean(f05_list)), float(np.mean(prec_list)), float(np.mean(rec_list))

def main():
    print(f"Loading {SAMPLE_SIZE} GT samples...")
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

    print("Indexing target records (train S2 + S3)...")
    idx, records = tb2.build_indices([
        os.path.join(DATA_DIR, "train_source2.tsv"),
        os.path.join(DATA_DIR, "train_source3.tsv"),
    ])

    print("Loading models...")
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_heavy.json")
    xgb_model.set_param({"predictor": "gpu_predictor"})

    lgb_model = lgb.Booster(model_file="lgb_heavy.txt")
    cat_model = CatBoostClassifier()
    cat_model.load_model("cat_heavy.cbm")

    print("Extracting candidates for S1 queries...")
    s1_rows = []
    with open(os.path.join(DATA_DIR, "train_source1.tsv"), encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if row[0] in gt:
                s1_rows.append(row)
                if len(s1_rows) >= SAMPLE_SIZE: break

    all_feats = []
    candidate_features = defaultdict(list)
    entity_slice = []

    for row in s1_rows:
        s1_id, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
        cn = super_clean_name(name)
        geo = tb2.extract_geo(addr, country)
        state = extract_us_state(addr) if country == 'US' else ''

        cands = tb2.get_candidates(cn, addr, country, idx)
        start_idx = len(all_feats)
        for cid in cands:
            if cid not in records: continue
            t_cn, t_addr, t_country, t_geo = records[cid]
            t_state = extract_us_state(t_addr) if t_country == 'US' else ''

            is_conflict = False
            if state and t_state and state != t_state:
                is_conflict = True
            elif geo and t_geo and geo != t_geo:
                is_conflict = True

            feats = extract_features_18d(
                cn, addr, t_cn, t_addr,
                s1_geo=geo, s2_geo=t_geo,
                s1_state=state, s2_state=t_state
            )
            all_feats.append(feats)
            candidate_features[s1_id].append((cid, feats, is_conflict))
        end_idx = len(all_feats)
        entity_slice.append((s1_id, start_idx, end_idx))

    print(f"Scoring {len(all_feats):,} candidate pairs...")
    X_all = np.array(all_feats, dtype=np.float32)
    p_xgb = xgb_model.predict(xgb.DMatrix(X_all))
    p_lgb = lgb_model.predict(X_all)
    p_cat = cat_model.predict_proba(X_all)[:, 1]
    all_probs = 0.40 * p_xgb + 0.35 * p_lgb + 0.25 * p_cat

    scored_candidates = defaultdict(list)
    for s1_id, start, end in entity_slice:
        c_list = candidate_features.get(s1_id, [])
        for i, (cid, _, is_conflict) in enumerate(c_list):
            prob = float(all_probs[start + i])
            if is_conflict:
                prob = 0.0
            scored_candidates[s1_id].append((cid, prob))

    print("\n--- EXPERIMENT RESULTS ---")
    # 1. Old Way: Strictly 1 S2 and 1 S3
    for thresh in [0.65, 0.70, 0.75, 0.80]:
        preds_1to1 = {}
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
            preds_1to1[s1_id] = p_set
        f05, prec, rec = evaluate(gt, preds_1to1)
        print(f"[Strict 1-S2, 1-S3] Thresh {thresh:.2f} -> Macro F0.5: {f05:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    # 2. Multi-match: Allow all matches above threshold
    for thresh in [0.70, 0.75, 0.80, 0.85, 0.90]:
        preds_multi = {}
        for s1_id in gt:
            p_set = set()
            for cid, prob in scored_candidates.get(s1_id, []):
                if prob >= thresh:
                    p_set.add(cid)
            preds_multi[s1_id] = p_set
        f05, prec, rec = evaluate(gt, preds_multi)
        print(f"[Multi-Match]      Thresh {thresh:.2f} -> Macro F0.5: {f05:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

    # 3. Hybrid Multi-match: Best match >= base_thresh (0.65), and additional matches >= high_thresh
    for high_thresh in [0.75, 0.80, 0.85, 0.90]:
        preds_hybrid = {}
        for s1_id in gt:
            cands = sorted(scored_candidates.get(s1_id, []), key=lambda x: x[1], reverse=True)
            p_set = set()
            # Group by source
            s2_cands = [c for c in cands if c[0].startswith("S2-")]
            s3_cands = [c for c in cands if c[0].startswith("S3-")]
            
            # S2: best if >= 0.65, others if >= high_thresh
            for idx_c, (cid, prob) in enumerate(s2_cands):
                if idx_c == 0 and prob >= 0.65:
                    p_set.add(cid)
                elif idx_c > 0 and prob >= high_thresh:
                    p_set.add(cid)

            # S3: best if >= 0.65, others if >= high_thresh
            for idx_c, (cid, prob) in enumerate(s3_cands):
                if idx_c == 0 and prob >= 0.65:
                    p_set.add(cid)
                elif idx_c > 0 and prob >= high_thresh:
                    p_set.add(cid)

            preds_hybrid[s1_id] = p_set
        f05, prec, rec = evaluate(gt, preds_hybrid)
        print(f"[Hybrid Multi]     High Thresh {high_thresh:.2f} -> Macro F0.5: {f05:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f}")

if __name__ == "__main__":
    main()
