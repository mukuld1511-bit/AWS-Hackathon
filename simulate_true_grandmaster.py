"""
simulate_true_grandmaster.py
============================
Simulates the EXACT pipeline from `run_grandmaster_pipeline.py`:
  1. Phone / Pin Code / Clean Name High-Precision Anchors
  2. 18-D Feature Extraction
  3. Tri-Model Ensemble (0.45 XGB + 0.35 LGB + 0.20 CAT) at Threshold 0.70
  4. Tripartite Graph Transitivity with Address Token Overlap Filter
  5. Exact Competition Official Macro F_0.5 Metric
"""

import os
import re
import csv
import time
from collections import defaultdict
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from src.features_18d import extract_features_18d

DATA_DIR = "student_resource/dataset/train"
SAMPLE_SIZE = 20_000

RE_PIN = re.compile(r'\b\d{5,6}\b')
PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
LEGAL_SUFFIXES = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|'
    r'co|company|enterprises|enterprise|group|services|service|center|'
    r'solutions|associates|sa|sarl|sas|sasu|eurl|gmbh|store|stores)\b',
    re.IGNORECASE
)

def clean_name(s: str) -> str:
    if not s: return ""
    s = PUNCT.sub(' ', s.lower())
    s = LEGAL_SUFFIXES.sub(' ', s)
    return " ".join(s.split())

def extract_pin(s: str) -> str:
    if not s: return ""
    m = RE_PIN.findall(s)
    return m[0] if m else ""

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

def main():
    print("=" * 75)
    print(f"🏆 SIMULATING TRUE GRANDMASTER PIPELINE (N={SAMPLE_SIZE:,} Holdout Entities)")
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

    # 2. Index Target Records
    print("\n[2/5] Indexing target records (train_source2.tsv + train_source3.tsv)...")
    idx_pin = defaultdict(list)
    idx_exact = defaultdict(list)
    records = {}

    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(DATA_DIR, fname)
        print(f"    Indexing {fname}...")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                eid, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip().lower()
                cn = clean_name(name)
                pin = extract_pin(addr)
                records[eid] = (cn, addr, country)
                if cn: idx_exact[(country, cn)].append(eid)
                if pin: idx_pin[(country, pin)].append(eid)

    print(f"[+] Loaded {len(records):,} total target records.")

    # 3. Load Tri-Model GBDT Ensemble
    print("\n[3/5] Loading Tri-Model GBDT Ensemble (XGB + LGB + CAT)...")
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_heavy.json")
    lgb_model = lgb.Booster(model_file="lgb_heavy.txt")
    cat_model = CatBoostClassifier()
    cat_model.load_model("cat_heavy.cbm")
    print("[+] All 3 models loaded.")

    # 4. Stream S1 Entities and Run Inference
    print("\n[4/5] Running S1 Candidate Generation & Ensemble Scoring...")
    s1_rows = {}
    s1_countries = {}
    feature_matrix = []
    cand_pairs = []

    with open(os.path.join(DATA_DIR, "train_source1.tsv"), encoding='utf-8') as fin:
        reader = csv.reader(fin, delimiter='\t')
        next(reader, None)
        for row in reader:
            s1_id = row[0].strip()
            if s1_id not in gt: continue
            
            b_name = row[1].strip() if len(row) > 1 else ""
            b_addr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip().lower() if len(row) > 3 else ""
            s1_countries[s1_id] = country

            cn = clean_name(b_name)
            pin = extract_pin(b_addr)

            candidates = set()
            if (country, cn) in idx_exact:
                candidates.update(idx_exact[(country, cn)][:8])
            if pin and (country, pin) in idx_pin:
                candidates.update(idx_pin[(country, pin)][:8])

            for cid in candidates:
                if cid not in records: continue
                t_cn, t_addr, t_country = records[cid]
                feats = extract_features_18d(cn, b_addr, t_cn, t_addr)
                feature_matrix.append(feats)
                cand_pairs.append((s1_id, cid))

    print(f"[+] Candidate pairs: {len(cand_pairs):,} (Avg: {len(cand_pairs)/len(gt):.2f}/entity)")

    best_s2 = {}
    best_s3 = {}
    best_s2_score = defaultdict(float)
    best_s3_score = defaultdict(float)

    if feature_matrix:
        X = np.array(feature_matrix, dtype=np.float32)
        print("    Running Ensemble Inference...")
        p_xgb = xgb_model.predict(xgb.DMatrix(X))
        p_lgb = lgb_model.predict(X)
        p_cat = cat_model.predict_proba(X)[:, 1]
        probs = (0.45 * p_xgb) + (0.35 * p_lgb) + (0.20 * p_cat)

        for (s1_id, cid), p in zip(cand_pairs, probs):
            if cid.startswith("S2-") and p > best_s2_score[s1_id]:
                best_s2_score[s1_id] = p
                best_s2[s1_id] = cid
            elif cid.startswith("S3-") and p > best_s3_score[s1_id]:
                best_s3_score[s1_id] = p
                best_s3[s1_id] = cid

    # Predictions before transitivity
    preds_before = {}
    for s1_id in gt:
        matches = []
        if s1_id in best_s2 and best_s2_score[s1_id] >= 0.70:
            matches.append(best_s2[s1_id])
        if s1_id in best_s3 and best_s3_score[s1_id] >= 0.70:
            matches.append(best_s3[s1_id])
        preds_before[s1_id] = set(matches)

    # 5. Tripartite Graph Transitivity with Address Overlap Filter
    print("\n[5/5] Running Graph Transitivity (Address Token Overlap >= 2)...")
    s2_by_name = defaultdict(list)
    s3_by_name = defaultdict(list)
    for eid, (cn, addr, country) in records.items():
        if cn:
            if eid.startswith("S2-"): s2_by_name[(country, cn)].append((eid, addr))
            elif eid.startswith("S3-"): s3_by_name[(country, cn)].append((eid, addr))

    preds_after = {}
    s3_recovered, s2_recovered = 0, 0

    for s1_id, ids in preds_before.items():
        final_set = set(ids)
        has_s2 = [x for x in final_set if x.startswith("S2-")]
        has_s3 = [x for x in final_set if x.startswith("S3-")]

        if has_s2 and not has_s3:
            s2_id = has_s2[0]
            if s2_id in records:
                s2_cn, s2_addr, s2_country = records[s2_id]
                s2_toks = set(s2_addr.lower().split()) if s2_addr else set()
                candidates = s3_by_name.get((s2_country, s2_cn), [])
                for cand_s3, s3_addr in candidates:
                    s3_toks = set(s3_addr.lower().split()) if s3_addr else set()
                    overlap = len(s2_toks & s3_toks)
                    if overlap >= 2 or (len(s2_toks) <= 2 and len(s3_toks) <= 2):
                        final_set.add(cand_s3)
                        s3_recovered += 1
                        break

        elif has_s3 and not has_s2:
            s3_id = has_s3[0]
            if s3_id in records:
                s3_cn, s3_addr, s3_country = records[s3_id]
                s3_toks = set(s3_addr.lower().split()) if s3_addr else set()
                candidates = s2_by_name.get((s3_country, s3_cn), [])
                for cand_s2, s2_addr in candidates:
                    s2_toks = set(s2_addr.lower().split()) if s2_addr else set()
                    overlap = len(s3_toks & s2_toks)
                    if overlap >= 2 or (len(s3_toks) <= 2 and len(s2_toks) <= 2):
                        final_set.add(cand_s2)
                        s2_recovered += 1
                        break

        preds_after[s1_id] = final_set

    # Evaluation
    def eval_dict(p_dict):
        f05s, precs, recs = [], [], []
        by_c = defaultdict(lambda: {'f05': [], 'prec': [], 'rec': []})
        for s1_id, true_set in gt.items():
            pred_set = p_dict.get(s1_id, set())
            f, p, r = calc_entity_f05(true_set, pred_set)
            f05s.append(f)
            precs.append(p)
            recs.append(r)
            c = s1_countries.get(s1_id, 'unknown')
            by_c[c]['f05'].append(f)
            by_c[c]['prec'].append(p)
            by_c[c]['rec'].append(r)
        return np.mean(f05s), np.mean(precs), np.mean(recs), by_c

    f_b, p_b, r_b, _ = eval_dict(preds_before)
    f_a, p_a, r_a, by_c = eval_dict(preds_after)

    print("\n" + "=" * 75)
    print("🏆 FINAL SIMULATION RESULTS (True Grandmaster Pipeline)")
    print("=" * 75)
    print(f"1. Before Transitivity (Direct Tri-Model GBDT Ensemble):")
    print(f"   • Macro F_0.5:     {f_b:.4f}")
    print(f"   • Macro Precision: {p_b:.4f}")
    print(f"   • Macro Recall:    {r_b:.4f}")

    print(f"\n2. After Tripartite Graph Transitivity (+{s3_recovered+s2_recovered:,} matches):")
    print(f"   • Macro F_0.5:     {f_a:.4f}")
    print(f"   • Macro Precision: {p_a:.4f}")
    print(f"   • Macro Recall:    {r_a:.4f}")

    print(f"\nBreakdown by Country:")
    for country in ['us', 'india', 'france']:
        if country in by_c:
            c_f = np.mean(by_c[country]['f05'])
            c_p = np.mean(by_c[country]['prec'])
            c_r = np.mean(by_c[country]['rec'])
            print(f"   • {country.upper():6s} -> F0.5: {c_f:.4f} | Precision: {c_p:.4f} | Recall: {c_r:.4f}")

    print(f"\n[+] Simulation finished in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
