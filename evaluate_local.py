import os
import re
import csv
import time
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz, distance
from tqdm import tqdm
import xgboost as xgb

DATA_DIR = "student_resource/dataset/train"

LEGAL_SUFFIXES = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|co|company|enterprises|enterprise|group|services|center|sa|sarl|gmbh)\b',
    re.IGNORECASE
)
PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
DIGITS = re.compile(r'\b\d{4,6}\b')

def clean_name(s: str) -> str:
    if not s: return ""
    s = PUNCT.sub(' ', s.lower())
    s = LEGAL_SUFFIXES.sub(' ', s)
    return " ".join(s.split())

def extract_pin(s: str) -> str:
    if not s: return ""
    m = DIGITS.findall(s)
    return m[0] if m else ""

def extract_features(s1_name, s1_addr, s2_name, s2_addr):
    features = []
    features.append(fuzz.ratio(s1_name, s2_name))
    features.append(fuzz.token_set_ratio(s1_name, s2_name))
    features.append(fuzz.token_sort_ratio(s1_name, s2_name))
    features.append(distance.JaroWinkler.normalized_similarity(s1_name, s2_name) * 100)
    s1_addr_tokens = set(s1_addr.lower().split()) if s1_addr else set()
    s2_addr_tokens = set(s2_addr.lower().split()) if s2_addr else set()
    overlap = len(s1_addr_tokens & s2_addr_tokens)
    features.append(overlap)
    features.append(abs(len(s1_name) - len(s2_name)))
    return features

def calculate_f05(y_true, y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)
    if not true_set and not pred_set:
        return 1.0
    if not true_set or not pred_set:
        return 0.0
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def main():
    print("="*70)
    print("🚀 LOCAL TRAIN ESTIMATION (N=100,000 entities)")
    print("="*70)
    
    print("[*] Loading XGBoost Re-ranker Model...")
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_reranker.json")

    # 1. Load S2 and S3 for indexing
    print("[*] Indexing Target Records (S2 & S3)...")
    idx_exact = defaultdict(list)
    idx_first3 = defaultdict(list)
    idx_pin = defaultdict(list)
    records = {}

    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        path = os.path.join(DATA_DIR, fname)
        print(f"    Reading {fname}...")
        with open(path, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) < 4: continue
                eid, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                cn = clean_name(name)
                pin = extract_pin(addr)
                records[eid] = (cn, addr, country)
                
                if cn:
                    idx_exact[(country, cn)].append(eid)
                    first_3 = cn[:3]
                    if len(first_3) >= 3:
                        idx_first3[(country, first_3)].append(eid)
                if pin:
                    idx_pin[(country, pin)].append(eid)

    # 2. Load Ground Truth
    gt_path = os.path.join(DATA_DIR, "train_ground_truth.tsv")
    ground_truth = {}
    with open(gt_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for i, row in enumerate(reader):
            if i >= 100000: break
            if len(row) < 2: continue
            matches = [x.strip() for x in row[1].split(',')] if row[1].strip() else []
            ground_truth[row[0].strip()] = matches

    # 3. Process S1 and evaluate
    s1_path = os.path.join(DATA_DIR, "train_source1.tsv")
    scores = [[], [], [], [], [], []]
    
    print("[*] Running Pipeline & Evaluating...")
    with open(s1_path, encoding='utf-8') as fin:
        r_s1 = csv.reader(fin, delimiter='\t')
        next(r_s1)
        
        BATCH_SIZE = 5000
        batch_s1 = []
        
        def process_batch(batch):
            feature_matrix = []
            cand_refs = [] # (b_idx, cid)
            
            for b_idx, row in enumerate(batch):
                s1_id = row[0].strip()
                if s1_id not in ground_truth: continue
                
                b_name = row[1].strip() if len(row) > 1 else ""
                b_addr = row[2].strip() if len(row) > 2 else ""
                country = row[3].strip() if len(row) > 3 else ""
                
                cn = clean_name(b_name)
                pin = extract_pin(b_addr)
                
                candidates = set()
                if (country, cn) in idx_exact:
                    candidates.update(idx_exact[(country, cn)][:15])
                if pin and (country, pin) in idx_pin:
                    candidates.update(idx_pin[(country, pin)][:15])
                if len(candidates) < 5 and cn:
                    first_3 = cn[:3]
                    if (country, first_3) in idx_first3:
                        candidates.update(idx_first3[(country, first_3)][:10])
                
                cand_list = list(candidates)
                if not cand_list:
                    f05 = calculate_f05(ground_truth[s1_id], [])
                    scores.append(f05)
                    continue
                    
                for cid in cand_list:
                    t_cn, t_addr, _ = records[cid]
                    feats = extract_features(cn, b_addr, t_cn, t_addr)
                    feature_matrix.append(feats)
                    cand_refs.append((b_idx, cid))
                    
            if not feature_matrix:
                return
                
            X = np.array(feature_matrix)
            dmatrix = xgb.DMatrix(X)
            probs = xgb_model.predict(dmatrix)
            
            best_s2 = [None] * len(batch)
            best_s2_score = [0.0] * len(batch)
            best_s3 = [None] * len(batch)
            best_s3_score = [0.0] * len(batch)
            
            for i, prob in enumerate(probs):
                b_idx, cid = cand_refs[i]
                if cid.startswith("S2-"):
                    if prob > best_s2_score[b_idx]:
                        best_s2_score[b_idx] = prob
                        best_s2[b_idx] = cid
                elif cid.startswith("S3-"):
                    if prob > best_s3_score[b_idx]:
                        best_s3_score[b_idx] = prob
                        best_s3[b_idx] = cid
                        
            for b_idx, row in enumerate(batch):
                s1_id = row[0].strip()
                if s1_id not in ground_truth: continue
                
                for t_idx, thresh in enumerate([0.65, 0.85, 0.90, 0.95, 0.98, 0.99]):
                    final_matches = []
                    if best_s2[b_idx] and best_s2_score[b_idx] >= thresh:
                        final_matches.append(best_s2[b_idx])
                    if best_s3[b_idx] and best_s3_score[b_idx] >= thresh:
                        final_matches.append(best_s3[b_idx])
                        
                    f05 = calculate_f05(ground_truth[s1_id], final_matches)
                    scores[t_idx].append(f05)
        pbar = tqdm(total=100000)
        for row in r_s1:
            if len(scores) >= 100000: break
            batch_s1.append(row)
            if len(batch_s1) >= BATCH_SIZE:
                process_batch(batch_s1)
                pbar.update(len(batch_s1))
                batch_s1 = []
                
        if batch_s1 and len(scores[0]) < 100000:
            process_batch(batch_s1)
            pbar.update(len(batch_s1))
        pbar.close()

    thresholds = [0.65, 0.85, 0.90, 0.95, 0.98, 0.99]
    print("\n[+] ESTIMATED MACRO F0.5 SCORES (XGBoost):")
    for t_idx, thresh in enumerate(thresholds):
        macro_f05 = sum(scores[t_idx]) / len(scores[t_idx]) if scores[t_idx] else 0.0
        print(f"    Threshold {thresh:.2f} -> {macro_f05:.4f}")

if __name__ == "__main__":
    main()
