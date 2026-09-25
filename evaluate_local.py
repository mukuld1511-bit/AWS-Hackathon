import os
import re
import csv
import time
from collections import defaultdict
from rapidfuzz import fuzz
from tqdm import tqdm

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

def calculate_f05(y_true, y_pred):
    true_set = set(y_true)
    pred_set = set(y_pred)
    
    # If both are empty (Singleton)
    if not true_set and not pred_set:
        return 1.0
        
    # If one is empty but not the other
    if not true_set or not pred_set:
        return 0.0
        
    tp = len(true_set & pred_set)
    if tp == 0:
        return 0.0
        
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05

def main():
    print("="*70)
    print("🚀 LOCAL TRAIN ESTIMATION (N=100,000 entities)")
    print("="*70)

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

    # 2. Load Ground Truth for 100k entities
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
    scores = []
    
    print("[*] Running Pipeline & Evaluating...")
    with open(s1_path, encoding='utf-8') as fin:
        r_s1 = csv.reader(fin, delimiter='\t')
        next(r_s1)
        
        for i, row in enumerate(tqdm(r_s1, total=100000)):
            if i >= 100000: break
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
            
            best_s2, best_s2_score = None, 0.0
            best_s3, best_s3_score = None, 0.0
            
            for cid in candidates:
                t_cn, t_addr, _ = records[cid]
                score = (fuzz.ratio(cn, t_cn) * 0.6) + (fuzz.token_set_ratio(cn, t_cn) * 0.4)
                if cid.startswith("S2-"):
                    if score > best_s2_score:
                        best_s2_score = score
                        best_s2 = cid
                elif cid.startswith("S3-"):
                    if score > best_s3_score:
                        best_s3_score = score
                        best_s3 = cid
            
            # Predict using Threshold 85.0
            final_matches = []
            if best_s2 and best_s2_score >= 85.0:
                final_matches.append(best_s2)
            if best_s3 and best_s3_score >= 85.0:
                final_matches.append(best_s3)
                
            # Evaluate
            y_true = ground_truth[s1_id]
            f05 = calculate_f05(y_true, final_matches)
            scores.append(f05)

    macro_f05 = sum(scores) / len(scores)
    print(f"\n[+] ESTIMATED MACRO F0.5 SCORE (Threshold 85.0): {macro_f05:.4f}")

if __name__ == "__main__":
    main()
