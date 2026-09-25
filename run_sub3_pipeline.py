"""
DGX Fast High-Precision Entity Resolution Pipeline (SUBMISSION 3)
Stage 2: XGBoost Re-ranker (No Dense Embeddings)
"""
import os
import re
import csv
import time
import zipfile
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz, distance
from tqdm import tqdm
import xgboost as xgb

DATA_DIR = "student_resource/dataset"
OUTPUT_DIR = "sub3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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

def main():
    print("="*70)
    print("🚀 STARTING SUBMISSION 3 PIPELINE (XGBoost Re-ranker)")
    print("="*70)

    print("[*] Loading XGBoost Re-ranker Model...")
    xgb_model = xgb.Booster()
    if os.path.exists("xgb_reranker.json"):
        xgb_model.load_model("xgb_reranker.json")
    else:
        print("[!] Warning: xgb_reranker.json not found! XGBoost stage will fail.")

    # STEP 1: INDEX S2 AND S3 BY COUNTRY + NAME PREFIX + PINCODE
    print("[*] Indexing Target Records (S2 & S3)...")
    idx_exact = defaultdict(list)
    idx_first3 = defaultdict(list)
    idx_pin = defaultdict(list)
    records = {}

    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(DATA_DIR, "test", fname)
        if not os.path.exists(path): continue
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

    print(f"[+] Loaded {len(records):,} target records.")

    # STEP 2: PROCESS S1 ENTITIES & GBDT RE-RANK
    s1_path = os.path.join(DATA_DIR, "test", "test_source1.tsv")
    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    print("[*] Matching S1 Entities with XGBoost scoring...")
    matched_count = 0
    total_s1 = 0

    with open(s1_path, encoding='utf-8') as fin, \
         open(matching_file, 'w', encoding='utf-8', newline='') as fout_m, \
         open(candidate_file, 'w', encoding='utf-8', newline='') as fout_c:
        
        r_s1 = csv.reader(fin, delimiter='\t')
        w_m = csv.writer(fout_m, delimiter='\t', lineterminator='\n')
        w_c = csv.writer(fout_c, delimiter='\t', lineterminator='\n')
        
        w_m.writerow(["source1_entity_id", "matched_entity_ids"])
        w_c.writerow(["source1_entity_id", "candidate_entity_ids"])
        
        next(r_s1)
        
        # Batching logic for XGBoost speed
        BATCH_SIZE = 5000
        batch_s1 = []
        
        def process_batch(batch):
            nonlocal matched_count, total_s1
            
            # Extract features for all candidates in batch
            feature_matrix = []
            cand_refs = [] # List of (batch_idx, cid)
            
            for b_idx, row in enumerate(batch):
                total_s1 += 1
                s1_id = row[0].strip()
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
                    w_c.writerow([s1_id, ""])
                    continue
                    
                w_c.writerow([s1_id, ",".join(cand_list)])
                
                for cid in cand_list:
                    t_cn, t_addr, _ = records[cid]
                    feats = extract_features(cn, b_addr, t_cn, t_addr)
                    feature_matrix.append(feats)
                    cand_refs.append((b_idx, cid))
            
            if not feature_matrix:
                # Need to write empty matches for the batch
                for b_idx, row in enumerate(batch):
                    w_m.writerow([row[0].strip(), ""])
                return
                
            # Predict batch
            X = np.array(feature_matrix)
            dmatrix = xgb.DMatrix(X)
            probs = xgb_model.predict(dmatrix)
            
            # Map predictions back to S1 entities
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
                        
            # Write matches (Threshold 0.65 as per DGX Prompt)
            for b_idx, row in enumerate(batch):
                s1_id = row[0].strip()
                final_matches = []
                if best_s2[b_idx] and best_s2_score[b_idx] >= 0.65:
                    final_matches.append(best_s2[b_idx])
                if best_s3[b_idx] and best_s3_score[b_idx] >= 0.65:
                    final_matches.append(best_s3[b_idx])
                    
                if final_matches:
                    matched_count += 1
                    w_m.writerow([s1_id, ",".join(final_matches)])
                else:
                    # If candidates existed but none passed 0.65 threshold
                    w_m.writerow([s1_id, ""])

        pbar = tqdm(total=1732544, desc="Processing S1")
        for row in r_s1:
            batch_s1.append(row)
            if len(batch_s1) >= BATCH_SIZE:
                process_batch(batch_s1)
                pbar.update(len(batch_s1))
                batch_s1 = []
                
        if batch_s1:
            process_batch(batch_s1)
            pbar.update(len(batch_s1))
        pbar.close()

    print(f"\n[+] Processing Completed!")
    print(f"[+] Total S1: {total_s1:,} | Matched: {matched_count:,} ({matched_count/total_s1*100:.2f}%)")

    # STEP 3: CREATE SUBMISSION CODE ZIP
    code_zip = "sub3_code.zip"
    with zipfile.ZipFile(code_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk('src'):
            for file in files:
                full = os.path.join(root, file)
                z.write(full, arcname=os.path.join('code/business_entity_resolution', full))
        if os.path.exists('requirements.txt'):
            z.write('requirements.txt', arcname='code/business_entity_resolution/requirements.txt')
        if os.path.exists('TEAM_INSTRUCTIONS.md'):
            z.write('TEAM_INSTRUCTIONS.md', arcname='code/business_entity_resolution/README.md')
        # Also include the model
        if os.path.exists('xgb_reranker.json'):
            z.write('xgb_reranker.json', arcname='code/business_entity_resolution/xgb_reranker.json')

    print(f"[+] Created {code_zip} ({os.path.getsize(code_zip)} bytes)")

    # STEP 4: RUN OFFICIAL VALIDATION SCRIPT
    print("\n[*] Running Official Submission Validator...")
    os.system(f"python3 student_resource/utils/validate_submission.py --matching {OUTPUT_DIR}/matching_results.tsv --candidate {OUTPUT_DIR}/candidate_pairs.tsv --test-dir student_resource/dataset/test")
    
    print("\n[+] To submit to Unstop:")
    print(f"    Code file: {code_zip}")
    print(f"    Results file: {OUTPUT_DIR}/matching_results.tsv")

if __name__ == "__main__":
    main()
