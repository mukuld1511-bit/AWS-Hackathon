"""
DGX Fast High-Precision Entity Resolution Pipeline
Stage 1: Multi-Key + Dense Semantic Retrieval (SentenceTransformers)
Stage 2: XGBoost Re-ranker
"""
import os
import re
import csv
import time
import zipfile
import numpy as np
import torch
from collections import defaultdict
from rapidfuzz import fuzz, distance
from tqdm import tqdm
import xgboost as xgb
from sentence_transformers import SentenceTransformer

DATA_DIR = "student_resource/dataset"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE_INFERENCE = 5000
EMBED_BATCH_SIZE = 2048

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
    print("🚀 STARTING DGX HYBRID RETRIEVAL + XGBOOST PIPELINE (TARGET: 0.90+)")
    print("="*70)

    print("[*] Loading XGBoost Re-ranker Model...")
    xgb_model = xgb.Booster()
    if os.path.exists("xgb_reranker.json"):
        xgb_model.load_model("xgb_reranker.json")
    else:
        print("[!] Warning: xgb_reranker.json not found! XGBoost stage will fail.")

    print(f"[*] Loading Dense Embedding Model ({EMBEDDING_MODEL})...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embed_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # STEP 1: INDEX S2 AND S3
    print("[*] Indexing Target Records (S2 & S3)...")
    idx_exact = defaultdict(list)
    idx_first3 = defaultdict(list)
    idx_pin = defaultdict(list)
    idx_first_word = defaultdict(list)
    
    records = {}
    target_names = []
    target_eids = []

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
                    first_word = cn.split()[0] if cn.split() else ""
                    if first_word:
                        idx_first_word[(country, first_word)].append(eid)
                if pin:
                    idx_pin[(country, pin)].append(eid)
                    
                target_names.append(cn)
                target_eids.append(eid)

    print(f"[+] Loaded {len(records):,} target records.")
    
    # Generate Embeddings for Target Records (DGX Multi-GPU Acceleration)
    print(f"[*] Generating Dense Embeddings for {len(target_names):,} Targets...")
    target_embeddings = embed_model.encode(
        target_names, 
        batch_size=EMBED_BATCH_SIZE, 
        show_progress_bar=True, 
        convert_to_tensor=True,
        normalize_embeddings=True
    )
    
    # Group target embeddings by country to narrow search space
    print("[*] Partitioning Embeddings by Country for Fast Search...")
    country_tensors = defaultdict(list)
    country_eids = defaultdict(list)
    
    for i, eid in enumerate(tqdm(target_eids, desc="Partitioning")):
        country = records[eid][2]
        country_tensors[country].append(target_embeddings[i])
        country_eids[country].append(eid)
        
    for c in country_tensors:
        country_tensors[c] = torch.stack(country_tensors[c])

    # STEP 2: PROCESS S1 ENTITIES
    s1_path = os.path.join(DATA_DIR, "test", "test_source1.tsv")
    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    print("[*] Matching S1 Entities with Hybrid Retrieval + XGBoost scoring...")
    matched_count = 0
    total_s1 = 0

    with open(s1_path, encoding='utf-8') as fin, \
         open(matching_file, 'w', encoding='utf-8', newline='') as fout_m, \
         open(candidate_file, 'w', encoding='utf-8', newline='') as fout_c:
        
        r_s1 = csv.reader(fin, delimiter='\t')
        w_m = csv.writer(fout_m, delimiter='\t', lineterminator='\n')
        w_c = csv.writer(fout_c, delimiter='\t', lineterminator='\n')
        
        fout_m.write("source1_entity_id\tmatched_entity_ids\n")
        w_c.writerow(["source1_entity_id", "candidate_entity_ids"])
        
        next(r_s1)
        batch_s1 = []
        
        def process_batch(batch):
            nonlocal matched_count, total_s1
            feature_matrix = []
            cand_refs = [] # List of (batch_idx, cid)
            
            # Step 2A: Dense Embedding Query for the whole batch
            batch_cns = [clean_name(row[1].strip() if len(row) > 1 else "") for row in batch]
            s1_embs = embed_model.encode(batch_cns, batch_size=len(batch), convert_to_tensor=True, normalize_embeddings=True)
            
            for b_idx, row in enumerate(batch):
                total_s1 += 1
                s1_id = row[0].strip()
                b_name = row[1].strip() if len(row) > 1 else ""
                b_addr = row[2].strip() if len(row) > 2 else ""
                country = row[3].strip() if len(row) > 3 else ""
                
                cn = batch_cns[b_idx]
                pin = extract_pin(b_addr)
                
                candidates = set()
                
                # Fast Heuristic Blocking
                if (country, cn) in idx_exact:
                    candidates.update(idx_exact[(country, cn)][:15])
                if pin and (country, pin) in idx_pin:
                    candidates.update(idx_pin[(country, pin)][:15])
                if cn:
                    first_word = cn.split()[0] if cn.split() else ""
                    if first_word and (country, first_word) in idx_first_word:
                        candidates.update(idx_first_word[(country, first_word)][:5])
                        
                # Semantic Fallback / Boost (Top 3 Dense Retrieval by Country)
                # ONLY run this expensive operation if fast blocking found 0 candidates!
                if not candidates and country in country_tensors and cn:
                    # Cosine similarity is just dot product for normalized embeddings
                    sims = torch.matmul(country_tensors[country], s1_embs[b_idx])
                    top_k_scores, top_k_idx = torch.topk(sims, k=3)
                    for k_idx, score in zip(top_k_idx, top_k_scores):
                        if score.item() > 0.80:  # Only add high-confidence semantic matches
                            candidates.add(country_eids[country][k_idx.item()])
                        
                cand_list = list(candidates)
                if not cand_list:
                    w_c.writerow([s1_id, ""])
                    continue
                    
                w_c.writerow([s1_id, ",".join(cand_list)])
                
                # Step 2B: XGBoost Feature Extraction
                for cid in cand_list:
                    t_cn, t_addr, _ = records[cid]
                    feats = extract_features(cn, b_addr, t_cn, t_addr)
                    feature_matrix.append(feats)
                    cand_refs.append((b_idx, cid))
            
            if not feature_matrix:
                # Need to write empty matches for the batch
                for b_idx, row in enumerate(batch):
                    fout_m.write(f"{row[0].strip()}\t\n")
                return
                
            # Step 2C: XGBoost Prediction
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
                        
            # Step 2D: Write matches
            for b_idx, row in enumerate(batch):
                s1_id = row[0].strip()
                final_matches = []
                if best_s2[b_idx] and best_s2_score[b_idx] >= 0.85:
                    final_matches.append(best_s2[b_idx])
                if best_s3[b_idx] and best_s3_score[b_idx] >= 0.85:
                    final_matches.append(best_s3[b_idx])
                    
                if final_matches:
                    matched_count += 1
                    fout_m.write(f"{s1_id}\t{','.join(final_matches)}\n")
                else:
                    fout_m.write(f"{s1_id}\t\n")

        pbar = tqdm(total=1732544, desc="Processing S1")
        for row in r_s1:
            batch_s1.append(row)
            if len(batch_s1) >= BATCH_SIZE_INFERENCE:
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
    code_zip = "code.zip"
    with zipfile.ZipFile(code_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, dirs, files in os.walk('src'):
            for file in files:
                full = os.path.join(root, file)
                z.write(full, arcname=os.path.join('code/business_entity_resolution', full))
        if os.path.exists('requirements.txt'):
            z.write('requirements.txt', arcname='code/business_entity_resolution/requirements.txt')
        if os.path.exists('TEAM_INSTRUCTIONS.md'):
            z.write('TEAM_INSTRUCTIONS.md', arcname='code/business_entity_resolution/README.md')
        if os.path.exists('xgb_reranker.json'):
            z.write('xgb_reranker.json', arcname='code/business_entity_resolution/xgb_reranker.json')

    print(f"[+] Created {code_zip} ({os.path.getsize(code_zip)} bytes)")

    # STEP 4: RUN OFFICIAL VALIDATION SCRIPT
    print("\n[*] Running Official Submission Validator...")
    os.system("python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test")

if __name__ == "__main__":
    main()
