"""
run_heavy_pipeline.py
=====================
Ultra-Heavy Tri-Model Ensemble Pipeline for Business Entity Resolution
Amazon ML Challenge 2026

Models Ensembled:
  1. XGBoost GPU (18-D Depth-wise Trees, CUDA)
  2. LightGBM (18-D Leaf-wise Trees)
  3. CatBoost GPU (18-D Oblivious Trees, CUDA)
  4. Multilingual MiniLM L12 Bi-Encoder (Indic Cross-Script)
  5. Graph Transitivity Inference Engine (Triangle Completion)

Output:
  output_final/matching_results.tsv
  output_final/candidate_pairs.tsv
"""
import os
import re
import sys
import csv
import time
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from tight_blocker_v2 import build_indices, get_candidates_v2, extract_geo, extract_numbers
from features_18d import super_clean_name, extract_features_18d, US_STATES, RE_US_STATE
from graph_transitivity import apply_graph_transitivity

DATA_DIR = "student_resource/dataset/test"
MULTILINGUAL_FILE = "output/multilingual_matches.tsv"
OUTPUT_DIR = "output_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

XGB_MODEL_FILE = "xgb_heavy.json"
LGB_MODEL_FILE = "lgb_heavy.txt"
CAT_MODEL_FILE = "cat_heavy.cbm"

ENSEMBLE_THRESHOLD = 0.70
MAX_CANDIDATES = 8
BATCH_SIZE = 5000


def extract_us_state(addr: str) -> str:
    if not addr: return ""
    for m in RE_US_STATE.finditer(addr):
        if m.group(1) in US_STATES:
            return m.group(1)
    return ""


def load_multilingual_matches(filepath: str) -> dict:
    matches = {}
    if not os.path.exists(filepath):
        print(f"[!] Multilingual file not found: {filepath}")
        return matches
    with open(filepath, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                matches[row[0].strip()] = [m for m in row[1].split(',') if m.strip()]
    print(f"[+] Loaded {len(matches):,} Multilingual Indic matches.")
    return matches


def main():
    t0 = time.time()
    print("=" * 70)
    print("🚀 HEAVY TRI-MODEL PIPELINE (XGBoost GPU + LightGBM + CatBoost GPU)")
    print("=" * 70)

    # 1. Load Trained Models
    print(f"[*] Loading XGBoost model from {XGB_MODEL_FILE}...")
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_FILE)

    print(f"[*] Loading LightGBM model from {LGB_MODEL_FILE}...")
    lgb_model = lgb.Booster(model_file=LGB_MODEL_FILE)

    print(f"[*] Loading CatBoost model from {CAT_MODEL_FILE}...")
    cat_model = CatBoostClassifier()
    cat_model.load_model(CAT_MODEL_FILE)

    # 2. Load Multilingual Matches
    ml_matches = load_multilingual_matches(MULTILINGUAL_FILE)

    # 3. Build Blocker v2 Index over S2 & S3
    s2_file = os.path.join(DATA_DIR, "test_source2.tsv")
    s3_file = os.path.join(DATA_DIR, "test_source3.tsv")
    s1_file = os.path.join(DATA_DIR, "test_source1.tsv")

    cache_file = "cache_blocker_v2.pkl"
    import pickle
    if os.path.exists(cache_file):
        print(f"[*] Loading cached index from {cache_file}...")
        with open(cache_file, 'rb') as f_pkl:
            idx, records = pickle.load(f_pkl)
        print(f"[+] Loaded {len(records):,} records from cache.")
    else:
        print("[*] Building hierarchical inverted index for Test S2 and S3...")
        idx, records = build_indices([s2_file, s3_file])
        print(f"[+] Indexed {len(records):,} records from S2 and S3.")
        print(f"[*] Caching index to {cache_file}...")
        with open(cache_file, 'wb') as f_pkl:
            pickle.dump((idx, records), f_pkl, protocol=pickle.HIGHEST_PROTOCOL)
        print("[+] Index cached.")

    # 4. Process S1 in Batches with Tri-Model Ensemble Re-Ranking
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cand_out_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    match_out_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")

    print(f"[*] Streaming results to {cand_out_file} and {match_out_file}...")

    total_s1 = 0
    total_cands = 0
    matched_count = 0
    ml_boost_count = 0
    ensemble_match_count = 0

    with open(s1_file, encoding='utf-8') as f_s1, \
         open(cand_out_file, 'w', encoding='utf-8', buffering=1024*1024) as fout_c, \
         open(match_out_file, 'w', encoding='utf-8', buffering=1024*1024) as fout_m:

        fout_c.write("source1_entity_id\tcandidate_entity_ids\n")
        fout_m.write("source1_entity_id\tmatched_entity_ids\n")

        reader = csv.reader(f_s1, delimiter='\t')
        next(reader)  # Header

        batch = []

        def process_batch(batch_rows):
            nonlocal total_s1, total_cands, matched_count, ml_boost_count, ensemble_match_count

            feature_matrix = []
            cand_refs = []  # (batch_row_idx, cid, is_conflict)

            for b_idx, s1_row in enumerate(batch_rows):
                s1_id = s1_row[0].strip()
                s1_name = s1_row[1].strip()
                s1_addr = s1_row[2].strip()
                s1_country = s1_row[3].strip().lower()

                s1_cn = super_clean_name(s1_name)
                s1_geo = extract_geo(s1_addr, s1_country)
                s1_state = extract_us_state(s1_addr) if s1_country == "us" else ""

                candidates = get_candidates_v2(s1_row, idx, max_candidates=MAX_CANDIDATES)

                # Ensure multilingual matches are included in candidates
                if s1_id in ml_matches:
                    for mid in ml_matches[s1_id]:
                        if mid not in candidates:
                            candidates.append(mid)

                total_s1 += 1
                total_cands += len(candidates)

                fout_c.write(f"{s1_id}\t{','.join(candidates)}\n")

                for cid in candidates:
                    if cid not in records: continue
                    t_cn, t_addr, t_country, t_geo = records[cid]
                    t_state = extract_us_state(t_addr) if t_country == "us" else ""

                    feats = extract_features_18d(
                        s1_cn, s1_addr,
                        t_cn, t_addr,
                        s1_geo, t_geo,
                        s1_state, t_state
                    )
                    feature_matrix.append(feats)
                    is_conflict = (s1_geo and t_geo and s1_geo != t_geo) or (s1_state and t_state and s1_state != t_state)
                    cand_refs.append((b_idx, cid, is_conflict))

            if not feature_matrix:
                for row in batch_rows:
                    fout_m.write(f"{row[0].strip()}\t\n")
                return

            # === Tri-Model Ensemble Inference ===
            X = np.array(feature_matrix, dtype=np.float32)

            # 1. XGBoost GPU
            dmat = xgb.DMatrix(X)
            probs_xgb = xgb_model.predict(dmat)

            # 2. LightGBM
            probs_lgb = lgb_model.predict(X)

            # 3. CatBoost GPU
            probs_cat = cat_model.predict_proba(X)[:, 1]

            # Weighted Ensemble: 40% XGB + 35% LGB + 25% CAT
            probs_ens = 0.40 * probs_xgb + 0.35 * probs_lgb + 0.25 * probs_cat

            best_s2 = [None] * len(batch_rows)
            best_s2_score = [0.0] * len(batch_rows)
            best_s3 = [None] * len(batch_rows)
            best_s3_score = [0.0] * len(batch_rows)

            for i, prob in enumerate(probs_ens):
                b_idx, cid, is_conflict = cand_refs[i]
                if is_conflict:
                    prob = 0.0

                if cid.startswith("S2-"):
                    if prob > best_s2_score[b_idx]:
                        best_s2_score[b_idx] = prob
                        best_s2[b_idx] = cid
                elif cid.startswith("S3-"):
                    if prob > best_s3_score[b_idx]:
                        best_s3_score[b_idx] = prob
                        best_s3[b_idx] = cid

            # Write Matches
            for b_idx, row in enumerate(batch_rows):
                s1_id = row[0].strip()
                final_s2 = None
                final_s3 = None
                used_ml = False

                # Priority 1: Multilingual Indic matches
                if s1_id in ml_matches:
                    for mid in ml_matches[s1_id]:
                        if mid.startswith("S2-") and final_s2 is None:
                            final_s2 = mid
                            used_ml = True
                        elif mid.startswith("S3-") and final_s3 is None:
                            final_s3 = mid
                            used_ml = True

                # Priority 2: Heavy Ensemble matches
                if final_s2 is None and best_s2[b_idx] and best_s2_score[b_idx] >= ENSEMBLE_THRESHOLD:
                    final_s2 = best_s2[b_idx]
                if final_s3 is None and best_s3[b_idx] and best_s3_score[b_idx] >= ENSEMBLE_THRESHOLD:
                    final_s3 = best_s3[b_idx]

                final_matches = [m for m in [final_s2, final_s3] if m]
                if final_matches:
                    matched_count += 1
                    if used_ml: ml_boost_count += 1
                    else: ensemble_match_count += 1

                fout_m.write(f"{s1_id}\t{','.join(final_matches)}\n")

        LIMIT_S1 = int(os.environ.get("LIMIT_S1", "0"))
        pbar = tqdm(total=LIMIT_S1 if LIMIT_S1 > 0 else 1_732_544, desc="Heavy Ensemble S1", unit="ent")
        for row in reader:
            if len(row) < 2: continue
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                process_batch(batch)
                pbar.update(len(batch))
                batch = []
                if LIMIT_S1 > 0 and total_s1 >= LIMIT_S1:
                    break
        if batch and (LIMIT_S1 == 0 or total_s1 < LIMIT_S1):
            process_batch(batch)
            pbar.update(len(batch))
        pbar.close()

    elapsed = time.time() - t0
    avg_cands = total_cands / max(total_s1, 1)

    print(f"\n{'='*70}")
    print(f"[+] HEAVY TRI-MODEL PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"    Total S1 processed:       {total_s1:,}")
    print(f"    Total matched:            {matched_count:,} ({matched_count/total_s1*100:.1f}%)")
    print(f"    Multilingual Indic boost: {ml_boost_count:,}")
    print(f"    Ensemble GBDT matches:    {ensemble_match_count:,}")
    print(f"    Avg candidates/entity:    {avg_cands:.2f}")
    print(f"{'='*70}")

    # Stage 5: Apply Graph Transitivity & Mutual Consistency
    print("\n[*] Applying Graph Transitivity & Mutual Consistency Engine...")
    match_trans = os.path.join(OUTPUT_DIR, "matching_results_trans.tsv")
    cand_trans = os.path.join(OUTPUT_DIR, "candidate_pairs_trans.tsv")
    apply_graph_transitivity(match_out_file, cand_out_file, DATA_DIR, match_trans, cand_trans)
    os.replace(match_trans, match_out_file)
    os.replace(cand_trans, cand_out_file)

    # Stage 6: Run Official Validator
    print("\n[*] Validating submission with official validator...")
    os.system(f"python3 student_resource/utils/validate_submission.py --matching {match_out_file} --candidate {cand_out_file} --test-dir {DATA_DIR} --check-ids")

    # Stage 7: Package into final sub
    print("\n[*] Building final submission package...")
    os.system("python3 build_final_submission.py")

if __name__ == "__main__":
    main()
