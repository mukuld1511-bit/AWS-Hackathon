"""
run_final_pipeline.py
=====================
Final End-to-End Entity Resolution Pipeline (Phase 1)
Amazon ML Challenge 2026

Architecture:
  Stage 1: Tight Blocker (src/tight_blocker.py)
           — Avg 5-8 candidates per S1 entity
           — Hierarchical multi-key inverted index
           — Optimized for competition ranking (smaller = better)

  Stage 2: XGBoost Re-Ranker (xgb_reranker_v2.json or xgb_reranker.json)
           — 10 features (7 string + 3 address/geo)
           — Threshold: 0.99 (empirically validated)
           — Hard cap: 1 S2 + 1 S3 per S1 entity

  Stage 3: Multilingual Indic Matcher (output/multilingual_matches.tsv)
           — GPU-accelerated cross-script matching for Indian entities
           — 81% precision (calibrated: Cosine >= 0.50 & Addr Toks >= 7)
           — Overrides XGBoost for Indic cross-script entities

Output:
  output_final/matching_results.tsv   — For Unstop leaderboard upload
  output_final/candidate_pairs.tsv    — For final ranking evaluation
"""
import os
import re
import sys
import csv
import time
import numpy as np
from collections import defaultdict
from rapidfuzz import fuzz, distance
from tqdm import tqdm
import xgboost as xgb

# Import tight blocker
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from tight_blocker import build_indices, get_candidates, clean_name, extract_geo, extract_numbers

DATA_DIR = "student_resource/dataset/test"
MULTILINGUAL_FILE = "output/multilingual_matches.tsv"
OUTPUT_DIR = "output_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model selection: prefer v2 if it exists
if os.path.exists("xgb_reranker_v2.json"):
    XGB_MODEL_FILE = "xgb_reranker_v2.json"
    N_FEATURES = 10
    print("[*] Using XGBoost v2 (10 features)")
else:
    XGB_MODEL_FILE = "xgb_reranker.json"
    N_FEATURES = 6
    print("[!] xgb_reranker_v2.json not found, falling back to v1 (6 features)")

XGB_THRESHOLD = 0.99
MAX_CANDIDATES = 10
BATCH_SIZE = 5000

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID',
    'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO',
    'MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA',
    'RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
}
RE_US_STATE = re.compile(r'\b([A-Z]{2})\b')


def extract_us_state(addr: str) -> str:
    if not addr: return ""
    for m in RE_US_STATE.finditer(addr):
        if m.group(1) in US_STATES:
            return m.group(1)
    return ""


def extract_features(s1_name, s1_addr, s2_name, s2_addr, n_features=6):
    """Feature extraction supporting both 6-feature (v1) and 10-feature (v2) models."""
    features = []
    s1_toks = set(s1_addr.lower().split()) if s1_addr else set()
    s2_toks = set(s2_addr.lower().split()) if s2_addr else set()
    overlap = len(s1_toks & s2_toks)

    if n_features == 6:
        # Exact match for trained xgb_reranker.json (v1)
        features.append(fuzz.ratio(s1_name, s2_name))
        features.append(fuzz.token_set_ratio(s1_name, s2_name))
        features.append(fuzz.token_sort_ratio(s1_name, s2_name))
        features.append(distance.JaroWinkler.normalized_similarity(s1_name, s2_name) * 100)
        features.append(overlap)
        features.append(abs(len(s1_name) - len(s2_name)))
    else:
        # Exact match for train_xgb_v2.py (10 features)
        features.append(fuzz.ratio(s1_name, s2_name))
        features.append(fuzz.token_set_ratio(s1_name, s2_name))
        features.append(fuzz.token_sort_ratio(s1_name, s2_name))
        features.append(fuzz.partial_ratio(s1_name, s2_name))
        features.append(distance.JaroWinkler.normalized_similarity(s1_name, s2_name) * 100)
        features.append(overlap)
        features.append(abs(len(s1_name) - len(s2_name)))
        s1_state = extract_us_state(s1_addr)
        s2_state = extract_us_state(s2_addr)
        features.append(1 if (s1_state and s2_state and s1_state == s2_state) else 0)
        features.append(1 if (s1_state and s2_state and s1_state != s2_state) else 0)
        union_size = len(s1_toks | s2_toks)
        features.append((overlap / union_size * 100) if union_size > 0 else 0.0)

    return features


def load_multilingual_matches(filepath: str) -> dict:
    """Load Multilingual Indic Matcher output. Returns {s1_id: [s2_id, s3_id...]}"""
    matches = {}
    if not os.path.exists(filepath):
        print(f"[!] Multilingual file not found: {filepath} — skipping Indic boost")
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
    print("🚀 FINAL PIPELINE (Tight Blocker + XGBoost v2 + Multilingual Indic)")
    print("=" * 70)

    # === Load XGBoost Model ===
    print(f"[*] Loading XGBoost model: {XGB_MODEL_FILE}...")
    xgb_model = xgb.Booster()
    xgb_model.load_model(XGB_MODEL_FILE)

    # === Load Multilingual Matches ===
    print(f"[*] Loading Multilingual Indic matches...")
    ml_matches = load_multilingual_matches(MULTILINGUAL_FILE)

    # === Build Blocking Indices ===
    print(f"[*] Building tight blocking indices over S2+S3...")
    idx, records = build_indices([
        os.path.join(DATA_DIR, "test_source2.tsv"),
        os.path.join(DATA_DIR, "test_source3.tsv"),
    ])
    print(f"[+] Indexed {len(records):,} S2+S3 target records.")

    # === Process S1 ===
    s1_path = os.path.join(DATA_DIR, "test_source1.tsv")
    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    total_s1 = 0
    matched_count = 0
    ml_boost_count = 0
    xgb_match_count = 0
    total_cands = 0

    print(f"[*] Processing S1 entities...")

    with open(s1_path, encoding='utf-8') as fin, \
         open(matching_file, 'w', encoding='utf-8') as fout_m, \
         open(candidate_file, 'w', encoding='utf-8') as fout_c:

        reader = csv.reader(fin, delimiter='\t')
        next(reader)

        fout_m.write("source1_entity_id\tmatched_entity_ids\n")
        fout_c.write("source1_entity_id\tcandidate_entity_ids\n")

        batch = []

        def process_batch(batch_rows):
            nonlocal total_s1, matched_count, ml_boost_count, xgb_match_count, total_cands

            feature_matrix = []
            cand_refs = []  # (batch_idx, cid)

            for b_idx, row in enumerate(batch_rows):
                s1_id = row[0].strip()
                s1_name = clean_name(row[1].strip() if len(row) > 1 else "")
                s1_addr = row[2].strip() if len(row) > 2 else ""

                # === Candidate Generation (Tight Blocker) ===
                candidates = get_candidates(row, idx, MAX_CANDIDATES)
                total_cands += len(candidates)
                total_s1 += 1

                fout_c.write(f"{s1_id}\t{','.join(candidates)}\n")

                for cid in candidates:
                    if cid not in records: continue
                    t_cn, t_addr, _, _ = records[cid]
                    feats = extract_features(s1_name, s1_addr, t_cn, t_addr, N_FEATURES)
                    feature_matrix.append(feats)
                    cand_refs.append((b_idx, cid))

            if not feature_matrix:
                for row in batch_rows:
                    fout_m.write(f"{row[0].strip()}\t\n")
                return

            # === XGBoost Re-Ranking ===
            X = np.array(feature_matrix, dtype=np.float32)
            dmatrix = xgb.DMatrix(X)
            probs = xgb_model.predict(dmatrix)

            best_s2 = [None] * len(batch_rows)
            best_s2_score = [0.0] * len(batch_rows)
            best_s3 = [None] * len(batch_rows)
            best_s3_score = [0.0] * len(batch_rows)

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

            # === Write Matches (Multilingual overrides XGBoost for Indic) ===
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

                # Priority 2: XGBoost matches (fill any gaps)
                if final_s2 is None and best_s2[b_idx] and best_s2_score[b_idx] >= XGB_THRESHOLD:
                    final_s2 = best_s2[b_idx]
                if final_s3 is None and best_s3[b_idx] and best_s3_score[b_idx] >= XGB_THRESHOLD:
                    final_s3 = best_s3[b_idx]

                final_matches = [m for m in [final_s2, final_s3] if m]
                if final_matches:
                    matched_count += 1
                    if used_ml: ml_boost_count += 1
                    else: xgb_match_count += 1

                fout_m.write(f"{s1_id}\t{','.join(final_matches)}\n")

        pbar = tqdm(total=1_732_544, desc="S1 Entities", unit="ent")
        for row in reader:
            if len(row) < 2: continue
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                process_batch(batch)
                pbar.update(len(batch))
                batch = []
        if batch:
            process_batch(batch)
            pbar.update(len(batch))
        pbar.close()

    elapsed = time.time() - t0
    avg_cands = total_cands / max(total_s1, 1)

    print(f"\n{'='*70}")
    print(f"[+] PIPELINE COMPLETE ({elapsed:.1f}s)")
    print(f"{'='*70}")
    print(f"    Total S1 processed:       {total_s1:,}")
    print(f"    Total matched:            {matched_count:,} ({matched_count/total_s1*100:.1f}%)")
    print(f"    Multilingual Indic boost: {ml_boost_count:,}")
    print(f"    XGBoost matches:          {xgb_match_count:,}")
    print(f"    Avg candidates/entity:    {avg_cands:.2f}")
    print(f"    Output dir:               {OUTPUT_DIR}/")
    print(f"{'='*70}")

    # Run validator
    print("\n[*] Validating submission file...")
    os.system(f"python3 student_resource/utils/validate_submission.py {OUTPUT_DIR}/matching_results.tsv")

    # Copy to sub5_output for easy upload
    print("\n[*] Copying to sub5_output/ for upload...")
    os.makedirs("sub5_output", exist_ok=True)
    os.system(f"cp {OUTPUT_DIR}/matching_results.tsv sub5_output/matching_results.tsv")
    os.system(f"python3 validate_dgx_tsv.py sub5_output/matching_results.tsv")


if __name__ == "__main__":
    main()
