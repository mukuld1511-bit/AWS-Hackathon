"""
run_grandmaster_v3_titan.py
===========================
Titan V3 Pipeline: High-Recall Multi-Key Blocking + 18-D Tri-Model GBDT + Global DSU Solver
Target: 0.85+ - 0.95 Macro F0.5
Amazon ML Challenge 2026
"""

import os
import re
import csv
import sys
import time
from collections import defaultdict
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from tqdm import tqdm

from src.features_18d import (
    super_clean_name,
    extract_features_18d,
    US_STATES,
    RE_US_STATE
)
from src.deterministic_anchors import extract_all_deterministic_anchors
import src.tight_blocker_v2 as tb2
from src.global_dsu_solver import solve_global_assignments

DATA_DIR = "student_resource/dataset/test"
OUTPUT_DIR = "output_titan"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CANDIDATE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

def extract_us_state(addr: str) -> str:
    if not addr: return ""
    for m in RE_US_STATE.finditer(addr):
        if m.group(1) in US_STATES:
            return m.group(1)
    return ""

def main():
    print("=" * 80)
    print("🚀 EXECUTING TITAN V3 PIPELINE (HIGH RECALL + TRI-MODEL GBDT + GLOBAL DSU)")
    print("=" * 80)
    t0 = time.time()

    # 1. Load Pretrained Heavy GBDT Models
    print("[1/5] Loading 18-D Tri-Model GBDT Ensemble (XGB GPU + LGB + CAT GPU)...")
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_heavy.json")
    lgb_model = lgb.Booster(model_file="lgb_heavy.txt")
    cat_model = CatBoostClassifier()
    cat_model.load_model("cat_heavy.cbm")
    print("[+] All 3 GBDT Models Ready.")

    # 2. Multi-Key Inverted Indexing of S2 & S3 Target Records
    print("\n[2/5] Building Multi-Key Inverted Index with Deterministic Anchors (S2 & S3)...")
    idx_tax = defaultdict(list)
    idx_phone = defaultdict(list)
    idx_domain = defaultdict(list)
    idx_postal = defaultdict(list)
    idx_exact = defaultdict(list)
    idx_cond = defaultdict(list)
    idx_geo_w1 = defaultdict(list)
    idx_post_w1 = defaultdict(list)

    records = {} # eid -> (cn, addr, country, geo, state)
    weighted_edges = []

    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(DATA_DIR, fname)
        print(f"    Scanning & Indexing {fname}...")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                eid, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip().lower()
                cn = super_clean_name(name)
                cond = cn.replace(' ', '')
                words = cn.split()
                w1 = words[0] if words else ""
                geo = tb2.extract_geo(addr, country)
                state = extract_us_state(addr) if country == "us" else ""
                records[eid] = (cn, addr, country, geo, state)

                # Deterministic Anchors
                anchors = extract_all_deterministic_anchors(name, addr, country)
                for t_key in ['GSTIN', 'PAN', 'CIN', 'SIREN', 'EIN']:
                    if t_key in anchors:
                        idx_tax[(country, t_key, anchors[t_key])].append(eid)
                if 'PHONE' in anchors:
                    idx_phone[(country, anchors['PHONE'])].append(eid)
                if 'DOMAIN' in anchors:
                    idx_domain[anchors['DOMAIN']].append(eid)
                postal = anchors.get('POSTAL', '')
                if postal:
                    idx_postal[(country, postal)].append(eid)

                # Multi-Key Blocking
                if cn and len(cn) >= 3:
                    idx_exact[(country, cn)].append(eid)
                if cond and len(cond) >= 4:
                    idx_cond[(country, cond)].append(eid)
                if geo and w1 and len(w1) >= 3:
                    idx_geo_w1[(country, geo, w1)].append(eid)
                if postal and w1 and len(w1) >= 3:
                    idx_post_w1[(country, postal, w1)].append(eid)

                # S2 <-> S3 Transitive Bridging
                if fname == "test_source3.tsv":
                    if cn and (country, cn) in idx_exact:
                        for s2_cand in idx_exact[(country, cn)][:3]:
                            if s2_cand.startswith("S2-"):
                                weighted_edges.append((s2_cand, eid, 0.95))
                    elif cond and (country, cond) in idx_cond:
                        for s2_cand in idx_cond[(country, cond)][:2]:
                            if s2_cand.startswith("S2-"):
                                weighted_edges.append((s2_cand, eid, 0.92))
                    for t_key in ['GSTIN', 'PAN', 'CIN', 'SIREN', 'EIN']:
                        if t_key in anchors:
                            for s2_cand in idx_tax.get((country, t_key, anchors[t_key]), []):
                                if s2_cand.startswith("S2-"):
                                    weighted_edges.append((s2_cand, eid, 0.999))
                    if 'PHONE' in anchors:
                        for s2_cand in idx_phone.get((country, anchors['PHONE']), []):
                            if s2_cand.startswith("S2-"):
                                weighted_edges.append((s2_cand, eid, 0.99))

    print(f"[+] Loaded {len(records):,} total target records.")
    print(f"    Exact Name Keys: {len(idx_exact):,} | Condensed Keys: {len(idx_cond):,}")
    print(f"    Postal Keys: {len(idx_postal):,} | Geo+Word Keys: {len(idx_geo_w1):,}")
    print(f"    Direct S2 <-> S3 Transitive Edges: {len(weighted_edges):,}")

    # 3. Stream S1 Entities with Multi-Key Retrieval + 18-D Feature Extraction
    print("\n[3/5] Querying S1 Entities with High-Recall Multi-Key Blocker...")
    s1_path = os.path.join(DATA_DIR, "test_source1.tsv")
    cand_pairs_out = {}
    total_s1 = 0
    deterministic_hits = 0
    batch_s1 = []

    def process_s1_batch(batch):
        nonlocal deterministic_hits
        feature_matrix = []
        gbdt_pairs = []

        for row in batch:
            s1_id = row[0].strip()
            name = row[1].strip() if len(row) > 1 else ""
            addr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip().lower() if len(row) > 3 else ""

            cn = super_clean_name(name)
            cond = cn.replace(' ', '')
            words = cn.split()
            w1 = words[0] if words else ""
            geo = tb2.extract_geo(addr, country)
            state = extract_us_state(addr) if country == "us" else ""

            anchors = extract_all_deterministic_anchors(name, addr, country)
            postal = anchors.get('POSTAL', '')

            cands = set()

            # Phase B: Deterministic Anchors
            for t_key in ['GSTIN', 'PAN', 'CIN', 'SIREN', 'EIN']:
                if t_key in anchors:
                    for cid in idx_tax.get((country, t_key, anchors[t_key]), []):
                        weighted_edges.append((s1_id, cid, 0.999))
                        cands.add(cid)
                        deterministic_hits += 1

            if 'DOMAIN' in anchors:
                for cid in idx_domain.get(anchors['DOMAIN'], []):
                    weighted_edges.append((s1_id, cid, 0.995))
                    cands.add(cid)
                    deterministic_hits += 1

            if 'PHONE' in anchors:
                for cid in idx_phone.get((country, anchors['PHONE']), []):
                    weighted_edges.append((s1_id, cid, 0.985))
                    cands.add(cid)
                    deterministic_hits += 1

            # Multi-Key Candidate Retrieval (Up to 15 candidates total)
            if (country, cn) in idx_exact:
                cands.update(idx_exact[(country, cn)][:6])
            if (country, cond) in idx_cond:
                cands.update(idx_cond[(country, cond)][:4])
            if postal and (country, postal, w1) in idx_post_w1:
                cands.update(idx_post_w1[(country, postal, w1)][:3])
            if geo and (country, geo, w1) in idx_geo_w1:
                cands.update(idx_geo_w1[(country, geo, w1)][:3])
            if postal and (country, postal) in idx_postal:
                cands.update(idx_postal[(country, postal)][:3])

            cand_list = list(cands)
            cand_pairs_out[s1_id] = cand_list

            for cid in cand_list:
                if cid not in records: continue
                t_cn, t_addr, t_country, t_geo, t_state = records[cid]
                feats = extract_features_18d(
                    cn, addr, t_cn, t_addr,
                    s1_geo=geo, s2_geo=t_geo,
                    s1_state=state, s2_state=t_state
                )
                feature_matrix.append(feats)
                gbdt_pairs.append((s1_id, cid))

        if feature_matrix:
            X = np.array(feature_matrix, dtype=np.float32)
            prob_xgb = xgb_model.predict(xgb.DMatrix(X))
            prob_lgb = lgb_model.predict(X)
            prob_cat = cat_model.predict_proba(X)[:, 1]
            probs = (0.45 * prob_xgb) + (0.35 * prob_lgb) + (0.20 * prob_cat)

            for (s1_id, cid), p in zip(gbdt_pairs, probs):
                if p >= 0.65:
                    weighted_edges.append((s1_id, cid, float(p)))

    with open(s1_path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin, delimiter='\t')
        next(reader, None)
        pbar = tqdm(total=1732544, desc="Titan S1 Queries")
        for row in reader:
            total_s1 += 1
            batch_s1.append(row)
            if len(batch_s1) >= 5000:
                process_s1_batch(batch_s1)
                pbar.update(len(batch_s1))
                batch_s1 = []
        if batch_s1:
            process_s1_batch(batch_s1)
            pbar.update(len(batch_s1))
        pbar.close()

    print(f"\n[+] Total Deterministic Anchor Hits: {deterministic_hits:,}")
    print(f"[+] Total High-Confidence Weighted Edges: {len(weighted_edges):,}")

    # 4. Global DSU Solver (Optimal 1-S2 / 1-S3 Hungarian assignment + Transitivity)
    print("\n[4/5] Solving Global DSU Assignments & Graph Transitivity...")
    final_assignments = solve_global_assignments(weighted_edges, min_threshold=0.65)
    print(f"[+] Total S1 Successfully Matched: {len(final_assignments):,} / {total_s1:,} ({len(final_assignments)/total_s1*100:.1f}%)")

    # 5. Write Compliant Submission TSVs
    print("\n[5/5] Writing Strict Submission Files...")
    with open(s1_path, 'r', encoding='utf-8') as fin, \
         open(OUT_MATCHING, 'w', encoding='utf-8', newline='') as fout_m, \
         open(OUT_CANDIDATE, 'w', encoding='utf-8', newline='') as fout_c:

        wm = csv.writer(fout_m, delimiter='\t', lineterminator='\n')
        wc = csv.writer(fout_c, delimiter='\t', lineterminator='\n')

        wm.writerow(["source1_entity_id", "matched_entity_ids"])
        wc.writerow(["source1_entity_id", "candidate_entity_ids"])

        reader = csv.reader(fin, delimiter='\t')
        next(reader, None)
        for row in reader:
            s1_id = row[0].strip()
            matches = final_assignments.get(s1_id, [])
            cands = cand_pairs_out.get(s1_id, [])
            all_cands = list(dict.fromkeys(matches + cands))

            wm.writerow([s1_id, ",".join(matches)])
            wc.writerow([s1_id, ",".join(all_cands)])

    # 6. Verification Validation
    print("\nRunning Official Validation Checks...")
    os.system(f"python3 validate_dgx_tsv.py {OUT_MATCHING}")
    os.system(f"python3 student_resource/utils/validate_submission.py -m {OUT_MATCHING} -c {OUT_CANDIDATE} -t student_resource/dataset/test")
    print(f"\n🎉 TITAN V3 PIPELINE READY: {OUT_MATCHING}")
    print(f"Total pipeline execution time: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
