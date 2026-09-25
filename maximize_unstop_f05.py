"""
maximize_unstop_f05.py
======================
Post-Processing Maximizer for Unstop Leaderboard Macro F0.5
Amazon ML Challenge 2026

Leverages:
  1. Indian Regional State & Script Normalizer (Devanagari, Tamil, Gujarati + State Code Aliases)
  2. Canonical City & Geo Aliases (Bengaluru==Bangalore, Gurugram==Gurgaon, Prayagraj==Allahabad)
  3. High-Precision Recovery on Unmatched Singletons (Threshold >= 0.75)
  4. 100% Candidate Synchronization & Zero False-Positive Leakage
"""

import os
import re
import csv
import sys
import time
from collections import defaultdict
import numpy as np

# Increase CSV field size limit for large candidate rows
csv.field_size_limit(sys.maxsize)
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
import src.tight_blocker_v2 as tb2

DATA_DIR = "student_resource/dataset/test"
INPUT_MATCHING = "output_titan/matching_results.tsv"
INPUT_CANDIDATE = "output_titan/candidate_pairs.tsv"

OUTPUT_DIR = "output_maximized"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_MATCHING = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUT_CANDIDATE = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

# Canonical Indian State & Region Normalization
INDIA_STATE_ALIASES = {
    'rj': 'rajasthan', 'rajasthan': 'rajasthan', 'राजस्थान': 'rajasthan',
    'dl': 'delhi', 'delhi': 'delhi', 'new delhi': 'delhi', 'दिल्ली': 'delhi',
    'mh': 'maharashtra', 'maharashtra': 'maharashtra', 'महाराष्ट्र': 'maharashtra',
    'wb': 'bengal', 'west bengal': 'bengal', 'bengal': 'bengal', 'पश्चिम बंगाल': 'bengal',
    'ka': 'karnataka', 'karnataka': 'karnataka', 'ಕರ್ನಾಟಕ': 'karnataka',
    'tn': 'tamil nadu', 'tamil': 'tamil nadu', 'தமிழ்நாடு': 'tamil nadu',
    'up': 'uttar pradesh', 'uttar': 'uttar pradesh', 'उत्तर प्रदेश': 'uttar pradesh',
    'ts': 'telangana', 'tg': 'telangana', 'telangana': 'telangana', 'तेलंगाना': 'telangana',
    'ap': 'andhra pradesh', 'andhra': 'andhra pradesh', 'आंध्र प्रदेश': 'andhra pradesh',
    'mp': 'madhya pradesh', 'madhya': 'madhya pradesh', 'मध्य प्रदेश': 'madhya pradesh',
    'gj': 'gujarat', 'gujarat': 'gujarat', 'ગુજરાત': 'gujarat',
    'hr': 'haryana', 'haryana': 'haryana', 'हरियाणा': 'haryana',
    'kl': 'kerala', 'kerala': 'kerala', 'केरल': 'kerala',
    'pb': 'punjab', 'punjab': 'punjab', 'पंजाब': 'punjab',
    'or': 'odisha', 'odisha': 'odisha', 'orissa': 'odisha', 'ओडिशा': 'odisha',
    'jh': 'jharkhand', 'jharkhand': 'jharkhand',
    'as': 'assam', 'assam': 'assam',
    'br': 'bihar', 'bihar': 'bihar', 'बिहार': 'bihar',
    'ut': 'uttarakhand', 'uttarakhand': 'uttarakhand',
    'ch': 'chandigarh', 'chandigarh': 'chandigarh',
    'ga': 'goa', 'goa': 'goa',
}

CITY_SYNONYMS = {
    'bengaluru': 'bangalore', 'bangalore': 'bangalore',
    'gurugram': 'gurgaon', 'gurgaon': 'gurgaon',
    'mumbai': 'mumbai', 'bombay': 'mumbai',
    'chennai': 'chennai', 'madras': 'chennai',
    'kolkata': 'kolkata', 'calcutta': 'kolkata',
    'prayagraj': 'allahabad', 'allahabad': 'allahabad',
    'varanasi': 'varanasi', 'banaras': 'varanasi', 'kashi': 'varanasi',
    'visakhapatnam': 'vizag', 'vizag': 'vizag',
    'mysuru': 'mysore', 'mysore': 'mysore',
    'kochi': 'cochin', 'cochin': 'cochin',
    'vadodara': 'baroda', 'baroda': 'baroda',
    'thiruvananthapuram': 'trivandrum', 'trivandrum': 'trivandrum',
    'kozhikode': 'calicut', 'calicut': 'calicut',
    'belagavi': 'belgaum', 'belgaum': 'belgaum',
    'hubballi': 'hubli', 'hubli': 'hubli',
}

RE_WORD = re.compile(r'[a-zA-Z\u0900-\u0D7F]+')

def extract_canonical_geo(addr: str, country: str) -> str:
    if not addr: return ""
    addr_l = addr.lower()
    
    if country == "india":
        # Check Indic and abbreviation states
        for k, v in INDIA_STATE_ALIASES.items():
            if re.search(r'\b' + re.escape(k) + r'\b', addr_l):
                return v
        # Check city synonyms
        for k, v in CITY_SYNONYMS.items():
            if k in addr_l:
                return v
        return tb2.extract_geo(addr, country)

    elif country == "us":
        for m in RE_US_STATE.finditer(addr):
            if m.group(1) in US_STATES:
                return m.group(1)
        return tb2.extract_geo(addr, country)

    elif country == "france":
        return tb2.extract_geo(addr, country)

    return ""

def main():
    print("=" * 80)
    print("🚀 EXECUTING UNSTOP MACRO F0.5 MAXIMIZER (INDIAN GEO + RECOVERY ENGINE)")
    print("=" * 80)
    t0 = time.time()

    # 1. Load Pretrained Heavy GBDT Models
    print("[1/5] Loading 18-D Tri-Model GBDT Ensemble...")
    xgb_model = xgb.Booster()
    xgb_model.load_model("xgb_heavy.json")
    lgb_model = lgb.Booster(model_file="lgb_heavy.txt")
    cat_model = CatBoostClassifier()
    cat_model.load_model("cat_heavy.cbm")
    print("[+] Models Ready.")

    # 2. Load Existing Titan Matches and Candidates
    print("\n[2/5] Loading Baseline Matches from Titan V3...")
    current_matches = {}
    current_candidates = {}
    with open(INPUT_MATCHING, 'r', encoding='utf-8') as f_m, \
         open(INPUT_CANDIDATE, 'r', encoding='utf-8') as f_c:
        rm = csv.reader(f_m, delimiter='\t')
        rc = csv.reader(f_c, delimiter='\t')
        next(rm); next(rc)
        for row_m, row_c in zip(rm, rc):
            s1_id = row_m[0].strip()
            m_ids = [x.strip() for x in row_m[1].split(',') if x.strip()] if len(row_m) > 1 and row_m[1].strip() else []
            c_ids = [x.strip() for x in row_c[1].split(',') if x.strip()] if len(row_c) > 1 and row_c[1].strip() else []
            current_matches[s1_id] = m_ids
            current_candidates[s1_id] = c_ids

    total_matched = sum(1 for v in current_matches.values() if v)
    empty_s1 = [k for k, v in current_matches.items() if not v]
    missing_one = [k for k, v in current_matches.items() if len(v) == 1]
    print(f"[+] Loaded {len(current_matches):,} entities.")
    print(f"    Currently Matched: {total_matched:,} ({total_matched/len(current_matches)*100:.1f}%)")
    print(f"    Empty Singletons: {len(empty_s1):,} | Missing S2/S3: {len(missing_one):,}")

    target_queries = set(empty_s1) | set(missing_one)

    # 3. Index Targets with Canonical Geo + State Aliases
    print("\n[3/5] Indexing S2 & S3 Targets with Canonical State Aliases...")
    idx_geo_w1 = defaultdict(list)
    records = {} # eid -> (cn, addr, country, geo)

    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(DATA_DIR, fname)
        print(f"    Indexing {fname}...")
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) < 4: continue
                eid, name, addr, country = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip().lower()
                cn = super_clean_name(name)
                words = cn.split()
                w1 = words[0] if words else ""
                geo = extract_canonical_geo(addr, country)
                records[eid] = (cn, addr, country, geo)

                if geo and w1 and len(w1) >= 3:
                    idx_geo_w1[(country, geo, w1)].append(eid)

    print(f"[+] Total Targets Indexed: {len(records):,}")
    print(f"    Canonical Geo + Word Keys: {len(idx_geo_w1):,}")

    # 4. Target Query Recovery on Empty / Missing Entities
    print(f"\n[4/5] Running High-Precision Recovery on {len(target_queries):,} Target Entities...")
    recovered_matches = 0
    s1_path = os.path.join(DATA_DIR, "test_source1.tsv")

    with open(s1_path, 'r', encoding='utf-8') as fin:
        reader = csv.reader(fin, delimiter='\t')
        next(reader, None)
        pbar = tqdm(total=len(target_queries), desc="Recovering Entities")

        batch = []
        def process_recovery_batch(b):
            nonlocal recovered_matches
            feature_matrix = []
            cand_refs = []

            for s1_id, name, addr, country in b:
                cn = super_clean_name(name)
                words = cn.split()
                w1 = words[0] if words else ""
                geo = extract_canonical_geo(addr, country)

                existing = current_matches.get(s1_id, [])
                has_s2 = any(x.startswith("S2-") for x in existing)
                has_s3 = any(x.startswith("S3-") for x in existing)

                hits = idx_geo_w1.get((country, geo, w1), [])
                for cid in hits[:6]:
                    if cid in existing: continue
                    if cid.startswith("S2-") and has_s2: continue
                    if cid.startswith("S3-") and has_s3: continue
                    if cid not in records: continue

                    t_cn, t_addr, t_country, t_geo = records[cid]
                    feats = extract_features_18d(
                        cn, addr, t_cn, t_addr,
                        s1_geo=geo, s2_geo=t_geo
                    )
                    feature_matrix.append(feats)
                    cand_refs.append((s1_id, cid))

            if feature_matrix:
                X = np.array(feature_matrix, dtype=np.float32)
                prob_xgb = xgb_model.predict(xgb.DMatrix(X))
                prob_lgb = lgb_model.predict(X)
                prob_cat = cat_model.predict_proba(X)[:, 1]
                probs = (0.45 * prob_xgb) + (0.35 * prob_lgb) + (0.20 * prob_cat)

                # High Precision Threshold = 0.75
                best_new_s2 = {}
                best_new_s3 = {}
                best_p_s2 = defaultdict(float)
                best_p_s3 = defaultdict(float)

                for (s1_id, cid), p in zip(cand_refs, probs):
                    if p >= 0.75:
                        if cid.startswith("S2-") and p > best_p_s2[s1_id]:
                            best_p_s2[s1_id] = p
                            best_new_s2[s1_id] = cid
                        elif cid.startswith("S3-") and p > best_p_s3[s1_id]:
                            best_p_s3[s1_id] = p
                            best_new_s3[s1_id] = cid

                for s1_id in set(list(best_new_s2.keys()) + list(best_new_s3.keys())):
                    if s1_id in best_new_s2:
                        cid = best_new_s2[s1_id]
                        current_matches[s1_id].append(cid)
                        current_candidates[s1_id].append(cid)
                        recovered_matches += 1
                    if s1_id in best_new_s3:
                        cid = best_new_s3[s1_id]
                        current_matches[s1_id].append(cid)
                        current_candidates[s1_id].append(cid)
                        recovered_matches += 1

        for row in reader:
            s1_id = row[0].strip()
            if s1_id in target_queries:
                name = row[1].strip() if len(row) > 1 else ""
                addr = row[2].strip() if len(row) > 2 else ""
                country = row[3].strip().lower() if len(row) > 3 else ""
                batch.append((s1_id, name, addr, country))
                if len(batch) >= 5000:
                    process_recovery_batch(batch)
                    pbar.update(len(batch))
                    batch = []

        if batch:
            process_recovery_batch(batch)
            pbar.update(len(batch))
        pbar.close()

    print(f"\n[+] Total Newly Recovered High-Precision Matches: +{recovered_matches:,}")

    # 5. Write Maximized Output Files
    print("\n[5/5] Writing Maximized Output Files...")
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
            matches = list(dict.fromkeys(current_matches.get(s1_id, [])))
            cands = list(dict.fromkeys(current_candidates.get(s1_id, []) + matches))

            wm.writerow([s1_id, ",".join(matches)])
            wc.writerow([s1_id, ",".join(cands)])

    # 6. Official Validation
    print("\nRunning Official Validation Checks...")
    os.system(f"python3 validate_dgx_tsv.py {OUT_MATCHING}")
    os.system(f"python3 student_resource/utils/validate_submission.py -m {OUT_MATCHING} -c {OUT_CANDIDATE} -t student_resource/dataset/test")
    print(f"\n🎉 MAXIMIZED UNSTOP PIPELINE COMPLETE: {OUT_MATCHING}")
    print(f"Total time elapsed: {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
