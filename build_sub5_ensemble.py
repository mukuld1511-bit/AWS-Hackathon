"""
Submission 5 Ensemble: Combines XGBoost GBDT (Sub 4) + Prateek's Multilingual Indic Matcher.
Targeting 0.65+ / 0.80+ on Unstop Leaderboard.
"""
import os
import csv
import sys
import shutil

OUTPUT_DIR = "sub5_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SUB4_TSV = "sub3_output/matching_results.tsv"
MULTILINGUAL_TSV = "output/multilingual_matches.tsv"
SUB5_TSV = os.path.join(OUTPUT_DIR, "matching_results.tsv")
TEST_S1 = "student_resource/dataset/test/test_source1.tsv"

def build_ensemble():
    print("=" * 70)
    print("🚀 BUILDING SUBMISSION 5 ENSEMBLE (XGBOOST + INDIC MULTILINGUAL)")
    print("=" * 70)
    
    # 1. Load Multilingual Matches (if available)
    multi_matches = {}
    if os.path.exists(MULTILINGUAL_TSV):
        print(f"[*] Loading Multilingual Indic matches from {MULTILINGUAL_TSV}...")
        with open(MULTILINGUAL_TSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[1].strip():
                    s1 = row[0].strip()
                    m_list = [x.strip() for x in row[1].split(',') if x.strip()]
                    multi_matches[s1] = m_list
        print(f"[+] Loaded {len(multi_matches):,} multilingual matches.")
    else:
        print(f"⚠️ {MULTILINGUAL_TSV} not found. Running pure XGBoost base.")

    # 2. Load Sub 4 Matches
    print(f"[*] Loading Sub 4 (XGBoost 0.85 Threshold) from {SUB4_TSV}...")
    sub4_matches = {}
    if os.path.exists(SUB4_TSV):
        with open(SUB4_TSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            for row in reader:
                if row:
                    s1 = row[0].strip()
                    m_list = [x.strip() for x in row[1].split(',') if x.strip()] if len(row) > 1 and row[1].strip() else []
                    sub4_matches[s1] = m_list
        print(f"[+] Loaded {len(sub4_matches):,} Sub 4 records.")
    else:
        print(f"❌ Error: {SUB4_TSV} not found!")
        return

    # 3. Ensemble Fusion adhering strictly to F0.5 Precision rules:
    # Rule: For each S1 entity:
    # - If Sub 4 has high-confidence GBDT match (>= 0.85 prob), KEEP IT (Proven 0.415 score).
    # - If Sub 4 had NO match (empty), but Multilingual Matcher found high-confidence Indic transliteration match, ADD IT!
    # - Constraint: Max 1 match from S2, Max 1 match from S3.
    
    print("[*] Fusing models...")
    total_rows = 0
    sub4_used = 0
    multi_added = 0
    singleton_count = 0
    
    with open(TEST_S1, 'r', encoding='utf-8') as fin, \
         open(SUB5_TSV, 'w', encoding='utf-8', newline='') as fout:
        
        reader = csv.reader(fin, delimiter='\t')
        writer = csv.writer(fout, delimiter='\t', lineterminator='\n')
        
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        next(reader, None)
        
        for row in reader:
            total_rows += 1
            s1_id = row[0].strip()
            
            final_matches = []
            
            # Primary: Sub 4 GBDT matches
            if s1_id in sub4_matches and sub4_matches[s1_id]:
                final_matches = list(sub4_matches[s1_id])
                sub4_used += 1
            
            # Secondary: Fill gaps with high-confidence Multilingual Indic matches
            if not final_matches and s1_id in multi_matches:
                m_cands = multi_matches[s1_id]
                s2_cand = [x for x in m_cands if x.startswith("S2-")]
                s3_cand = [x for x in m_cands if x.startswith("S3-")]
                if s2_cand: final_matches.append(s2_cand[0])
                if s3_cand: final_matches.append(s3_cand[0])
                if final_matches:
                    multi_added += 1
            
            if final_matches:
                writer.writerow([s1_id, ",".join(final_matches)])
            else:
                writer.writerow([s1_id, ""])
                singleton_count += 1
                
    print(f"\n[+] Fusion Complete!")
    print(f"    Total S1 rows:      {total_rows:,} (Expected: 1,732,544)")
    print(f"    Sub 4 Base Matches: {sub4_used:,}")
    print(f"    Multilingual Boost: +{multi_added:,} new high-confidence matches")
    print(f"    Singletons Retained:{singleton_count:,}")
    print(f"    Final Output File:  {SUB5_TSV}")
    
    # 4. Strict Validation
    os.system(f"python3 validate_dgx_tsv.py {SUB5_TSV}")
    
    # 5. Build Final Submission Zip
    staging_dir = "sub5_staging"
    if os.path.exists(staging_dir): shutil.rmtree(staging_dir)
    out_dir = os.path.join(staging_dir, "output")
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(SUB5_TSV, out_dir)
    
    code_dir = os.path.join(staging_dir, "code", "business_entity_resolution")
    src_dir = os.path.join(code_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    for f in os.listdir("src"):
        if f.endswith(".py"):
            shutil.copy2(os.path.join("src", f), src_dir)
    if os.path.exists("requirements.txt"): shutil.copy2("requirements.txt", code_dir)
    if os.path.exists("xgb_reranker.json"): shutil.copy2("xgb_reranker.json", code_dir)
    
    zip_path = "sub5_team_submission"
    shutil.make_archive(zip_path, 'zip', staging_dir)
    shutil.rmtree(staging_dir)
    print(f"\n🎉 SUBMISSION 5 ZIP READY: {zip_path}.zip")
    print(f"🎉 TSV FILE FOR DIRECT UPLOAD: {SUB5_TSV}")

if __name__ == "__main__":
    build_ensemble()
