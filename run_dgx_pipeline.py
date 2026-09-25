"""
DGX Fast High-Precision Entity Resolution Pipeline
Accelerated via RapidFuzz, Multi-Attribute Blocking, and F0.5 Calibration.
Targeting 0.90+ Macro F0.5 on Unstop Leaderboard.
"""
import os
import re
import csv
import time
import zipfile
from collections import defaultdict
from rapidfuzz import fuzz
from tqdm import tqdm

DATA_DIR = "student_resource/dataset"
OUTPUT_DIR = "output"
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

def main():
    print("="*70)
    print("🚀 STARTING DGX HIGH-PRECISION PIPELINE (TARGET: 0.90+)")
    print("="*70)

    # STEP 1: INDEX S2 AND S3 BY COUNTRY + NAME PREFIX + PINCODE
    print("[*] Indexing Target Records (S2 & S3)...")
    idx_exact = defaultdict(list)
    idx_first3 = defaultdict(list)
    idx_pin = defaultdict(list)
    records = {}

    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        path = os.path.join(DATA_DIR, "test", fname)
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
                
                # Country-partitioned keys
                if cn:
                    idx_exact[(country, cn)].append(eid)
                    first_3 = cn[:3]
                    if len(first_3) >= 3:
                        idx_first3[(country, first_3)].append(eid)
                if pin:
                    idx_pin[(country, pin)].append(eid)

    print(f"[+] Loaded {len(records):,} target records.")

    # STEP 2: PROCESS S1 ENTITIES & FUZZY RE-RANK
    s1_path = os.path.join(DATA_DIR, "test", "test_source1.tsv")
    matching_file = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_file = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    print("[*] Matching S1 Entities with RAPIDFUZZ scoring...")
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
        
        for row in tqdm(r_s1, total=1732544, desc="Processing S1"):
            total_s1 += 1
            s1_id = row[0].strip()
            b_name = row[1].strip() if len(row) > 1 else ""
            b_addr = row[2].strip() if len(row) > 2 else ""
            country = row[3].strip() if len(row) > 3 else ""
            
            cn = clean_name(b_name)
            pin = extract_pin(b_addr)
            
            candidates = set()
            
            # 1. Exact Name Matches (Highest weight)
            if (country, cn) in idx_exact:
                candidates.update(idx_exact[(country, cn)][:15])
            
            # 2. Pincode Matches
            if pin and (country, pin) in idx_pin:
                candidates.update(idx_pin[(country, pin)][:15])
                
            # 3. First 3 chars prefix (if candidates count is small)
            if len(candidates) < 5 and cn:
                first_3 = cn[:3]
                if (country, first_3) in idx_first3:
                    candidates.update(idx_first3[(country, first_3)][:10])
            
            if not candidates:
                w_m.writerow([s1_id, ""])
                w_c.writerow([s1_id, ""])
                continue
                
            w_c.writerow([s1_id, ",".join(candidates)])
            
            # SCORE CANDIDATES FOR F0.5 PRECISION
            best_s2, best_s2_score = None, 0.0
            best_s3, best_s3_score = None, 0.0
            
            for cid in candidates:
                t_cn, t_addr, _ = records[cid]
                # Rapid Token Set Ratio + Ratio
                score = (fuzz.ratio(cn, t_cn) * 0.6) + (fuzz.token_set_ratio(cn, t_cn) * 0.4)
                
                if cid.startswith("S2-"):
                    if score > best_s2_score:
                        best_s2_score = score
                        best_s2 = cid
                elif cid.startswith("S3-"):
                    if score > best_s3_score:
                        best_s3_score = score
                        best_s3 = cid
            
            # HIGH PRECISION THRESHOLD: Only accept if match score >= 85%
            # Low scores are rejected as Singletons (Scores 1.0 in Macro F0.5)
            final_matches = []
            if best_s2 and best_s2_score >= 85.0:
                final_matches.append(best_s2)
            if best_s3 and best_s3_score >= 85.0:
                final_matches.append(best_s3)
                
            if final_matches:
                matched_count += 1
                w_m.writerow([s1_id, ",".join(final_matches)])
            else:
                w_m.writerow([s1_id, ""])

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

    print(f"[+] Created {code_zip} ({os.path.getsize(code_zip)} bytes)")

    # STEP 4: RUN OFFICIAL VALIDATION SCRIPT
    print("\n[*] Running Official Submission Validator...")
    os.system("python3 student_resource/utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir student_resource/dataset/test")

if __name__ == "__main__":
    main()
