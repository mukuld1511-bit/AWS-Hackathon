"""
Fast, bulletproof submission validator for DGX.
Checks TSV against Unstop's exact evaluation rules.
"""
import sys
import os
import csv

def validate(tsv_path, test_source1_path="student_resource/dataset/test/test_source1.tsv"):
    print("="*70)
    print(f"🔍 CHECKING FILE: {tsv_path}")
    print("="*70)
    
    if not os.path.exists(tsv_path):
        print(f"❌ ERROR: File does not exist at {tsv_path}")
        return False

    size_mb = os.path.getsize(tsv_path) / (1024 * 1024)
    print(f"[*] File Size: {size_mb:.2f} MB")
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        try:
            header = next(reader)
        except StopIteration:
            print("❌ ERROR: File is empty!")
            return False
            
        print(f"[*] Header: {header}")
        if header != ["source1_entity_id", "matched_entity_ids"]:
            print(f"❌ ERROR: Invalid header! Expected ['source1_entity_id', 'matched_entity_ids'], got {header}")
            return False
            
        row_count = 0
        invalid_ids = 0
        self_matches = 0
        bad_format = 0
        
        for row in reader:
            row_count += 1
            if len(row) < 1 or len(row) > 2:
                bad_format += 1
                if bad_format <= 3:
                    print(f"❌ Line {row_count+1}: Bad column count ({len(row)}) -> {row}")
                continue
                
            s1_id = row[0].strip()
            if not s1_id.startswith("S1-"):
                invalid_ids += 1
                if invalid_ids <= 3:
                    print(f"❌ Line {row_count+1}: First column is not S1- ID -> {s1_id}")
                    
            if len(row) > 1 and row[1].strip():
                matches = [m.strip() for m in row[1].split(',') if m.strip()]
                for m in matches:
                    if m.startswith("S1-"):
                        self_matches += 1
                        if self_matches <= 3:
                            print(f"❌ Line {row_count+1}: Contains illegal S1 self-match -> {m}")
                    elif not (m.startswith("S2-") or m.startswith("S3-")):
                        invalid_ids += 1
                        if invalid_ids <= 3:
                            print(f"❌ Line {row_count+1}: Invalid target ID prefix -> {m}")

    print(f"[*] Total Data Rows: {row_count:,} (Expected: 1,732,544)")
    if row_count != 1732544:
        print(f"❌ ERROR: Row count mismatch! Found {row_count:,}, expected exactly 1,732,544.")
        return False
        
    if bad_format > 0 or invalid_ids > 0 or self_matches > 0:
        print(f"❌ ERROR: Found {bad_format} bad format lines, {invalid_ids} invalid IDs, {self_matches} self-matches.")
        return False

    print("="*70)
    print("🎉 100% PERFECT PASS! TSV file is 100% valid and safe for Unstop.")
    print("="*70)
    return True

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "sub3_output/matching_results.tsv"
    validate(path)
