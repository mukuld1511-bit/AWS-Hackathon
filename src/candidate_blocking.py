import os
import csv
import re
from collections import defaultdict

LEGAL_SUFFIXES = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|'
    r'co|company|enterprises|enterprise|group|services|service|center|'
    r'solutions|associates|consulting|consultants|technologies|tech)\b',
    re.IGNORECASE
)
PUNCT = re.compile(r'[^\w\s]', re.UNICODE)

def normalize_name(name):
    if not name: return ""
    text = PUNCT.sub(' ', name.lower())
    text = LEGAL_SUFFIXES.sub(' ', text)
    return " ".join(text.split())

def extract_numbers(text):
    """Extract all digit sequences from text."""
    if not text: return ""
    nums = re.findall(r'\d+', text)
    return "_".join(sorted(set(nums)))  # deterministic order

def address_anchor(address):
    """Extract first number + first alphabetic word from address."""
    if not address: return ""
    tokens = address.lower().split()
    nums = [t for t in tokens if t.isdigit()]
    words = [t for t in tokens if t.isalpha() and len(t) > 2]
    first_num = nums[0] if nums else ""
    first_word = words[0] if words else ""
    return f"{first_num}_{first_word}"

def main():
    print("Building Inverted Indices...")
    # Phase 1: Build inverted indices over S2 and S3
    index_name = defaultdict(list)    # Key A
    index_prefix = defaultdict(list)  # Key B
    index_addr = defaultdict(list)    # Key C

    # Stream S2 and S3 records
    for source_file in ["test_source2.tsv", "test_source3.tsv"]:
        filepath = f"student_resource/dataset/test/{source_file}"
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue
            
        with open(filepath, encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)  # skip header
            for row in reader:
                if len(row) < 4: continue
                eid, name, addr, country = row[0], row[1], row[2], row[3].strip()
                norm_n = normalize_name(name)
                
                if norm_n and country:
                    index_name[(country, norm_n)].append(eid)
                
                prefix = norm_n[:4] if norm_n else ""
                addr_nums = extract_numbers(addr)
                if prefix and addr_nums:
                    index_prefix[(country, prefix, addr_nums)].append(eid)
                
                addr_anch = address_anchor(addr)
                if addr_anch and "_" in addr_anch:
                    index_addr[(country, addr_anch)].append(eid)

    print("Generating candidates for S1...")
    # Phase 2: For each S1 entity, collect candidates from all 3 indices
    MAX_CANDIDATES = 25
    
    os.makedirs("output", exist_ok=True)
    s1_path = "student_resource/dataset/test/test_source1.tsv"
    
    if not os.path.exists(s1_path):
        print(f"File not found: {s1_path}")
        return

    with open(s1_path, encoding="utf-8") as f_in, \
         open("output/candidate_pairs.tsv", "w", encoding="utf-8", newline="") as f_out:
        
        reader = csv.reader(f_in, delimiter="\t")
        next(reader)  # skip header
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        
        count = 0
        for row in reader:
            if len(row) < 4: continue
            s1_id, name, addr, country = row[0], row[1], row[2], row[3].strip()
            norm_n = normalize_name(name)
            candidates = set()
            
            # Key A lookup
            candidates.update(index_name.get((country, norm_n), []))
            
            # Key B lookup
            prefix = norm_n[:4] if norm_n else ""
            addr_nums = extract_numbers(addr)
            if prefix and addr_nums:
                candidates.update(index_prefix.get((country, prefix, addr_nums), []))
            
            # Key C lookup
            addr_anch = address_anchor(addr)
            if addr_anch and "_" in addr_anch:
                candidates.update(index_addr.get((country, addr_anch), []))
            
            # Cap candidates
            cand_list = list(candidates)[:MAX_CANDIDATES]
            cand_str = ",".join(cand_list)
            f_out.write(f"{s1_id}\t{cand_str}\n")
            
            count += 1
            if count % 100000 == 0:
                print(f"Processed {count} S1 records...")

    print("Candidate generation complete.")

if __name__ == "__main__":
    main()
