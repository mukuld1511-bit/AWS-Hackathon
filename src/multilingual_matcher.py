"""
Multilingual Entity Matcher for Indian Indic-Script Business Names.
Detects cross-lingual matches (English ↔ Tamil/Hindi/Kannada) using
multilingual sentence embeddings + address overlap bridging.
"""
import csv
import re
import sys
import os
import time
from collections import defaultdict

# --- Configuration ---
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COSINE_THRESHOLD = 0.85      # High threshold for F_0.5 precision safety
BATCH_SIZE = 128              # GPU batch size for encoding
MAX_CANDIDATES_PER_ENTITY = 15

HAS_INDIC = re.compile(r'[\u0900-\u0D7F]')

def has_indic_chars(text):
    return bool(HAS_INDIC.search(text)) if text else False

def extract_address_tokens(addr):
    """Extract normalized tokens from address for overlap comparison."""
    if not addr: return set()
    return set(re.findall(r'\w+', addr.lower()))

def main():
    start = time.time()
    
    # Step 1: Load only India records from S1 that might need cross-lingual matching
    print("[*] Loading India S1 entities...")
    s1_india = []
    with open("student_resource/dataset/test/test_source1.tsv", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row[3].strip() == "India":
                s1_india.append(row)
    print(f"[+] Loaded {len(s1_india):,} India S1 entities")
    
    # Step 2: Load India records from S2/S3 that contain Indic characters
    print("[*] Loading India S2/S3 entities with Indic text...")
    indic_candidates = []
    for src in ["test_source2.tsv", "test_source3.tsv"]:
        with open(f"student_resource/dataset/test/{src}", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if row[3].strip() == "India" and has_indic_chars(row[1]):
                    indic_candidates.append(row)
    print(f"[+] Loaded {len(indic_candidates):,} Indic-text candidates from S2/S3")
    
    # Step 3: Encode business names using multilingual model
    print(f"[*] Loading embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("[*] Encoding S1 India names...")
    s1_names = [row[1] for row in s1_india]
    s1_embeddings = model.encode(s1_names, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    
    print("[*] Encoding Indic candidate names...")
    cand_names = [row[1] for row in indic_candidates]
    cand_embeddings = model.encode(cand_names, batch_size=BATCH_SIZE, show_progress_bar=True, normalize_embeddings=True)
    
    # Step 4: Compute cosine similarities and find matches
    import numpy as np
    print("[*] Computing similarities (this may take a while)...")
    
    matches = defaultdict(list)
    
    # Process in chunks to avoid OOM
    CHUNK = 1000
    for i in range(0, len(s1_india), CHUNK):
        chunk_embs = s1_embeddings[i:i+CHUNK]
        sims = np.dot(chunk_embs, cand_embeddings.T)  # (CHUNK, num_candidates)
        
        for local_idx in range(len(chunk_embs)):
            global_idx = i + local_idx
            s1_id = s1_india[global_idx][0]
            s1_addr_tokens = extract_address_tokens(s1_india[global_idx][2])
            
            top_indices = np.where(sims[local_idx] >= COSINE_THRESHOLD)[0]
            
            for cand_idx in top_indices:
                cand_row = indic_candidates[cand_idx]
                cand_addr_tokens = extract_address_tokens(cand_row[2])
                
                # Extra confidence: check address token overlap
                addr_overlap = len(s1_addr_tokens & cand_addr_tokens) if s1_addr_tokens and cand_addr_tokens else 0
                
                if sims[local_idx][cand_idx] >= 0.90 or (sims[local_idx][cand_idx] >= COSINE_THRESHOLD and addr_overlap >= 2):
                    matches[s1_id].append(cand_row[0])
        
        if (i + CHUNK) % 10000 == 0:
            print(f"    Processed {i+CHUNK:,} / {len(s1_india):,} S1 entities...")
    
    # Step 5: Write output
    output_path = "output/multilingual_matches.tsv"
    os.makedirs("output", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, match_ids in matches.items():
            unique_ids = list(dict.fromkeys(match_ids))[:MAX_CANDIDATES_PER_ENTITY]
            f.write(f"{s1_id}\t{','.join(unique_ids)}\n")
    
    elapsed = time.time() - start
    print(f"\n[+] DONE in {elapsed:.1f}s!")
    print(f"[+] Found multilingual matches for {len(matches):,} S1 entities")
    print(f"[+] Output: {output_path}")

if __name__ == "__main__":
    main()
