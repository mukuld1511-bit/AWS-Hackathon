"""
src/graph_transitivity.py
=========================
Phase 4: Graph Transitivity & Mutual Consistency
Amazon ML Challenge 2026

Concept:
In 3-source entity resolution, when S1 matches S2, that real-world business
frequently also exists in S3 (and vice-versa).
In Sub 8, 284,761 entities matched ONLY S2, and 97,538 matched ONLY S3.
Graph Transitivity bridges this gap:
  If S1 -> S2 (high confidence), and S2 <-> S3 (exact name + same geo + addr overlap),
  then infer S1 -> S3!
"""
import os
import csv
import re
from collections import defaultdict
from rapidfuzz import fuzz

LEGAL_SUFFIXES = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|'
    r'co|company|enterprises|enterprise|group|services|service|center|'
    r'solutions|associates|consulting|consultants|technologies|tech|'
    r'sa|sarl|sas|sasu|eurl|ei|snc|sci|gie|gmbh|international|india|trading|agency|'
    r'works|auto|general|medical|industries|industry|brothers|sons|fils|'
    r'traders|trader|dealer|dealers|store|stores|association|societe|société|'
    r'comite|comité|ecole|école|clinique|maison)\b',
    re.IGNORECASE
)
PUNCT = re.compile(r'[^\w\s]', re.UNICODE)


def clean_name(s: str) -> str:
    if not s: return ""
    s = PUNCT.sub(' ', s.lower())
    s = LEGAL_SUFFIXES.sub(' ', s)
    return " ".join(s.split())


def apply_graph_transitivity(
    matching_in: str,
    candidate_in: str,
    test_dir: str,
    matching_out: str,
    candidate_out: str
):
    print("=" * 70)
    print("🕸️  PHASE 4: GRAPH TRANSITIVITY & MUTUAL CONSISTENCY")
    print("=" * 70)

    # Step 1: Load S2 records that are currently matched
    print("\n[1/4] Loading existing matches...")
    s1_matches = {}
    matched_s2_needed = set()
    matched_s3_needed = set()

    with open(matching_in, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            s1_id = row[0].strip()
            ids = [x.strip() for x in row[1].split(',') if x.strip()] if len(row) > 1 and row[1].strip() else []
            s1_matches[s1_id] = ids
            for cid in ids:
                if cid.startswith("S2-"): matched_s2_needed.add(cid)
                elif cid.startswith("S3-"): matched_s3_needed.add(cid)

    print(f"   Total S1: {len(s1_matches):,}")
    print(f"   Matched S2 count: {len(matched_s2_needed):,}")
    print(f"   Matched S3 count: {len(matched_s3_needed):,}")

    # Step 2: Load S2 and S3 target records into memory
    print("\n[2/4] Indexing S2 and S3 for cross-source bridging...")
    # Index S3 by (country, clean_name) for fast S2->S3 lookup
    s3_by_name = defaultdict(list)
    s3_records = {}
    with open(os.path.join(test_dir, "test_source3.tsv"), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 4: continue
            eid = row[0].strip()
            cn = clean_name(row[1].strip())
            addr = row[2].strip()
            country = row[3].strip().lower()
            if cn:
                s3_by_name[(country, cn)].append(eid)
                s3_records[eid] = (cn, addr, country)

    # Index S2 by (country, clean_name) for fast S3->S2 lookup
    s2_by_name = defaultdict(list)
    s2_records = {}
    with open(os.path.join(test_dir, "test_source2.tsv"), 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader)
        for row in reader:
            if len(row) < 4: continue
            eid = row[0].strip()
            cn = clean_name(row[1].strip())
            addr = row[2].strip()
            country = row[3].strip().lower()
            if cn:
                s2_by_name[(country, cn)].append(eid)
                s2_records[eid] = (cn, addr, country)

    print(f"   Indexed {len(s3_records):,} S3 and {len(s2_records):,} S2 records.")

    # Step 3: Graph Transitivity inference
    print("\n[3/4] Performing transitive graph inference...")
    s3_recovered = 0
    s2_recovered = 0

    for s1_id, ids in s1_matches.items():
        has_s2 = [x for x in ids if x.startswith("S2-")]
        has_s3 = [x for x in ids if x.startswith("S3-")]

        # Case A: Has S2, but missing S3
        if has_s2 and not has_s3:
            s2_id = has_s2[0]
            if s2_id in s2_records:
                s2_cn, s2_addr, s2_country = s2_records[s2_id]
                s2_toks = set(s2_addr.lower().split()) if s2_addr else set()
                candidates = s3_by_name.get((s2_country, s2_cn), [])
                # Filter candidates by address token overlap (>= 2 common words)
                for cand_s3 in candidates:
                    _, s3_addr, _ = s3_records[cand_s3]
                    s3_toks = set(s3_addr.lower().split()) if s3_addr else set()
                    overlap = len(s2_toks & s3_toks)
                    if overlap >= 2 or (len(s2_toks) <= 2 and len(s3_toks) <= 2):
                        ids.append(cand_s3)
                        s3_recovered += 1
                        break

        # Case B: Has S3, but missing S2
        elif has_s3 and not has_s2:
            s3_id = has_s3[0]
            if s3_id in s3_records:
                s3_cn, s3_addr, s3_country = s3_records[s3_id]
                s3_toks = set(s3_addr.lower().split()) if s3_addr else set()
                candidates = s2_by_name.get((s3_country, s3_cn), [])
                for cand_s2 in candidates:
                    _, s2_addr, _ = s2_records[cand_s2]
                    s2_toks = set(s2_addr.lower().split()) if s2_addr else set()
                    overlap = len(s3_toks & s2_toks)
                    if overlap >= 2 or (len(s3_toks) <= 2 and len(s2_toks) <= 2):
                        ids.append(cand_s2)
                        s2_recovered += 1
                        break

    print(f"   Transitively recovered S3 matches: +{s3_recovered:,}")
    print(f"   Transitively recovered S2 matches: +{s2_recovered:,}")
    print(f"   Total new matches: +{s3_recovered + s2_recovered:,}")

    # Step 4: Write matching results and ensure candidate alignment
    print("\n[4/4] Writing output files with 100% candidate synchronization...")
    with open(matching_out, 'w', encoding='utf-8') as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, ids in s1_matches.items():
            fm.write(f"{s1_id}\t{','.join(ids)}\n")

    # Sync candidates
    with open(candidate_in, 'r', encoding='utf-8') as fcin, open(candidate_out, 'w', encoding='utf-8') as fcout:
        reader = csv.reader(fcin, delimiter='\t')
        header = next(reader)
        fcout.write('\t'.join(header) + '\n')
        for row in reader:
            s1_id = row[0].strip()
            cands = [x.strip() for x in row[1].split(',') if x.strip()] if len(row) > 1 and row[1].strip() else []
            cand_set = set(cands)
            if s1_id in s1_matches:
                for m in s1_matches[s1_id]:
                    if m not in cand_set:
                        cands.append(m)
                        cand_set.add(m)
            fcout.write(f"{s1_id}\t{','.join(cands)}\n")

    print(f"✅ DONE! Matching output: {matching_out}")
    print(f"✅ DONE! Candidate output: {candidate_out}")


if __name__ == "__main__":
    import sys
    apply_graph_transitivity(
        matching_in="output_final/matching_results.tsv",
        candidate_in="output_final/candidate_pairs.tsv",
        test_dir="student_resource/dataset/test",
        matching_out="output_final/matching_results_transitive.tsv",
        candidate_out="output_final/candidate_pairs_transitive.tsv"
    )
