"""
src/features_18d.py
===================
18-Dimensional Ultra-Capacity Feature Extractor for Business Entity Resolution
Amazon ML Challenge 2026

Features:
 1. fuzz_ratio                     — Levenshtein character similarity (0-100)
 2. fuzz_token_set_ratio           — Token set overlap ratio
 3. fuzz_token_sort_ratio          — Alphabetically sorted token similarity
 4. fuzz_partial_ratio             — Substring similarity ratio
 5. jaro_winkler                   — Jaro-Winkler prefix-weighted similarity (0-100)
 6. lcs_ratio                      — Longest Common Subsequence length / max length (0-100)
 7. condensed_exact                — 1.0 if spaceless normalized names match (e.g. primemoney)
 8. name_len_diff                  — Absolute difference in name characters
 9. name_word_diff                 — Absolute difference in word count
10. addr_overlap_count             — Count of common non-generic address tokens
11. addr_jaccard_ratio             — Jaccard token overlap ratio (0-100)
12. us_state_match                 — 1.0 if US states match
13. us_state_conflict              — 1.0 if US states conflict (strong FP shield)
14. city_match                     — 1.0 if detected cities/regions match
15. city_conflict                  — 1.0 if detected cities differ
16. number_match                   — 1.0 if first address numbers match
17. number_conflict                — 1.0 if both have numbers and they differ
18. first_word_match               — 1.0 if first words match exactly
"""
import re
import unicodedata
from rapidfuzz import fuzz, distance

PREFIXES = re.compile(r'^(mr|mrs|ms|shri|smt|m/s|messrs|dr|prof|the)\b\s*', re.IGNORECASE)
DBA = re.compile(r'\b(dba|d\.b\.a\.|trading as|t/a|c/o)\b\s*', re.IGNORECASE)
METADATA = re.compile(r'[\(\[\{](?:id|branch|division|code|store|ref)[\s:]*[\w\d\s]+[\)\]\}]', re.IGNORECASE)
DOMAIN_EXT = re.compile(r'\.(com|org|net|in|us|fr|co|biz|info)\b', re.IGNORECASE)

LEGAL_SUFFIXES = re.compile(
    r'\b(inc|incorporated|llc|llp|ltd|limited|pvt|private|corp|corporation|'
    r'co|company|enterprises|enterprise|group|services|service|center|'
    r'solutions|associates|consulting|consultants|technologies|tech|'
    r'sa|sarl|sas|sasu|eurl|ei|snc|sci|gie|gmbh|international|india|trading|agency|'
    r'works|auto|general|medical|industries|industry|brothers|sons|fils|'
    r'traders|trader|dealer|dealers|store|stores|association|societe|société|'
    r'comite|comité|ecole|école|clinique|maison|barbershop|salon|boutique|shop)\b',
    re.IGNORECASE
)
PUNCT = re.compile(r'[^\w\s]', re.UNICODE)
RE_NUM = re.compile(r'\b\d+\b')
RE_US_STATE = re.compile(r'\b([A-Z]{2})\b')

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID',
    'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO',
    'MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA',
    'RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
}

ADDR_STOPWORDS = {
    'flat','no','plot','shop','gala','floor','bldg','building','wing','room',
    'near','opp','opposite','behind','beside','road','rd','street','st','lane',
    'ln','avenue','ave','nagar','colony','complex','house','h','sector','phase',
    'city','west','east','north','south','rue','av','bd','boulevard','allee','allée',
    'chemin','impasse','place','cours','quai','route','rte','residence','batiment',
    'null','undefined','suite','ste','dr','drive','blvd','ct','court'
}


def super_clean_name(s: str) -> str:
    """Normalize business name by removing diacritics, prefixes, dba trade names, and legal suffixes."""
    if not s:
        return ""
    # NFKD diacritics stripping
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    if DBA.search(s):
        s = DBA.split(s)[-1]
    s = METADATA.sub(' ', s)
    s = DOMAIN_EXT.sub(' ', s)
    s = PREFIXES.sub('', s.strip())
    # Acronym dots (e.g. L.L.C. -> LLC)
    s = re.sub(r'(?<=\w)\.(?=\w)', '', s)
    s = PUNCT.sub(' ', s.lower())
    s = LEGAL_SUFFIXES.sub(' ', s)
    return " ".join(s.split())


def extract_numbers_list(addr: str):
    if not addr:
        return []
    return [m for m in RE_NUM.findall(addr) if len(m) <= 5]


def lcs_length(s1: str, s2: str) -> int:
    """Longest common substring length."""
    if not s1 or not s2:
        return 0
    m, n = len(s1), len(s2)
    dp = [0] * (n + 1)
    max_len = 0
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev + 1
                if dp[j] > max_len:
                    max_len = dp[j]
            else:
                dp[j] = 0
            prev = temp
    return max_len


def extract_features_18d(
    s1_name_clean: str,
    s1_addr: str,
    s2_name_clean: str,
    s2_addr: str,
    s1_geo: str = "",
    s2_geo: str = "",
    s1_state: str = "",
    s2_state: str = ""
):
    """
    Extract full 18-dimensional ultra-capacity feature vector.
    Assumes s1_name_clean and s2_name_clean are pre-cleaned via super_clean_name.
    """
    feats = []

    # 1. Levenshtein Full Ratio
    feats.append(fuzz.ratio(s1_name_clean, s2_name_clean))

    # 2. Token Set Ratio
    feats.append(fuzz.token_set_ratio(s1_name_clean, s2_name_clean))

    # 3. Token Sort Ratio
    feats.append(fuzz.token_sort_ratio(s1_name_clean, s2_name_clean))

    # 4. Partial Ratio
    feats.append(fuzz.partial_ratio(s1_name_clean, s2_name_clean))

    # 5. Jaro-Winkler Similarity
    feats.append(distance.JaroWinkler.normalized_similarity(s1_name_clean, s2_name_clean) * 100)

    # 6. LCS Ratio
    max_l = max(len(s1_name_clean), len(s2_name_clean), 1)
    feats.append((lcs_length(s1_name_clean, s2_name_clean) / max_l) * 100)

    # 7. Condensed Exact Match (spaceless e.g. primemoney == primemoney)
    cond1 = s1_name_clean.replace(" ", "")
    cond2 = s2_name_clean.replace(" ", "")
    feats.append(1.0 if (cond1 and cond1 == cond2) else 0.0)

    # 8. Name Length Difference
    feats.append(abs(len(s1_name_clean) - len(s2_name_clean)))

    # 9. Name Word Count Difference
    w1 = s1_name_clean.split()
    w2 = s2_name_clean.split()
    feats.append(abs(len(w1) - len(w2)))

    # Address Tokens (filtered of generic words)
    s1_toks = set(re.findall(r'[a-z0-9]+', s1_addr.lower())) - ADDR_STOPWORDS if s1_addr else set()
    s2_toks = set(re.findall(r'[a-z0-9]+', s2_addr.lower())) - ADDR_STOPWORDS if s2_addr else set()
    overlap = len(s1_toks & s2_toks)
    union_l = len(s1_toks | s2_toks)

    # 10. Address Non-Generic Overlap Count
    feats.append(overlap)

    # 11. Address Jaccard Ratio
    feats.append((overlap / union_l * 100) if union_l > 0 else 0.0)

    # 12. US State Match
    state_match = 1.0 if (s1_state and s2_state and s1_state == s2_state) else 0.0
    feats.append(state_match)

    # 13. US State Conflict
    state_conflict = 1.0 if (s1_state and s2_state and s1_state != s2_state) else 0.0
    feats.append(state_conflict)

    # 14. City Match
    city_match = 1.0 if (s1_geo and s2_geo and s1_geo == s2_geo) else 0.0
    feats.append(city_match)

    # 15. City Conflict
    city_conflict = 1.0 if (s1_geo and s2_geo and s1_geo != s2_geo) else 0.0
    feats.append(city_conflict)

    # Address Numbers
    nums1 = extract_numbers_list(s1_addr)
    nums2 = extract_numbers_list(s2_addr)
    num_match = 0.0
    num_conflict = 0.0
    if nums1 and nums2:
        if nums1[0] == nums2[0]:
            num_match = 1.0
        else:
            num_conflict = 1.0

    # 16. Number Match
    feats.append(num_match)

    # 17. Number Conflict
    feats.append(num_conflict)

    # 18. First Word Match
    first1 = w1[0] if w1 else ""
    first2 = w2[0] if w2 else ""
    feats.append(1.0 if (first1 and first1 == first2) else 0.0)

    return feats
