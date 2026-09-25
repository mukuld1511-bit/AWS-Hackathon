"""
src/features_16d.py
===================
16-Dimensional High-Capacity Feature Extractor for Business Entity Resolution
Amazon ML Challenge 2026

Features:
 1. fuzz_ratio                     — Levenshtein character similarity (0-100)
 2. fuzz_token_set_ratio           — Token set overlap ratio (handles reordering & noise)
 3. fuzz_token_sort_ratio          — Alphabetically sorted token similarity
 4. fuzz_partial_ratio             — Substring similarity ratio
 5. jaro_winkler                   — Jaro-Winkler prefix-weighted similarity (0-100)
 6. lcs_ratio                      — Longest Common Subsequence length / max length (0-100)
 7. name_len_diff                  — Absolute difference in name characters
 8. name_word_diff                 — Absolute difference in word count
 9. addr_overlap_count             — Count of common address tokens
10. addr_jaccard_ratio             — Jaccard token overlap ratio (0-100)
11. us_state_match                 — 1 if US states match, 0 otherwise
12. us_state_conflict              — 1 if both have US states and they differ
13. city_match                     — 1 if detected cities/regions match
14. city_conflict                  — 1 if detected cities differ (strong false-positive shield)
15. number_match                   — 1 if first numbers in addresses match
16. number_conflict                — 1 if both have numbers and they differ
"""
import re
import unicodedata
from rapidfuzz import fuzz, distance

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
RE_NUM = re.compile(r'\b\d+\b')
RE_US_STATE = re.compile(r'\b([A-Z]{2})\b')

US_STATES = {
    'AL','AK','AZ','AR','CA','CO','CT','DE','DC','FL','GA','HI','ID',
    'IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MO',
    'MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA',
    'RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY'
}

COMMON_ADDR = {
    'flat','no','plot','shop','gala','floor','bldg','building','wing','room',
    'near','opp','opposite','behind','beside','road','rd','street','st','lane',
    'ln','avenue','ave','nagar','colony','complex','house','h','sector','phase',
    'city','west','east','north','south','rue','av','bd','boulevard','allee','allée',
    'chemin','impasse','place','cours','quai','route','rte','residence','batiment'
}


def clean_name_enhanced(s: str) -> str:
    if not s: return ""
    # Unicode NFKD unaccenting
    s = unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore').decode('utf-8')
    # Strip dots in acronyms (e.g. L.L.C. -> LLC, P. Ltd -> Pvt Ltd)
    s = re.sub(r'(?<=\w)\.(?=\w)', '', s)
    s = PUNCT.sub(' ', s.lower())
    s = LEGAL_SUFFIXES.sub(' ', s)
    return " ".join(s.split())


def extract_numbers_list(addr: str):
    if not addr: return []
    return [m for m in RE_NUM.findall(addr) if len(m) <= 5]


def lcs_length(s1: str, s2: str) -> int:
    """Longest common substring length."""
    if not s1 or not s2: return 0
    m, n = len(s1), len(s2)
    dp = [0] * (n + 1)
    max_len = 0
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if s1[i - 1] == s2[j - 1]:
                dp[j] = prev + 1
                if dp[j] > max_len: max_len = dp[j]
            else:
                dp[j] = 0
            prev = temp
    return max_len


def extract_features_16d(
    s1_name: str,
    s1_addr: str,
    s2_name: str,
    s2_addr: str,
    s1_geo: str = "",
    s2_geo: str = "",
    s1_state: str = "",
    s2_state: str = ""
):
    """Extract full 16-dimensional high-capacity feature vector."""
    features = []

    # 1. Full Levenshtein Ratio
    features.append(fuzz.ratio(s1_name, s2_name))

    # 2. Token Set Ratio
    features.append(fuzz.token_set_ratio(s1_name, s2_name))

    # 3. Token Sort Ratio
    features.append(fuzz.token_sort_ratio(s1_name, s2_name))

    # 4. Partial Ratio
    features.append(fuzz.partial_ratio(s1_name, s2_name))

    # 5. Jaro-Winkler Similarity
    features.append(distance.JaroWinkler.normalized_similarity(s1_name, s2_name) * 100)

    # 6. LCS Ratio
    max_l = max(len(s1_name), len(s2_name), 1)
    features.append((lcs_length(s1_name, s2_name) / max_l) * 100)

    # 7. Name length difference
    features.append(abs(len(s1_name) - len(s2_name)))

    # 8. Name word count difference
    w1 = len(s1_name.split())
    w2 = len(s2_name.split())
    features.append(abs(w1 - w2))

    # Address Tokens
    s1_toks = set(s1_addr.lower().split()) if s1_addr else set()
    s2_toks = set(s2_addr.lower().split()) if s2_addr else set()
    overlap = len(s1_toks & s2_toks)
    union_l = len(s1_toks | s2_toks)

    # 9. Address Token Overlap Count
    features.append(overlap)

    # 10. Address Jaccard Ratio
    features.append((overlap / union_l * 100) if union_l > 0 else 0.0)

    # 11. US State Match
    state_match = 1.0 if (s1_state and s2_state and s1_state == s2_state) else 0.0
    features.append(state_match)

    # 12. US State Conflict
    state_conflict = 1.0 if (s1_state and s2_state and s1_state != s2_state) else 0.0
    features.append(state_conflict)

    # 13. City Match
    city_match = 1.0 if (s1_geo and s2_geo and s1_geo == s2_geo) else 0.0
    features.append(city_match)

    # 14. City Conflict
    city_conflict = 1.0 if (s1_geo and s2_geo and s1_geo != s2_geo) else 0.0
    features.append(city_conflict)

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

    # 15. Number Match
    features.append(num_match)

    # 16. Number Conflict
    features.append(num_conflict)

    return features
