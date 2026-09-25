"""
Stage 4.7 Failure Analysis & Diagnostic Signal Extractor.

Provides deterministic tools for:
- Extracting the remaining failure cohort (missed true matches under A+B+C+D+F cap 250)
- Computing name-level, address-level, and country-level diagnostic signals
- Classifying failure reasons into measurable categories
- Identifying recurring structural patterns in failure cases
"""

import re
import difflib
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter, defaultdict

from src.normalization import normalize_name, extract_address_numbers, clean_punctuation
from src.experimental_blocking import extract_significant_tokens, extract_street_tokens, COMMON_BUSINESS_STOP_WORDS

ADDRESS_STOP_WORDS: Set[str] = {
    "road", "street", "drive", "avenue", "lane", "court", "suite", "floor",
    "unit", "north", "south", "east", "west", "building", "block", "phase",
    "near", "opp", "opposite", "behind", "beside", "facing", "main", "cross",
    "plot", "no", "number", "nagar", "colony", "area", "dist", "district"
}


def is_latin(text: str) -> bool:
    """Check if all alphabetic characters in text are Latin script."""
    for char in text:
        if char.isalpha():
            name = unicodedata.name(char, "")
            if "LATIN" not in name:
                return False
    return True


def compute_token_jaccard(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute Jaccard similarity between two token lists."""
    s1 = set(tokens1)
    s2 = set(tokens2)
    if not s1 and not s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def compute_pair_diagnostics(
    s1_name: str,
    s1_address: str,
    s1_country: str,
    target_name: str,
    target_address: str,
    target_country: str
) -> Dict:
    """Compute comprehensive diagnostic signals for a pair of entities."""
    norm_s1_n = normalize_name(s1_name)
    norm_tgt_n = normalize_name(target_name)

    norm_s1_a = clean_punctuation(s1_address.lower()) if s1_address else ""
    norm_tgt_a = clean_punctuation(target_address.lower()) if target_address else ""

    # Name tokens & char ratio
    n_tokens1 = norm_s1_n.split() if norm_s1_n else []
    n_tokens2 = norm_tgt_n.split() if norm_tgt_n else []
    name_jaccard = compute_token_jaccard(n_tokens1, n_tokens2)
    shared_name_tokens = list(set(n_tokens1) & set(n_tokens2))
    
    matcher = difflib.SequenceMatcher(None, norm_s1_n, norm_tgt_n)
    name_char_ratio = round(matcher.ratio(), 4)

    is_name_substr = (norm_s1_n in norm_tgt_n or norm_tgt_n in norm_s1_n) if (norm_s1_n and norm_tgt_n) else False
    name_len_diff = abs(len(norm_s1_n) - len(norm_tgt_n))
    name_order_diff = (sorted(n_tokens1) == sorted(n_tokens2)) and (n_tokens1 != n_tokens2) and len(n_tokens1) > 1

    s1_sig_n = extract_significant_tokens(s1_name)
    tgt_sig_n = extract_significant_tokens(target_name)
    shared_sig_n = list(set(s1_sig_n) & set(tgt_sig_n))

    # Address tokens & numbers
    a_tokens1 = [t for t in norm_s1_a.split() if len(t) >= 2]
    a_tokens2 = [t for t in norm_tgt_a.split() if len(t) >= 2]
    addr_jaccard = compute_token_jaccard(a_tokens1, a_tokens2)
    shared_addr_tokens = list(set(a_tokens1) & set(a_tokens2))

    s1_nums = [n for n in extract_address_numbers(s1_address).split("_") if n]
    tgt_nums = [n for n in extract_address_numbers(target_address).split("_") if n]
    shared_nums = list(set(s1_nums) & set(tgt_nums))

    s1_has_digits = len(s1_nums) > 0
    tgt_has_digits = len(tgt_nums) > 0
    both_no_digits = (not s1_has_digits) and (not tgt_has_digits)
    either_no_digits = (not s1_has_digits) or (not tgt_has_digits)

    # Street tokens
    s1_streets = extract_street_tokens(s1_address)
    tgt_streets = extract_street_tokens(target_address)
    
    first_street_match = False
    second_street_match = False
    if s1_streets and tgt_streets:
        first_street_match = (s1_streets[0] == tgt_streets[0])
        second_street_match = (s1_streets[1] == tgt_streets[1])

    is_latin_s1 = is_latin(s1_name)
    is_latin_tgt = is_latin(target_name)
    is_non_latin = (not is_latin_s1) or (not is_latin_tgt)

    return {
        "s1_country": s1_country,
        "target_country": target_country,
        "country_match": (s1_country.strip() == target_country.strip()) if s1_country and target_country else False,
        "exact_name_match": (norm_s1_n == norm_tgt_n),
        "name_jaccard": round(name_jaccard, 4),
        "name_char_ratio": name_char_ratio,
        "shared_name_tokens_count": len(shared_name_tokens),
        "shared_sig_name_tokens_count": len(shared_sig_n),
        "is_name_substr": is_name_substr,
        "name_len_diff": name_len_diff,
        "name_order_diff": name_order_diff,
        "is_non_latin": is_non_latin,
        "exact_addr_match": (norm_s1_a == norm_tgt_a),
        "addr_jaccard": round(addr_jaccard, 4),
        "shared_addr_tokens_count": len(shared_addr_tokens),
        "shared_numeric_tokens_count": len(shared_nums),
        "s1_has_digits": s1_has_digits,
        "tgt_has_digits": tgt_has_digits,
        "both_no_digits": both_no_digits,
        "either_no_digits": either_no_digits,
        "first_street_match": first_street_match,
        "second_street_match": second_street_match,
    }


def classify_failure_categories(diag: Dict) -> List[str]:
    """Classify a pair into non-exclusive failure category flags based on diagnostic signals."""
    categories = []

    # Cat 1: Strong Name Similarity
    if diag["name_jaccard"] >= 0.5 or diag["name_char_ratio"] >= 0.75 or diag["is_name_substr"]:
        categories.append("1. Strong Name Similarity")

    # Cat 2: Strong Address Similarity
    if diag["addr_jaccard"] >= 0.5 or diag["shared_numeric_tokens_count"] >= 1:
        categories.append("2. Strong Address Similarity")

    # Cat 3: Shared Name Tokens (D missed token or capped out)
    if diag["shared_sig_name_tokens_count"] >= 1:
        categories.append("3. Shared Significant Name Tokens")

    # Cat 4: Shared Address Tokens (F missed pair)
    if diag["shared_addr_tokens_count"] >= 2:
        categories.append("4. Shared Address Tokens (>=2)")

    # Cat 5: Name Word-Order Variation
    if diag["name_order_diff"]:
        categories.append("5. Name Word-Order Variation")

    # Cat 6: Missing Address Digits
    if diag["both_no_digits"]:
        categories.append("6. Missing Address Digits (Both)")
    elif diag["either_no_digits"]:
        categories.append("7. Missing Address Digits (Either)")

    # Cat 7: Minor Typo / Character Variation
    if 0.65 <= diag["name_char_ratio"] < 0.85 and diag["name_jaccard"] < 0.5:
        categories.append("8. Name Typo / Minor Char Variation")

    # Cat 8: Non-Latin / Multilingual Script
    if diag["is_non_latin"]:
        categories.append("9. Non-Latin / Multilingual Script")

    # Cat 9: Low Lexical Overlap (Severe Noise)
    if diag["name_jaccard"] < 0.2 and diag["addr_jaccard"] < 0.2 and not diag["shared_numeric_tokens_count"]:
        categories.append("10. Low Lexical Overlap (Severe Noise)")

    if not categories:
        categories.append("11. Other / Unclassified")

    return categories
