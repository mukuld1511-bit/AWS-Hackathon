"""
Normalization Module for Business Entity Resolution.

Provides deterministic string cleaning, legal suffix removal, and address token
extraction functions for candidate generation and blocking.
"""

import re
import unicodedata
from typing import Optional, List, Set, Tuple

# Configuration: List of business/legal suffixes ordered by length (multi-token first)
DEFAULT_LEGAL_SUFFIXES: Tuple[str, ...] = (
    "प्राइवेट लिमिटेड",
    "private limited",
    "pvt ltd",
    "private ltd",
    "pvt limited",
    "incorporated",
    "corporation",
    "enterprises",
    "enterprise",
    "associates",
    "solutions",
    "services",
    "limited",
    "private",
    "center",
    "group",
    "company",
    "co",
    "inc",
    "llc",
    "llp",
    "ltd",
    "pvt",
    "corp",
    "sarl",
)

# Regex matching multiple whitespaces
WHITESPACE_PATTERN: re.Pattern = re.compile(r'\s+', re.UNICODE)

def _build_legal_suffix_regex(suffixes: Tuple[str, ...]) -> re.Pattern:
    """Construct a regex pattern matching legal suffixes at word boundaries."""
    escaped_suffixes = [re.escape(s) for s in suffixes]
    pattern_str = r'(?:\b|\s)(?:' + '|'.join(escaped_suffixes) + r')(?:\b|\s|$)'
    return re.compile(pattern_str, re.IGNORECASE | re.UNICODE)

LEGAL_SUFFIXES_REGEX: re.Pattern = _build_legal_suffix_regex(DEFAULT_LEGAL_SUFFIXES)

def clean_punctuation(text: str) -> str:
    """Replace Unicode Punctuation ('P') and Symbol ('S') characters with spaces.
    
    Preserves Unicode letters, digits, and combining marks (matras/accents).
    """
    return "".join(" " if unicodedata.category(c).startswith(('P', 'S')) else c for c in text)


def normalize_name(name: Optional[str], custom_suffixes_regex: Optional[re.Pattern] = None) -> str:
    """Normalize a business name string deterministically.

    Steps:
    1. Handle None / empty / whitespace-only inputs.
    2. Convert to lower case.
    3. Replace punctuation and symbols with spaces (preserving Unicode matras & accents).
    4. Strip legal suffixes (e.g., Inc, LLC, Pvt Ltd, Sarl) at word boundaries.
    5. Collapse multiple whitespaces.

    Args:
        name: Raw business name string.
        custom_suffixes_regex: Optional custom regex pattern for legal suffixes.

    Returns:
        Cleaned, deterministic normalized string.
    """
    if name is None:
        return ""
    
    text = name.strip().lower()
    if not text:
        return ""
    
    # Replace punctuation and symbols with whitespace
    text = clean_punctuation(text)
    
    # Remove legal suffixes
    regex = custom_suffixes_regex or LEGAL_SUFFIXES_REGEX
    text = regex.sub(' ', text)
    
    # Collapse extra whitespace
    return WHITESPACE_PATTERN.sub(' ', text).strip()


def extract_address_numbers(address: Optional[str]) -> str:
    """Extract numeric address tokens from a physical address string.

    Args:
        address: Raw address string (may be None or empty).

    Returns:
        Underscore-separated string of extracted digit tokens, or empty string.
    """
    if address is None:
        return ""
    
    text = address.strip()
    if not text:
        return ""
    
    # Clean punctuation to separate tokens cleanly
    clean_text = clean_punctuation(text)
    tokens = clean_text.split()
    
    nums = [t for t in tokens if t.isdigit()]
    return "_".join(nums)


def extract_address_anchor(address: Optional[str]) -> str:
    """Extract a compact physical address anchor (first numeric token + first alphabetic word > 2 chars).

    Example:
        "85 Wayne Avenue, Floor 2" -> "85_wayne"

    Args:
        address: Raw physical address string.

    Returns:
        Anchor string in format "{first_num}_{first_word}", or empty string if missing required tokens.
    """
    if address is None:
        return ""
    
    text = address.strip()
    if not text:
        return ""
    
    clean_text = clean_punctuation(text.lower())
    tokens = clean_text.split()
    
    first_num = ""
    first_word = ""
    
    for t in tokens:
        if not first_num and t.isdigit():
            first_num = t
        elif not first_word and t.isalpha() and len(t) > 2:
            first_word = t
            
        if first_num and first_word:
            break
            
    if first_num and first_word:
        return f"{first_num}_{first_word}"
    
    return ""
