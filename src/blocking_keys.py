"""
Blocking Keys Module for Entity Resolution.

Provides functions to generate deterministic blocking keys (Key A, Key B, Key C)
using string normalization utilities from src/normalization.py.
"""

from typing import Optional, Tuple
from src.normalization import (
    normalize_name,
    extract_address_numbers,
    extract_address_anchor,
)

# Evidence Type Identifiers
KEY_NAME: str = "NAME"
KEY_PREFIX: str = "PREFIX"
KEY_ADDRESS: str = "ADDRESS"


def generate_key_a(country: str, business_name: str) -> Optional[Tuple[str, str]]:
    """Generate Key A: (country, normalized_name).

    Args:
        country: Country string label.
        business_name: Raw business name.

    Returns:
        Tuple of (country, normalized_name) or None if name is empty.
    """
    if not country:
        return None
    
    country_norm = country.strip()
    norm_n = normalize_name(business_name)
    if not norm_n:
        return None
    
    return (country_norm, norm_n)


def generate_key_b(
    country: str,
    business_name: str,
    business_address: str,
    prefix_len: int = 4
) -> Optional[Tuple[str, str, str]]:
    """Generate Key B: (country, name_prefix, address_numbers).

    Args:
        country: Country string label.
        business_name: Raw business name.
        business_address: Raw physical address.
        prefix_len: Length of normalized name prefix (default: 4).

    Returns:
        Tuple of (country, name_prefix, address_numbers) or None if name/addr numbers missing.
    """
    if not country:
        return None
    
    country_norm = country.strip()
    norm_n = normalize_name(business_name)
    if not norm_n:
        return None
    
    name_prefix = norm_n[:prefix_len]
    addr_nums = extract_address_numbers(business_address)
    
    if not name_prefix or not addr_nums:
        return None
    
    return (country_norm, name_prefix, addr_nums)


def generate_key_c(country: str, business_address: str) -> Optional[Tuple[str, str]]:
    """Generate Key C: (country, address_anchor).

    Args:
        country: Country string label.
        business_address: Raw physical address.

    Returns:
        Tuple of (country, address_anchor) or None if address anchor missing.
    """
    if not country:
        return None
    
    country_norm = country.strip()
    anchor = extract_address_anchor(business_address)
    if not anchor:
        return None
    
    return (country_norm, anchor)
