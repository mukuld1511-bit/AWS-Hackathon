"""
src/deterministic_anchors.py
============================
Phase B: Ultra-High Precision Deterministic Identity Anchoring
Extracts:
  - Phone (E.164 formatted digits)
  - Pincode / Postal Code (US 5-digit, FR 5-digit, IN 6-digit)
  - Tax & Corporate IDs (GSTIN, PAN, CIN, SIREN, EIN)
  - Domain names from raw text
"""

import re
from typing import Dict, List, Set, Tuple, Optional

# Compiled Regular Expressions
RE_GSTIN = re.compile(r'\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z]{1}[1-9A-Z]{1}Z[0-9A-Z]{1}\b', re.IGNORECASE)
RE_PAN = re.compile(r'\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b', re.IGNORECASE)
RE_CIN = re.compile(r'\b[UL][0-9]{5}[A-Z]{2}[0-9]{4}[A-Z]{3}[0-9]{6}\b', re.IGNORECASE)
RE_SIREN = re.compile(r'\b[0-9]{3}\s?[0-9]{3}\s?[0-9]{3}\b')
RE_EIN = re.compile(r'\b[0-9]{2}-[0-9]{7}\b')

# Postal codes
RE_IN_PIN = re.compile(r'\b[1-9][0-9]{5}\b')
RE_FR_POST = re.compile(r'\b(?:0[1-9]|[1-8][0-9]|9[0-8])[0-9]{3}\b')
RE_US_ZIP = re.compile(r'\b[0-9]{5}(?:-[0-9]{4})?\b')

# Phone & Domain
RE_PHONE_RAW = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,5}\)?[-.\s]?\d{3,5}[-.\s]?\d{3,5}')
RE_DOMAIN = re.compile(r'\b(?:https?://)?(?:www\.)?([a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?)\b', re.IGNORECASE)

COMMON_EMAIL_DOMAINS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'rediffmail.com', 'icloud.com', 'aol.com'
}

def extract_tax_ids(text: str, country: str) -> List[Tuple[str, str]]:
    if not text:
        return []
    ids = []
    c = country.upper()
    
    if c in ("INDIA", "IN"):
        for m in RE_GSTIN.findall(text):
            ids.append(("GSTIN", m.upper()))
        for m in RE_PAN.findall(text):
            ids.append(("PAN", m.upper()))
        for m in RE_CIN.findall(text):
            ids.append(("CIN", m.upper()))
    elif c in ("FRANCE", "FR"):
        for m in RE_SIREN.findall(text):
            clean_siren = m.replace(" ", "")
            ids.append(("SIREN", clean_siren))
    elif c in ("UNITED STATES", "USA", "US"):
        for m in RE_EIN.findall(text):
            ids.append(("EIN", m))
            
    return ids

def extract_postal_code(address: str, country: str) -> Optional[str]:
    if not address:
        return None
    c = country.upper()
    if c in ("INDIA", "IN"):
        m = RE_IN_PIN.findall(address)
        if m: return m[-1]
    elif c in ("FRANCE", "FR"):
        m = RE_FR_POST.findall(address)
        if m: return m[-1]
    elif c in ("UNITED STATES", "USA", "US"):
        m = RE_US_ZIP.findall(address)
        if m: return m[-1][:5]
    return None

def extract_phone_digits(text: str) -> Optional[str]:
    if not text:
        return None
    matches = RE_PHONE_RAW.findall(text)
    for m in matches:
        digits = re.sub(r'\D', '', m)
        if 10 <= len(digits) <= 13:
            return digits[-10:] # Standardize to last 10 digits
    return None

def extract_domain(text: str) -> Optional[str]:
    if not text:
        return None
    matches = RE_DOMAIN.findall(text)
    for d in matches:
        d_lower = d.lower()
        if d_lower not in COMMON_EMAIL_DOMAINS and len(d_lower) > 4:
            return d_lower
    return None

def extract_all_deterministic_anchors(name: str, address: str, country: str) -> Dict[str, str]:
    combined = f"{name} {address}"
    anchors = {}
    
    tax_ids = extract_tax_ids(combined, country)
    for t_type, t_val in tax_ids:
        anchors[t_type] = t_val
        
    pin = extract_postal_code(address, country)
    if pin:
        anchors["POSTAL"] = pin
        
    phone = extract_phone_digits(combined)
    if phone:
        anchors["PHONE"] = phone
        
    domain = extract_domain(combined)
    if domain:
        anchors["DOMAIN"] = domain
        
    return anchors
