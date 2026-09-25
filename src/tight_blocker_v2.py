"""
src/tight_blocker_v2.py
=======================
Ultra-Tight Candidate Blocker v2 for Business Entity Resolution
Amazon ML Challenge 2026

Engineered for:
 1. Super-Clean Name Normalization (diacritics, dba, metadata, legal suffixes, condensed)
 2. Hierarchical Multi-Key Inverted Indexing
 3. Compact Candidate Sets (Average ≤8 candidates per S1 entity)
 4. Near-zero false-positive rate through geographic anchoring
"""
import os
import re
import csv
from collections import defaultdict
from src.features_18d import super_clean_name, US_STATES

RE_NUM = re.compile(r'\b\d+\b')
RE_US_STATE = re.compile(r'\b([A-Z]{2})\b')

INDIA_GEO = {
    'andhra','arunachal','assam','bihar','chhattisgarh','goa','gujarat','haryana',
    'himachal','jharkhand','karnataka','kerala','madhya','maharashtra','manipur',
    'meghalaya','mizoram','nagaland','odisha','orissa','punjab','rajasthan','sikkim',
    'tamil','nadu','telangana','tripura','uttar','pradesh','uttarakhand','bengal',
    'delhi','chandigarh','pondicherry','puducherry','ladakh','kashmir',
    'mumbai','delhi','bangalore','bengaluru','hyderabad','ahmedabad','chennai','kolkata',
    'surat','pune','jaipur','lucknow','kanpur','nagpur','indore','thane','bhopal',
    'visakhapatnam','vizag','patna','vadodara','baroda','ghaziabad','ludhiana','agra',
    'nashik','faridabad','meerut','rajkot','varanasi','srinagar','aurangabad','dhanbad',
    'amritsar','navi','allahabad','prayagraj','ranchi','howrah','coimbatore','jabalpur',
    'gwalior','vijayawada','jodhpur','madurai','raipur','kota','guwahati','chandigarh',
    'solapur','hubli','dharwad','bareilly','moradabad','mysore','mysuru','gurgaon','gurugram',
    'aligarh','jalandhar','tiruchirappalli','trichy','bhubaneswar','salem','warangal',
    'mira','bhayandar','thiruvananthapuram','trivandrum','bhiwandi','saharanpur','gorakhpur',
    'guntur','bikaner','amravati','noida','jamshedpur','bhilai','cuttack','firozabad',
    'kochi','cochin','nellore','bhavnagar','dehradun','durgapur','asansol','rourkela',
    'nanded','kolhapur','ajmer','akola','gulbarga','kalaburagi','jamnagar','ujjain',
    'loni','siliguri','jhansi','ulhasnagar','jammu','sangli','mangalore','mangaluru',
    'erode','belgaum','belagavi','ambattur','tirunelveli','malegaon','gaya','udaipur',
    'kakinada','davanagere','kozhikode','calicut','rajahmundry','bokaro','bellary',
    'patiala','agartala','bhagalpur','muzaffarnagar','latur','dhule','tirupati','rohtak',
    'korba','bhilwara','berhampur','muzaffarpur','ahmednagar','mathura','kollam',
    'avadi','kadapa','bilaspur','satara','bijapur','vijayapura','shivamogga','shimoga',
    'chandrapur','junagadh','ambala','karauli','mirzapur','abohar','khordha'
}

FRANCE_GEO = {
    'auvergne','rhone','alpes','bretagne','bourgogne','normandie','occitanie',
    'aquitaine','nouvelle','alsace','lorraine','champagne','ardenne','picardie',
    'languedoc','roussillon','provence','cote','dazur','limousin',
    'poitou','charentes','franche','comte','ile','france','hauts','loire','centre',
    'corse','paris','marseille','lyon','toulouse','nice','nantes','montpellier','strasbourg',
    'bordeaux','lille','rennes','reims','toulon','etienne','havre',
    'grenoble','dijon','angers','nimes','villeurbanne','clermont','ferrand','mans',
    'aix','brest','tours','amiens','limoges','annecy','perpignan',
    'boulogne','billancourt','metz','besancon','orleans','argenteuil',
    'rouen','montreuil','mulhouse','caen','nancy','tourcoing','roubaix','nanterre',
    'vitry','seine','avignon','creteil','dunkerque','poitiers','asnieres','versailles',
    'colombes','nazaire','herblain','teste','buch',
    'aubervilliers','aulnay','courbevoie','cherbourg','calais','rochelle','beziers'
}


def extract_geo(addr: str, country: str) -> str:
    if not addr: return ""
    addr_lower = addr.lower()
    if country == "us":
        for m in RE_US_STATE.finditer(addr):
            if m.group(1) in US_STATES:
                return m.group(1)
    elif country == "india":
        tokens = set(re.findall(r'[a-z]+', addr_lower))
        for geo in INDIA_GEO:
            if geo in tokens:
                return geo
    elif country == "france":
        tokens = set(re.findall(r'[a-z]+', addr_lower))
        for geo in FRANCE_GEO:
            if geo in tokens:
                return geo
    return ""


def extract_numbers(addr: str) -> str:
    if not addr: return ""
    nums = RE_NUM.findall(addr)
    return nums[0] if nums else ""


def build_indices(source_files: list) -> tuple:
    """
    Build multi-key inverted index over S2 and S3 files.
    """
    idx = {
        'exact_geo':        defaultdict(list),  # (country, geo, clean_name)
        'condensed_geo':    defaultdict(list),  # (country, geo, condensed_name)
        'condensed_no_geo': defaultdict(list),  # (country, condensed_name)
        'geo_word':         defaultdict(list),  # (country, geo, first_word)
        'geo_tok':          defaultdict(list),  # (country, geo, token)
        'geo_num':          defaultdict(list),  # (country, geo, street_num)
        'word_num':         defaultdict(list),  # (country, first_word, street_num)
    }
    records = {}

    for filepath in source_files:
        if not os.path.exists(filepath):
            continue
        print(f"[*] Indexing {os.path.basename(filepath)} with Blocker v2...")
        with open(filepath, encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader)
            for row in reader:
                if len(row) < 4: continue
                eid = row[0].strip()
                name = row[1].strip()
                addr = row[2].strip()
                country = row[3].strip().lower()

                cn = super_clean_name(name)
                cond = cn.replace(' ', '')
                geo = extract_geo(addr, country)
                num = extract_numbers(addr)
                words = cn.split()
                w1 = words[0] if words else ""
                toks = [t for t in words if len(t) >= 4]

                records[eid] = (cn, addr, country, geo)
                if not cn: continue

                if geo:
                    idx['exact_geo'][(country, geo, cn)].append(eid)
                    if cond:
                        idx['condensed_geo'][(country, geo, cond)].append(eid)
                    if w1:
                        idx['geo_word'][(country, geo, w1)].append(eid)
                    for tok in toks:
                        idx['geo_tok'][(country, geo, tok)].append(eid)
                    if num:
                        idx['geo_num'][(country, geo, num)].append(eid)

                if cond:
                    idx['condensed_no_geo'][(country, cond)].append(eid)
                if w1 and num:
                    idx['word_num'][(country, w1, num)].append(eid)

    return idx, records


def get_candidates_v2(s1_row: list, idx: dict, max_candidates: int = 8) -> list:
    """
    Ultra-tight hierarchical retrieval for S1 entity.
    """
    if len(s1_row) < 4:
        return []

    name = s1_row[1].strip()
    addr = s1_row[2].strip()
    country = s1_row[3].strip().lower()

    cn = super_clean_name(name)
    cond = cn.replace(' ', '')
    geo = extract_geo(addr, country)
    num = extract_numbers(addr)
    words = cn.split()
    w1 = words[0] if words else ""
    toks = [t for t in words if len(t) >= 4]

    seen = set()
    candidates = []

    def add_hits(hits, cap=None):
        added = 0
        for eid in hits:
            if eid not in seen:
                seen.add(eid)
                candidates.append(eid)
                added += 1
                if cap is not None and added >= cap:
                    break

    # Tier 1: (country, geo, exact_name) — UNCAPPED (high precision)
    if geo and cn:
        add_hits(idx['exact_geo'].get((country, geo, cn), []))

    if len(candidates) >= max_candidates:
        return candidates

    # Tier 2: (country, geo, condensed_name)
    if geo and cond:
        add_hits(idx['condensed_geo'].get((country, geo, cond), []),
                 cap=max_candidates - len(candidates))

    if len(candidates) >= max_candidates:
        return candidates

    # Tier 3: (country, condensed_name) without geo — only if rare name (<=8 hits)
    if cond:
        no_geo = idx['condensed_no_geo'].get((country, cond), [])
        if len(no_geo) <= 8:
            add_hits(no_geo, cap=max_candidates - len(candidates))

    if len(candidates) >= max_candidates:
        return candidates

    # Tier 4: (country, geo, first_word)
    if geo and w1:
        add_hits(idx['geo_word'].get((country, geo, w1), []),
                 cap=max_candidates - len(candidates))

    if len(candidates) >= max_candidates:
        return candidates

    # Tier 5: (country, geo, significant_token)
    if geo:
        for tok in toks:
            if len(candidates) >= max_candidates:
                break
            add_hits(idx['geo_tok'].get((country, geo, tok), []),
                     cap=max_candidates - len(candidates))

    if len(candidates) >= max_candidates:
        return candidates

    # Tier 6: (country, geo, street_num)
    if geo and num:
        add_hits(idx['geo_num'].get((country, geo, num), []),
                 cap=max_candidates - len(candidates))

    if len(candidates) >= max_candidates:
        return candidates

    # Tier 7: (country, first_word, street_num)
    if w1 and num:
        add_hits(idx['word_num'].get((country, w1, num), []),
                 cap=max_candidates - len(candidates))

    return candidates
