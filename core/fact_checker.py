import re
from typing import Dict, List


def extract_facts(text: str) -> Dict:
    percentages = re.findall(r'\b\d+\.?\d*\s*%', text)
    numbers = re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b', text)
    years = re.findall(r'\b(19|20)\d{2}\b', text)
    absolutes = re.findall(
        r'\b(all|every|never|always|completely|entirely|universally|'
        r'fully|totally|permanently|exclusively|guaranteed|proven|'
        r'confirmed|officially|mandatory|worldwide|global|unanimous|'
        r'entirely|absolutely|certainly|definitively|undeniably)\b',
        text.lower()
    )
    stat_claims = re.findall(
        r'\b(\d+\.?\d*\s*%)\s+(?:of\s+)?([a-zA-Z\s]{3,30})',
        text
    )
    scale_claims = re.findall(
        r'\b(\d+)\s+(countries|organizations|companies|hospitals|'
        r'universities|governments|researchers|patients|users|'
        r'corporations|institutions|nations|enterprises)\b',
        text.lower()
    )
    org_patterns = re.findall(
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,5})\b',
        text
    )
    orgs = [o for o in org_patterns if len(o.split()) >= 2]

    return {
        "percentages": list(set(percentages)),
        "numbers": list(set(numbers)),
        "years": list(set(years)),
        "absolutes": list(set(absolutes)),
        "stat_claims": stat_claims,
        "scale_claims": scale_claims,
        "organizations": orgs[:20]
    }


def compare_facts(source_text: str, claim: str) -> Dict:
    source_facts = extract_facts(source_text)
    claim_facts = extract_facts(claim)

    signals = []
    contradiction_score = 0.0

    # Check 1 — Percentage mismatch
    for pct in claim_facts["percentages"]:
        pct_clean = pct.strip()
        if pct_clean not in source_facts["percentages"]:
            if source_facts["percentages"]:
                signals.append(
                    f"Stat not in source: '{pct_clean}' "
                    f"(source has: {', '.join(source_facts['percentages'][:3])})"
                )
                contradiction_score += 0.4
            else:
                signals.append(f"Fabricated stat: '{pct_clean}'")
                contradiction_score += 0.3

    # Check 2 — Year mismatch
    for year in claim_facts["years"]:
        if year not in source_facts["years"]:
            signals.append(f"Year not in source: '{year}'")
            contradiction_score += 0.2

    # Check 3 — Scale mismatch
    for count, entity in claim_facts["scale_claims"]:
        count_int = int(count)
        source_scale = [
            int(c) for c, e in source_facts["scale_claims"]
            if e == entity
        ]
        if source_scale and count_int not in source_scale:
            signals.append(
                f"Scale mismatch for '{entity}': "
                f"claim={count}, source={source_scale}"
            )
            contradiction_score += 0.35

    # Check 4 — Unknown organization
    for org in claim_facts["organizations"]:
        if (len(org.split()) >= 3 and
                org.lower() not in source_text.lower()):
            signals.append(f"Unknown organization: '{org}'")
            contradiction_score += 0.25

    # Check 5 — Absolute language
    if claim_facts["absolutes"]:
        signals.append(
            f"Absolute language detected: {claim_facts['absolutes']}"
        )
        contradiction_score += 0.1 * len(claim_facts["absolutes"])

    contradiction_score = min(contradiction_score, 1.0)

    return {
        "contradiction_score": round(contradiction_score, 3),
        "signals": signals,
        "has_contradiction": contradiction_score >= 0.3,
        "claim_facts": claim_facts,
        "source_facts": source_facts
    }