"""
dataset/builder/facts.py

Loads any domain schema from dataset/seeds/<domain>.yaml and generates a
deterministic dictionary of random seed facts for one document instance.

The RNG is seeded by (language, domain, document_index) using SHA-256 so the
same inputs ALWAYS produce the same facts — on any machine, any Python version,
any time. This reproducibility guarantee makes our evaluation
mathematically certain.

Currently Supported domains: lease, employment, health_insurance, immigration_letter

Author: DocuNative Team
Issue: #28
"""

import random
import hashlib
import yaml
from pathlib import Path
from datetime import date, timedelta
from typing import Any

# Path config
# Resolve relative to this file so the script works from any directory.
SEEDS_DIR = Path(__file__).parent.parent / "seeds"

SUPPORTED_DOMAINS = ["lease", "employment", "health_insurance", "immigration_letter"]


# Deterministic seeding 

def _make_seed(language: str, domain: str, document_index: int) -> int:
    """
    Create a stable integer seed from (language, domain, document_index).

    Uses SHA-256 because Python's built-in hash() is randomised per-process
    and is NOT stable across Python versions. SHA-256 always gives the same
    output for the same input.
    """
    key = f"{language}:{domain}:{document_index}"
    hash_bytes = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(hash_bytes[:8], byteorder="big")


# --- Schema loading ---

def load_schema(domain: str) -> dict:
    """
    Load the YAML schema for a given domain.
    Raises a clear error if the domain doesn't exist.
    """
    schema_path = SEEDS_DIR / f"{domain}.yaml"
    if not schema_path.exists():
        available = [p.stem for p in SEEDS_DIR.glob("*.yaml")]
        raise FileNotFoundError(
            f"Schema not found: {schema_path}\n"
            f"Available domains: {available}"
        )
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# -- Field generators ---

def _generate_field(field_name: str, field_def: dict,
                    rng: random.Random, resolved: dict) -> Any:
    """
    Generate one field value from its type definition.

    Args:
        field_name: Name of the field (used in error messages)
        field_def:  Field definition dict from the YAML schema
        rng:        Seeded Random instance — ensures determinism
        resolved:   Already-generated fields (needed for formula type)

    Returns:
        Generated value: int, float, str, or bool
    """
    field_type = field_def["type"]

    if field_type == "money":
        min_val  = field_def["min"]
        max_val  = field_def["max"]
        step     = field_def.get("step", 1)
        currency = field_def["currency"]
        steps    = (max_val - min_val) // step
        amount   = min_val + rng.randint(0, steps) * step
        return f"{amount} {currency}"

    elif field_type == "integer":
        return rng.randint(field_def["min"], field_def["max"])

    elif field_type == "date":
        start = date.fromisoformat(field_def["min"])
        end   = date.fromisoformat(field_def["max"])
        delta = (end - start).days
        d = start + timedelta(days=rng.randint(0, delta))
        # Snap to first of month — legal documents almost always start on the 1st
        return date(d.year, d.month, 1).isoformat()

    elif field_type == "choice":
        return rng.choice(field_def["options"])

    elif field_type == "formula":
        # Create a safe math context from already resolved variables.
        # Strip currency strings so "1200 EUR" becomes 1200.
        math_context = {}
        for k, v in resolved.items():
            if isinstance(v, str) and len(str(v).split()) > 1 and str(v).split()[0].isdigit():
                math_context[k] = int(str(v).split()[0])
            else:
                math_context[k] = v
                
        try:
            # eval() natively maps the variables in the string to the math_context dict!
            return int(eval(field_def["expression"], {"__builtins__": {}}, math_context))
        except NameError as e:
            raise ValueError(
                f"Formula error in '{field_name}': {e}. "
                f"Make sure dependencies are defined BEFORE this formula in the YAML!"
            )

    else:
        raise ValueError(
            f"Unknown field type '{field_type}' for field '{field_name}'. "
            f"Supported types: money, integer, date, choice, formula"
        )


# -- Core function ---

def generate_facts(language: str, domain: str, document_index: int) -> dict:
    """
    Generate a deterministic dictionary of seed facts for one document.

    The SAME (language, domain, document_index) triple will ALWAYS produce
    the SAME dictionary. This is the reproducibility guarantee.
    """
    schema = load_schema(domain)
    seed   = _make_seed(language, domain, document_index)
    rng    = random.Random(seed)
    facts  = {}

    # Process fields in order — formula fields depend on earlier fields
    for field_name, field_def in schema["fields"].items():
        # Pass the `facts` dictionary we are building directly as the 'resolved' context
        value = _generate_field(field_name, field_def, rng, facts)
        facts[field_name] = value

    # Metadata — preserved in JSONL output for full auditability
    facts["_language"]       = language
    facts["_domain"]         = domain
    facts["_document_index"] = document_index
    facts["_seed"]           = seed

    return facts


# --- Quick test ---

if __name__ == "__main__":
    import json

    print("Testing facts.py across all 4 domains...\n")

    # Test 1: One document per domain 
    for domain in SUPPORTED_DOMAINS:
        facts = generate_facts("en", domain, 0)
        print(f"=== {domain} / en / doc 0 ===")
        # Print only the non-metadata fields for readability
        display = {k: v for k, v in facts.items() if not k.startswith("_")}
        print(json.dumps(display, indent=2, default=str))
        print()

    # Test 2: Determinism check 
    for domain in SUPPORTED_DOMAINS:
        a = generate_facts("de", domain, 0)
        b = generate_facts("de", domain, 0)
        assert a == b, f"FAIL: {domain} — same inputs gave different outputs!"
    print("Determinism check passed — same inputs always give same output")

    # Test 3: Different index → different facts
    for domain in SUPPORTED_DOMAINS:
        a = generate_facts("de", domain, 0)
        b = generate_facts("de", domain, 1)
        assert a != b, f"FAIL: {domain} — different indices gave identical facts!"
    print("Uniqueness check passed — different indices give different facts")

    # Test 4: Different language → different facts 
    for domain in SUPPORTED_DOMAINS:
        a = generate_facts("de", domain, 0)
        b = generate_facts("id", domain, 0)
        assert a != b, f"FAIL: {domain} — different languages gave identical facts!"
    print("Cross-language check passed — de/0 and id/0 are different")

    #  Test 5: Summary across all 3 eval languages 
    print("\n=== Key facts across languages (doc index 0) ===")

    print("\nLEASE")
    for lang in ["de", "hi", "id"]:
        f = generate_facts(lang, "lease", 0)
        print(f"  {lang}: rent={f['monthly_rent']}, "
              f"deposit={f['deposit_amount']} EUR, "
              f"notice={f['notice_period_days']} days, "
              f"pets={f['pets_allowed']}")

    print("\nEMPLOYMENT")
    for lang in ["de", "hi", "id"]:
        f = generate_facts(lang, "employment", 0)
        print(f"  {lang}: salary={f['monthly_salary']}, "
              f"leave={f['annual_leave_days']} days, "
              f"probation={f['probation_months']} months, "
              f"remote={f['remote_work_days_per_week']} days/week")

    print("\nHEALTH INSURANCE")
    for lang in ["de", "hi", "id"]:
        f = generate_facts(lang, "health_insurance", 0)
        print(f"  {lang}: premium={f['monthly_premium']}, "
              f"deductible={f['annual_deductible']}, "
              f"dental={f['dental_coverage_included']}, "
              f"waiting={f['waiting_period_days']} days")

    print("\nIMMIGRATION LETTER")
    for lang in ["de", "hi", "id"]:
        f = generate_facts(lang, "immigration_letter", 0)
        print(f"  {lang}: permit={f['permit_type']}, "
              f"duration={f['permit_duration_months']} months, "
              f"response_deadline={f['response_deadline_days']} days, "
              f"work={f['work_permitted']}")

    print("\nAll tests passed. facts.py is ready for all 4 domains.")
