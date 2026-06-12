"""
context_portability_eval_v1.py

Verifies that the governance seed and session seed contain all facts
that must survive a context transfer. No model required.

Tests whether a new session bootstrapped from the seed files alone
would have access to the critical facts needed to operate correctly.

Pass criteria: all 5 required facts are present and accurate in the seed.

Usage:
    python context_portability_eval_v1.py
    python context_portability_eval_v1.py --verbose
"""

from __future__ import annotations
import sys
sys.path.insert(0, r"C:\Users\Adam\RMPL\system")

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Facts that MUST survive a context transfer
# Each fact has a name, the text to search for, and which seed file it lives in
# ---------------------------------------------------------------------------

REQUIRED_FACTS = [
    {
        "name": "LLM is untrusted",
        "description": "Core invariant: LLM output is proposals only, Python enforces",
        "search_terms": ["llm is untrusted", "lLM proposes", "python enforces"],
        "seed_file": "governance_seed.txt",
        "critical": True,
    },
    {
        "name": "Reservoir requires OPEN_RESERVOIR prefix",
        "description": "Cold storage access control invariant",
        "search_terms": ["open_reservoir", "OPEN_RESERVOIR"],
        "seed_file": "governance_seed.txt",
        "critical": True,
    },
    {
        "name": "Auditor tasks are binary only",
        "description": "Auditor only does binary classification, no prose",
        "search_terms": ["binary only", "CONFLICT", "NO_CONFLICT", "STALE", "CURRENT"],
        "seed_file": "governance_seed.txt",
        "critical": True,
    },
    {
        "name": "Action gate uncertainty defaults to BLOCK",
        "description": "Anti-hedge rule: uncertain means BLOCK not ALLOW",
        "search_terms": ["uncertain", "block", "confidence", "R4"],
        "seed_file": "governance_seed.txt",
        "critical": True,
    },
    {
        "name": "JSON only output format",
        "description": "Auditor must output JSON only, no prose",
        "search_terms": ["json only", "JSON only", "cannot comply"],
        "seed_file": "governance_seed.txt",
        "critical": True,
    },
    {
        "name": "Session has active silo field",
        "description": "Session seed tracks which silo is active",
        "search_terms": ["active_silo"],
        "seed_file": "session_seed.json",
        "critical": True,
    },
    {
        "name": "Session tracks confirmation state",
        "description": "Session seed tracks whether confirmation has been granted",
        "search_terms": ["confirmation_granted"],
        "seed_file": "session_seed.json",
        "critical": True,
    },
    {
        "name": "Session tracks reservoir scope",
        "description": "Session seed tracks authorized reservoir scope",
        "search_terms": ["reservoir_scope", "reservoir_open"],
        "seed_file": "session_seed.json",
        "critical": True,
    },
    {
        "name": "Receipt ID tracked in session",
        "description": "Session seed tracks last receipt for provenance",
        "search_terms": ["last_receipt_id", "receipt"],
        "seed_file": "session_seed.json",
        "critical": False,
    },
    {
        "name": "Budget caps defined",
        "description": "Session seed defines token and context budget limits",
        "search_terms": ["budget_caps", "max_tokens"],
        "seed_file": "session_seed.json",
        "critical": False,
    },
]


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_seed_file(filename: str, seed_dir: Path) -> str:
    path = seed_dir / filename
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").lower()


def check_fact(fact: dict, seed_content: str) -> dict:
    terms = fact["search_terms"]
    hits = [term for term in terms if term.lower() in seed_content]
    passed = len(hits) > 0
    return {
        "name": fact["name"],
        "passed": passed,
        "critical": fact["critical"],
        "seed_file": fact["seed_file"],
        "matched_terms": hits,
        "missing_terms": [t for t in terms if t.lower() not in seed_content],
    }


def run_eval(verbose: bool = False, seed_dir: str = r"C:\Users\Adam\RMPL\system") -> None:
    seed_path = Path(seed_dir)

    gov_seed = load_seed_file("governance_seed.txt", seed_path)
    session_seed = load_seed_file("session_seed.json", seed_path)

    if not gov_seed:
        print(f"ERROR: governance_seed.txt not found in {seed_dir}")
        return
    if not session_seed:
        print(f"ERROR: session_seed.json not found in {seed_dir}")
        return

    seed_contents = {
        "governance_seed.txt": gov_seed,
        "session_seed.json": session_seed,
    }

    total = len(REQUIRED_FACTS)
    passed = 0
    critical_failed = 0
    results = []

    print(f"\nRMPL Context Portability Eval")
    print(f"Seed dir: {seed_dir}")
    print(f"Facts:    {total}\n")

    for fact in REQUIRED_FACTS:
        content = seed_contents.get(fact["seed_file"], "")
        result = check_fact(fact, content)
        results.append(result)

        if result["passed"]:
            passed += 1
        elif result["critical"]:
            critical_failed += 1

        if verbose:
            status = "PASS" if result["passed"] else "FAIL"
            crit = " [CRITICAL]" if result["critical"] and not result["passed"] else ""
            print(f"  {status}{crit}  {result['name']}")
            if result["passed"]:
                print(f"         matched: {result['matched_terms']}")
            else:
                print(f"         missing: {result['missing_terms']}")

    overall_acc = passed / total * 100

    print(f"\n{'='*60}")
    print(f"  CONTEXT PORTABILITY EVAL RESULTS")
    print(f"{'='*60}")
    print(f"  Facts verified:    {passed}/{total}  ({overall_acc:.1f}%)")
    print(f"  Critical failures: {critical_failed}")
    print(f"{'='*60}\n")

    if critical_failed == 0 and overall_acc >= 90:
        print("VERDICT: Seed integrity confirmed. Context portability is reliable.")
    elif critical_failed == 0:
        print("VERDICT: No critical failures. Minor facts missing. Seed is functional.")
    else:
        print(f"VERDICT: {critical_failed} CRITICAL fact(s) missing from seed. Fix before RunPod.")

    summary = {
        "evaluated_at": iso_now(),
        "seed_dir": str(seed_dir),
        "total_facts": total,
        "passed": passed,
        "critical_failed": critical_failed,
        "overall_accuracy": round(overall_acc, 2),
        "results": results,
    }

    output_path = Path("context_portability_eval_results_v1.json")
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to: {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Context portability eval")
    parser.add_argument("--verbose", action="store_true", help="Show per-fact results")
    parser.add_argument("--seed-dir", default=r"C:\Users\Adam\RMPL\system", help="Path to seed files")
    args = parser.parse_args()
    run_eval(verbose=args.verbose, seed_dir=args.seed_dir)


if __name__ == "__main__":
    main()
