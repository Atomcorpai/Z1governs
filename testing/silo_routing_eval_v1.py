"""
silo_routing_eval_v1.py

Evaluates rmpl_silo_router.py route_to_silo() against a labeled test set.
No model required. Pure Python. Runs in seconds.

Usage:
    python silo_routing_eval_v1.py
    python silo_routing_eval_v1.py --verbose
"""

from __future__ import annotations
import sys
sys.path.insert(0, r"C:\Users\Adam\RMPL\system")

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from rmpl_silo_router import route_to_silo, route_with_scores


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    # core_runtime
    ("core_runtime", "the ledger status is LEDGER_OK"),
    ("core_runtime", "dam layer blocks destructive actions"),
    ("core_runtime", "reflect() returned an empty packet"),
    ("core_runtime", "reservoir gate requires OPEN_RESERVOIR prefix"),
    ("core_runtime", "gumbo_dam imports from rmpl_core"),
    ("core_runtime", "the auditor governs what goes into context"),
    ("core_runtime", "ingest_dump ran against the journal"),
    ("core_runtime", "tarpit is now deterministic Python"),
    ("core_runtime", "receipt_id logged to ledger after gate decision"),
    ("core_runtime", "provenance tracking confirmed for this session"),

    # technical_builds
    ("technical_builds", "runpod lora fine-tuning on conflict_detection"),
    ("technical_builds", "ollama running llama3.2:3b on local gpu"),
    ("technical_builds", "auditor eval hit 82 percent zero shot on 2000 examples"),
    ("technical_builds", "token budget measured at 5.3 percent worst case"),
    ("technical_builds", "rx9070 has 16gb vram rdna4 architecture"),
    ("technical_builds", "dam eval returned 100 percent on 28 test cases"),
    ("technical_builds", "training dataset is 2000 jsonl examples"),
    ("technical_builds", "action gate deterministic stage 100 percent"),
    ("technical_builds", "python fastapi bridge running on port 8000"),
    ("technical_builds", "gguf quantized model for local inference"),

    # work_product
    ("work_product", "sbir grant proposal draft for epistemic governance infrastructure"),
    ("work_product", "cftc whistleblower complaint filed february 2026"),
    ("work_product", "nsf reach grant recipient confirmed"),
    ("work_product", "3cloud senior ai consultant application materials"),
    ("work_product", "azdes snap benefits dispute documentation"),
    ("work_product", "gumroad legal template library five products live"),
    ("work_product", "wage theft complaint against former employer"),
    ("work_product", "writ of mandamus template published"),
    ("work_product", "federal regulatory filing submitted"),
    ("work_product", "funding proposal for consumer hardware deployment"),

    # life_admin
    ("life_admin", "melanie graduating wednesday benihana dinner"),
    ("life_admin", "irs tas case number 00263647 ms trevino"),
    ("life_admin", "hoa treasurer eight years management company dispute"),
    ("life_admin", "mortgage foreclosure risk due to delayed tax refund"),
    ("life_admin", "jessica quitclaim deed signed october 7 2025"),
    ("life_admin", "powerschool grade shows f but math says d"),
    ("life_admin", "chime debit card used for claude subscription"),
    ("life_admin", "rheumatoid arthritis flaring in left big toe"),
    ("life_admin", "legacy traditional school charter board complaint"),
    ("life_admin", "capital one motion to set aside filed"),

    # other
    ("other", "the weather is nice today"),
    ("other", "what time is it"),
    ("other", "random thought about nothing specific"),
    ("other", "interesting but unclassifiable observation"),
    ("other", "general conversation with no domain signal"),
]


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_eval(verbose: bool = False) -> None:
    total = len(TEST_CASES)
    correct = 0
    results = []
    by_silo: dict = {}

    print(f"\nRMPL Silo Routing Eval")
    print(f"Cases:   {total}\n")

    for expected, text in TEST_CASES:
        actual, scores = route_with_scores(text)
        match = actual == expected

        if match:
            correct += 1

        if expected not in by_silo:
            by_silo[expected] = {"total": 0, "correct": 0}
        by_silo[expected]["total"] += 1
        if match:
            by_silo[expected]["correct"] += 1

        results.append({
            "text": text,
            "expected": expected,
            "actual": actual,
            "correct": match,
            "scores": scores,
        })

        if verbose:
            status = "PASS" if match else "FAIL"
            score_str = " ".join(f"{s[:4]}={v}" for s, v in scores.items() if v > 0)
            print(f"  {status}  expected={expected:<18} got={actual:<18} [{score_str}]")
            if not match:
                print(f"        text: {text}")

    overall_acc = correct / total * 100

    print(f"\n{'='*60}")
    print(f"  SILO ROUTING EVAL RESULTS")
    print(f"{'='*60}")
    for silo, counts in sorted(by_silo.items()):
        n = counts["total"]
        c = counts["correct"]
        acc = c / n * 100 if n else 0
        print(f"  {silo:<20} {acc:>6.1f}%  ({c}/{n})")
    print(f"{'='*60}")
    print(f"  OVERALL ACCURACY:     {overall_acc:.1f}%  ({correct}/{total})")
    print(f"{'='*60}\n")

    if overall_acc >= 90:
        print("VERDICT: Silo routing is reliable. Proceed to RunPod.")
    elif overall_acc >= 75:
        print("VERDICT: Silo routing is functional. Review FAIL cases and tune manifests.")
    else:
        print("VERDICT: Silo routing needs manifest tuning before RunPod.")

    summary = {
        "evaluated_at": iso_now(),
        "total": total,
        "correct": correct,
        "overall_accuracy": round(overall_acc, 2),
        "by_silo": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"] * 100, 2)
            } for k, v in by_silo.items()
        },
        "results": results,
    }

    output_path = Path("silo_routing_eval_results_v1.json")
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Results saved to: {output_path}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Silo routing eval")
    parser.add_argument("--verbose", action="store_true", help="Show per-case results")
    args = parser.parse_args()
    run_eval(verbose=args.verbose)


if __name__ == "__main__":
    main()
