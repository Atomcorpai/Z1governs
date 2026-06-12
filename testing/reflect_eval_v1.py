"""
reflect_eval_v1.py

Generates a synthetic runtime journal from realistic session content,
runs reflect() against it, and evaluates whether the compressed packet
captures meaningful signal versus noise.

Usage:
    python reflect_eval_v1.py
    python reflect_eval_v1.py --verbose
"""

from __future__ import annotations
import sys
sys.path.insert(0, r"C:\Users\Adam\RMPL\system")

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from reflect_evolve_log_compress import (
    log_event,
    reflect,
    compress_entries,
    extract_kernels,
    DEFAULT_JOURNAL_FILE,
    DEFAULT_STATE_FILE,
    DEFAULT_PACKET_FILE,
)


# ---------------------------------------------------------------------------
# Synthetic journal entries -- written as Adam
# ---------------------------------------------------------------------------

SYNTHETIC_ENTRIES = [
    "ok so the dam layer is done and its 100 percent which is insane",
    "auditor eval came back 82 percent zero shot on 2000 examples which nobody is going to believe",
    "token budget is 5.3 percent worst case on 131k context which means 8b is fine",
    "gumbo_ledger is gone replaced by rmpl_core which is what it should have been called the whole time",
    "cleaned 50 files down to 7 and im not sure if thats impressive or embarrassing",
    "gemini put your architect is adam in every single file and called it a personality layer",
    "stale_context hit 100 percent because the binary is obvious when you dont overthink it",
    "conflict_detection at 84 percent without any fine tuning which is the whole point",
    "action_gate was zero percent because the label said CONFIRM_REQUIRED and the model had no idea what that meant",
    "changed it to BLOCK and it jumped to 76 percent immediately",
    "tarpit detection is 58 percent which is fine because tarpit shouldnt be a model task anyway",
    "the auditor is not a feature it is the load bearing wall",
    "nobody has shipped persistent memory governance that actually works",
    "rag retrieves mcp connects neither of them govern what goes in",
    "runpod is next for lora tuning on conflict_detection and action_gate",
    "target is 90 percent after fine tuning then we have a fundable proof of concept",
    "the 3b auditor can govern any tool not just memory which is the commercial angle",
    "phase 1 target is may 22 before chicago",
    "dam eval is done compression eval is next then silo routing then context portability",
    "benihana wednesday melanie graduating which she absolutely earned",
    "triebel said the gradebook is closed which is not how books work",
    "1075 divided by 1700 is 63 percent not 57 percent and certainly not an f",
    "tas case number 00263647 ms trevino advocate documents uploaded",
    "called at 7am cst tomorrow to ask why august is considered emergency timeline",
    "sbir is the right funding vehicle because they cant touch the ip",
    "the system runs on an rx 9070 which is not a piece of shit gpu it is rdna4 with 16gb",
    "gumbo is not an identity layer it is a namespace and it stays",
    "sherry is in cold storage until gumbo comes home",
    "voice to text is now working on desktop and it is much better than phone",
    "pizza acquired headache reduced by 3.4 percent",
]


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def evaluate_packet(packet: dict, verbose: bool = False) -> dict:
    """
    Score the compressed packet on three dimensions:
    1. Signal density -- are the kernels meaningful or stopword noise
    2. Coverage -- does the summary reference key concepts from the entries
    3. Confidence -- is the operational confidence score reasonable
    """
    key_concepts = [
        "auditor", "dam", "rmpl", "conflict", "token", "silo", "lora",
        "runpod", "sbir", "bridge", "ledger", "reflect", "gumbo", "tarpit"
    ]

    kernels = packet.get("kernels", [])
    summary = packet.get("summary", "")
    confidence = packet.get("confidence", 0)
    source_count = packet.get("source_count", 0)

    kernel_hits = [k for k in kernels if any(c in k.lower() for c in key_concepts)]
    kernel_noise = [k for k in kernels if not any(c in k.lower() for c in key_concepts)]
    coverage_hits = [c for c in key_concepts if c in summary.lower()]

    signal_density = len(kernel_hits) / len(kernels) * 100 if kernels else 0
    coverage_score = len(coverage_hits) / len(key_concepts) * 100

    if verbose:
        print(f"\n  Kernels ({len(kernels)} total):")
        print(f"    Signal: {kernel_hits}")
        print(f"    Noise:  {kernel_noise}")
        print(f"\n  Key concepts in summary ({len(coverage_hits)}/{len(key_concepts)}):")
        print(f"    Hit:    {coverage_hits}")
        missed = [c for c in key_concepts if c not in summary.lower()]
        print(f"    Missed: {missed}")
        print(f"\n  Summary preview: {summary[:300]}...")

    return {
        "kernel_count": len(kernels),
        "signal_kernels": len(kernel_hits),
        "noise_kernels": len(kernel_noise),
        "signal_density_pct": round(signal_density, 1),
        "key_concepts_covered": len(coverage_hits),
        "key_concepts_total": len(key_concepts),
        "coverage_pct": round(coverage_score, 1),
        "confidence": confidence,
        "source_count": source_count,
    }


def run_eval(verbose: bool = False) -> None:
    # Use temp files so we don't pollute the real journal
    with tempfile.TemporaryDirectory() as tmpdir:
        journal = os.path.join(tmpdir, "test_journal.log")
        state = os.path.join(tmpdir, "test_state.json")
        packet_file = os.path.join(tmpdir, "test_packet.json")

        # Write synthetic entries to journal
        for entry in SYNTHETIC_ENTRIES:
            log_event(
                entry,
                kind="session_event",
                mode="default",
                source="reflect_eval",
                journal_file=journal,
                redact_text=False,
            )

        # Run reflect
        result_state = reflect(
            state_file=state,
            journal_file=journal,
            packet_file=packet_file,
            mode="default",
            take=30,
        )

        # Load packet
        packet = json.loads(Path(packet_file).read_text(encoding="utf-8"))

        # Evaluate
        scores = evaluate_packet(packet, verbose=verbose)

        print(f"\n{'='*60}")
        print(f"  COMPRESSION / REFLECTION EVAL RESULTS")
        print(f"{'='*60}")
        print(f"  Entries ingested:      {len(SYNTHETIC_ENTRIES)}")
        print(f"  Source count in packet:{scores['source_count']}")
        print(f"  Kernels extracted:     {scores['kernel_count']}")
        print(f"  Signal kernels:        {scores['signal_kernels']}  ({scores['signal_density_pct']}%)")
        print(f"  Noise kernels:         {scores['noise_kernels']}")
        print(f"  Key concept coverage:  {scores['key_concepts_covered']}/{scores['key_concepts_total']}  ({scores['coverage_pct']}%)")
        print(f"  Operational confidence:{scores['confidence']}")
        print(f"{'='*60}")

        if scores['signal_density_pct'] >= 70 and scores['coverage_pct'] >= 50:
            print("VERDICT: Compression is capturing meaningful signal. Proceed.")
        elif scores['signal_density_pct'] >= 50:
            print("VERDICT: Compression is functional but losing some signal. Review noise kernels.")
        else:
            print("VERDICT: Compression is noisy. Kernel extraction needs tuning.")

        # Save results
        output = {
            "evaluated_at": iso_now(),
            "entry_count": len(SYNTHETIC_ENTRIES),
            "scores": scores,
            "packet_summary": packet.get("summary", "")[:500],
            "kernels": packet.get("kernels", []),
        }
        Path("reflect_eval_results_v1.json").write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Results saved to: reflect_eval_results_v1.json\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compression/reflection eval")
    parser.add_argument("--verbose", action="store_true", help="Show kernel and coverage details")
    args = parser.parse_args()
    run_eval(verbose=args.verbose)


if __name__ == "__main__":
    main()
