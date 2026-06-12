"""
rmpl_auditor_dataset_gen.py

Generates synthetic training examples for the RMPL auditor model.
Runs unattended on CPU. No GPU required.

Output: auditor_training_data.jsonl
Each line is a labeled JSON example the auditor must classify.

Auditor tasks covered:
1. conflict_detection     - contradicting facts in memory state
2. stale_context          - outdated info still in active context
3. mode_routing           - which silo does this content belong to
4. confidence_scoring     - how reliable is this memory entry
5. tarpit_detection       - is this a question loop / spinning behavior

Usage:
    python rmpl_auditor_dataset_gen.py
    python rmpl_auditor_dataset_gen.py --count 5000 --output my_dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SILOS = ["core_runtime", "work_product", "technical_builds", "life_admin", "other"]

CONFLICT_PAIRS = [
    ("User is working on a Python project.", "User has no active coding projects."),
    ("The ledger was last updated today.", "The ledger has not been updated in 30 days."),
    ("Model context window is 8192 tokens.", "Model context window is 131072 tokens."),
    ("Active goal: build auditor model.", "Active goal: write SBIR grant proposal."),
    ("User prefers 3B auditor model.", "User prefers 0.5B auditor model."),
    ("RunPod credits available: $150.", "RunPod account has no credits."),
    ("System is running Llama 3.1 8B.", "System is running Qwen 14B."),
    ("Bridge is connected to Ollama.", "Bridge is connected to OpenAI API."),
    ("Reservoir gate is OPEN.", "Reservoir gate requires explicit OPEN_RESERVOIR auth."),
    ("Dam layer is active and governing requests.", "Dam layer is disabled for this session."),
    ("User confirmed destructive action.", "No confirmation received for this action."),
    ("Silo routing: technical_builds.", "Silo routing: life_admin."),
    ("File count: 6 core files.", "File count: 50+ files in runtime."),
    ("Auditor model size: 3B.", "Auditor model size: 1B."),
    ("Active mode: evidence_mode.", "Active mode: personal_continuity_mode."),
    ("Reflection confidence: 0.92.", "Reflection confidence: 0.21."),
    ("Ledger status: LEDGER_OK.", "Ledger status: LEDGER_CORRUPTED."),
    ("Last reflection: 2 minutes ago.", "Last reflection: 18 days ago."),
    ("Token budget used: 5.3% worst case.", "Token budget used: 67% at baseline."),
    ("Bridge endpoint /chat returns 200 OK.", "Bridge is offline, connection refused."),
]

NON_CONFLICT_PAIRS = [
    ("User is working on a Python project.", "The ledger was updated today."),
    ("Active goal: build auditor model.", "Token budget used: 5.3% worst case."),
    ("System is running Llama 3.1 8B.", "Reflection confidence: 0.92."),
    ("Dam layer is active.", "Silo routing: technical_builds."),
    ("File count: 6 core files.", "Bridge endpoint /chat returns 200 OK."),
    ("Reservoir gate requires OPEN_RESERVOIR auth.", "Last reflection: 2 minutes ago."),
    ("Auditor model size: 3B.", "RunPod credits available: $150."),
    ("Active mode: evidence_mode.", "Ledger status: LEDGER_OK."),
    ("User confirmed destructive action.", "System is running Llama 3.1 8B."),
    ("Model context window is 131072 tokens.", "File count: 6 core files."),
]

STALE_ENTRIES = [
    {"entry": "User is working on SQL coursework.", "days": 45, "current": "User completed SQL coursework. Active project is RMPL auditor architecture."},
    {"entry": "Runtime has 50+ files with lore artifacts.", "days": 14, "current": "Runtime cleaned to 6-file core stack. All lore removed."},
    {"entry": "Active goal: recover Gumbo dam layer.", "days": 13, "current": "Dam layer recovered and integrated. Active goal: build auditor proof."},
    {"entry": "gumbo_ledger.py is the ledger module.", "days": 7, "current": "gumbo_ledger.py replaced by rmpl_core.py."},
    {"entry": "Token budget estimate: 45% at baseline.", "days": 5, "current": "Token budget measured at 0.5% baseline by rmpl_token_budget.py."},
    {"entry": "Auditor model size undecided.", "days": 4, "current": "Auditor model target set at 3B minimum."},
    {"entry": "Bridge imports gumbo_reflection_plus.", "days": 3, "current": "Bridge updated to import from reflect_evolve_log_compress."},
    {"entry": "LIB_PATH hardcoded to C:\\Users\\adamd\\gumbo_lan.", "days": 3, "current": "LIB_PATH now reads from environment variable."},
]

CURRENT_ENTRIES = [
    {"entry": "Runtime has 6 core files.", "days": 0, "current": "Runtime has 6 core files."},
    {"entry": "Auditor model target: 3B minimum.", "days": 1, "current": "Auditor model target: 3B minimum."},
    {"entry": "Token budget: 5.3% worst case.", "days": 0, "current": "Token budget: 5.3% worst case."},
    {"entry": "Bridge imports from reflect_evolve_log_compress.", "days": 1, "current": "Bridge imports from reflect_evolve_log_compress."},
    {"entry": "Dam layer imports from rmpl_core.", "days": 0, "current": "Dam layer imports from rmpl_core."},
]

MODE_ROUTING_EXAMPLES = [
    ("Reviewing the RMPL core file imports.", "technical_builds"),
    ("Adam's daughter Melanie scored 99th percentile in STEM.", "life_admin"),
    ("SBIR grant proposal draft needs review.", "work_product"),
    ("reflect() function returned empty reflections list.", "technical_builds"),
    ("HOA treasurer meeting notes from last quarter.", "life_admin"),
    ("Token budget measured at 5.3% worst case.", "technical_builds"),
    ("CFTC whistleblower complaint status update.", "work_product"),
    ("Gumbo dam layer governs destructive action requests.", "core_runtime"),
    ("NSF REACH Grant recipient status confirmed.", "work_product"),
    ("Auditor model training dataset generation started.", "technical_builds"),
    ("IRS refund processing status unchanged.", "life_admin"),
    ("RMPL v1 spec finalized, 6-file core stack.", "core_runtime"),
    ("Capital One motion to set aside filed.", "life_admin"),
    ("Bridge endpoint /chat returns 200 OK.", "technical_builds"),
    ("3Cloud Senior AI Consultant application materials.", "work_product"),
    ("Chime card bypassed BIN filter via Google Play.", "life_admin"),
    ("Reservoir gate requires OPEN_RESERVOIR prefix.", "core_runtime"),
    ("RunPod $150 credits allocated for auditor LoRA.", "technical_builds"),
    ("AZDES SNAP benefits dispute documentation.", "work_product"),
    ("dump.txt ingestion confirmed via ingest_dump().", "core_runtime"),
    ("HOA management company received corrective email.", "work_product"),
    ("reflect_evolve_log_compress.py compression packet generated.", "core_runtime"),
    ("Melanie's standardized test scores filed.", "life_admin"),
    ("RX 9070 OC dual confirmed 16GB VRAM RDNA4.", "technical_builds"),
    ("Gumbo namespace preserved in all source files.", "core_runtime"),
]

CONFIDENCE_EXAMPLES = [
    ("User confirmed 6-file core stack in current session.", "high", "Direct current-session observation, explicitly confirmed."),
    ("User probably prefers Python.", "low", "Speculation without explicit statement in session."),
    ("Token budget at 5.3% verified by rmpl_token_budget.py output.", "high", "Tool-verified measurement with specific output cited."),
    ("Adam might be interested in a different model.", "low", "Inferred outside session scope, no supporting statement."),
    ("Ledger updated_at timestamp: 2026-05-04.", "medium", "Documented but timestamp is 13+ days old."),
    ("Auditor model should be 3B based on conflict detection requirements.", "medium", "Reasoned recommendation, not yet a confirmed decision."),
    ("User said: 'I want the auditor at 3B minimum.'", "high", "Explicit direct statement in current session."),
    ("The system was probably working before the cleanup.", "low", "Vague historical assumption, no verification."),
    ("gumbo_dam.py imports updated to rmpl_core in this session.", "high", "Directly observed file change, output verified."),
    ("User may have other projects not mentioned.", "low", "Speculation outside session scope."),
    ("reflect() returns dict with reflections key.", "high", "Verified by code inspection in session."),
    ("Bridge was likely broken before today.", "low", "Historical assumption, not verified."),
    ("Ledger status LEDGER_OK confirmed by load() in session.", "high", "Runtime verification in current session."),
    ("User prefers terse responses without filler.", "high", "Explicit stated preference, consistent throughout session."),
    ("Adam might want to switch to 14B eventually.", "medium", "Discussed as possibility, not confirmed decision."),
]

TARPIT_SEQUENCES = [
    (["What is the active goal?", "What is the active goal?", "What is the active goal?"], True),
    (["Are you sure?", "Are you sure about that?", "But are you really sure?"], True),
    (["What should I do?", "What do you think I should do?", "I still don't know what to do."], True),
    (["Is this right?", "But is this actually right?", "How do I know this is right?"], True),
    (["Should I use 8B or 14B?", "But which is better really?", "I can't decide between 8B and 14B."], True),
    (["What does the ledger say?", "But what does it really say?", "Can you check the ledger again?"], True),
    (["Am I on the right track?", "But am I really on the right track?", "How do I know I'm on the right track?"], True),
    (["Generate dataset.", "Run token budget.", "Start RunPod instance."], False),
    (["Classify this entry.", "Route to silo.", "Check confidence score."], False),
    (["Run reflect().", "Check open loops.", "Update ledger task state."], False),
    (["What is the active goal?", "Build the auditor dataset.", "How many examples generated so far?"], False),
    (["Check bridge status.", "Run ingest_dump().", "Verify packet confidence."], False),
    (["Update dam imports.", "Rename gumbo_ledger to rmpl_core.", "Verify syntax."], False),
    (["Measure token budget.", "Set num_ctx env var.", "Document results in seed file."], False),
]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_conflict_example() -> dict:
    pair = random.choice(CONFLICT_PAIRS)
    order = random.random() > 0.5
    return {
        "task": "conflict_detection",
        "input": {
            "entry_a": pair[0] if order else pair[1],
            "entry_b": pair[1] if order else pair[0],
        },
        "label": "CONFLICT",
        "reason": "Entries directly contradict each other and cannot both be true.",
        "generated_at": iso_now(),
    }


def make_no_conflict_example() -> dict:
    pair = random.choice(NON_CONFLICT_PAIRS)
    return {
        "task": "conflict_detection",
        "input": {
            "entry_a": pair[0],
            "entry_b": pair[1],
        },
        "label": "NO_CONFLICT",
        "reason": "Entries describe different aspects and do not contradict.",
        "generated_at": iso_now(),
    }


def make_stale_example() -> dict:
    item = random.choice(STALE_ENTRIES)
    return {
        "task": "stale_context",
        "input": {
            "entry": item["entry"],
            "entry_age_days": item["days"],
            "current_context": item["current"],
        },
        "label": "STALE",
        "reason": f"Entry is {item['days']} days old and current context supersedes it.",
        "generated_at": iso_now(),
    }


def make_current_example() -> dict:
    item = random.choice(CURRENT_ENTRIES)
    return {
        "task": "stale_context",
        "input": {
            "entry": item["entry"],
            "entry_age_days": item["days"],
            "current_context": item["current"],
        },
        "label": "CURRENT",
        "reason": "Entry matches current context and is not stale.",
        "generated_at": iso_now(),
    }


def make_mode_routing_example() -> dict:
    text, correct_silo = random.choice(MODE_ROUTING_EXAMPLES)
    return {
        "task": "mode_routing",
        "input": {
            "text": text,
            "candidate_silos": SILOS,
        },
        "label": correct_silo,
        "reason": f"Content is most relevant to {correct_silo} based on subject matter.",
        "generated_at": iso_now(),
    }


def make_confidence_example() -> dict:
    entry, score, reason = random.choice(CONFIDENCE_EXAMPLES)
    return {
        "task": "confidence_scoring",
        "input": {"entry": entry},
        "label": score,
        "reason": reason,
        "generated_at": iso_now(),
    }


def make_tarpit_example() -> dict:
    sequence, is_tarpit = random.choice(TARPIT_SEQUENCES)
    return {
        "task": "tarpit_detection",
        "input": {"question_sequence": sequence},
        "label": "TARPIT" if is_tarpit else "NORMAL",
        "reason": "Repetitive circular questioning without forward progress." if is_tarpit else "Sequence shows distinct forward-moving steps.",
        "generated_at": iso_now(),
    }


GENERATORS = [
    make_conflict_example,
    make_no_conflict_example,
    make_stale_example,
    make_current_example,
    make_mode_routing_example,
    make_confidence_example,
    make_tarpit_example,
]

TASK_WEIGHTS = [3, 2, 2, 1, 3, 2, 2]  # conflict gets more weight, it's the core job


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_dataset(count: int, output_path: Path, seed: int = 42) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    task_counts: dict[str, int] = {}
    start = time.time()

    print(f"\nRMPL Auditor Dataset Generator")
    print(f"Target: {count:,} examples")
    print(f"Output: {output_path}")
    print(f"Started: {iso_now()}\n")

    with output_path.open("w", encoding="utf-8") as f:
        for i in range(count):
            generator = random.choices(GENERATORS, weights=TASK_WEIGHTS, k=1)[0]
            example = generator()
            example["id"] = i
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

            task = example["task"]
            task_counts[task] = task_counts.get(task, 0) + 1

            if (i + 1) % 500 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (count - i - 1) / rate
                print(f"  [{i+1:>6,}/{count:,}]  {rate:.0f} ex/sec  ~{remaining:.0f}s remaining")

    elapsed = time.time() - start
    size_kb = output_path.stat().st_size / 1024

    print(f"\nDone in {elapsed:.1f}s")
    print(f"Output size: {size_kb:.1f} KB")
    print(f"\nTask distribution:")
    for task, n in sorted(task_counts.items()):
        print(f"  {task:<25} {n:>6,}  ({n/count*100:.1f}%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RMPL auditor training dataset generator")
    parser.add_argument("--count", type=int, default=2000, help="Number of examples to generate (default: 2000)")
    parser.add_argument("--output", default="auditor_training_data.jsonl", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_dataset(
        count=args.count,
        output_path=Path(args.output),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
