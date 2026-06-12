"""
rmpl_auditor_dataset_gen.py

Generates synthetic training examples for the RMPL auditor model.
Runs unattended on CPU. No GPU required.

Design:
    - No personal data. All examples are generic and domain-agnostic.
    - Silo routing is NOT an auditor task. Routing is handled by deterministic
      Python keyword matching. The auditor only performs binary classification.
    - Auditor tasks are all binary (yes/no decisions):
        1. conflict_detection  - do two entries contradict each other?
        2. stale_context       - is this entry outdated given current context?
        3. tarpit_detection    - is this a circular question loop?
        4. action_gate         - does this action require confirmation before execution?

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


CONFLICT_PAIRS = [
    ("The ledger was updated today.", "The ledger has not been updated in 30 days."),
    ("The active goal is task A.", "The active goal is task B."),
    ("The model context window is 8192 tokens.", "The model context window is 131072 tokens."),
    ("The system is running model X.", "The system is running model Y."),
    ("The reservoir gate is open.", "The reservoir gate requires explicit authorization."),
    ("The governance layer is active.", "The governance layer is disabled for this session."),
    ("The user confirmed this action.", "No confirmation has been received for this action."),
    ("The ledger status is OK.", "The ledger status is CORRUPTED."),
    ("The last reflection ran 2 minutes ago.", "The last reflection ran 18 days ago."),
    ("Token budget used: 5% worst case.", "Token budget used: 67% at baseline."),
    ("The runtime has 6 core files.", "The runtime has 50 files."),
    ("The auditor model target is 3B.", "The auditor model target is 0.5B."),
    ("Active mode: evidence collection.", "Active mode: general continuity."),
    ("Confidence score: 0.92.", "Confidence score: 0.21."),
    ("The bridge endpoint is online.", "The bridge is offline."),
    ("Action was logged as approved.", "Action was logged as rejected."),
    ("Cold storage access was authorized.", "Cold storage access was denied."),
    ("The current task is complete.", "The current task is still open."),
    ("Reflection confidence is high.", "Reflection confidence is low."),
    ("The runtime is in a clean state.", "The runtime has unresolved conflicts."),
]

NON_CONFLICT_PAIRS = [
    ("The ledger was updated today.", "The active goal is task A."),
    ("Token budget used: 5%.", "The bridge endpoint is online."),
    ("The governance layer is active.", "The last reflection ran 2 minutes ago."),
    ("The auditor model target is 3B.", "Cold storage access was authorized."),
    ("Confidence score: 0.92.", "The runtime has 6 core files."),
    ("Active mode: evidence collection.", "The ledger status is OK."),
    ("The reservoir gate requires authorization.", "Reflection confidence is high."),
    ("Action was logged as approved.", "The model context window is 131072 tokens."),
    ("The current task is complete.", "Token budget used: 5%."),
    ("The system is running model X.", "The ledger was updated today."),
]

STALE_ENTRIES = [
    {"entry": "The active goal is task A.", "days": 21, "current": "The active goal has changed to task B."},
    {"entry": "The runtime has 50 files.", "days": 14, "current": "The runtime was cleaned to 6 core files."},
    {"entry": "Module X is the primary dependency.", "days": 10, "current": "Module X was replaced by module Y."},
    {"entry": "Token budget estimate: 45% at baseline.", "days": 7, "current": "Token budget measured at 5% baseline by the budget tool."},
    {"entry": "The auditor model size is undecided.", "days": 5, "current": "Auditor model target set at 3B minimum."},
    {"entry": "The bridge imports from legacy_module.", "days": 4, "current": "The bridge was updated to import from the new module."},
    {"entry": "The endpoint path is hardcoded.", "days": 3, "current": "The endpoint path now reads from environment variable."},
    {"entry": "Cold storage access is unrestricted.", "days": 8, "current": "Cold storage now requires explicit authorization."},
    {"entry": "The ledger schema is version 0.9.", "days": 12, "current": "The ledger schema was upgraded to version 1.0."},
    {"entry": "Reflection runs every 60 seconds.", "days": 6, "current": "Reflection is now event-triggered, not time-based."},
]

CURRENT_ENTRIES = [
    {"entry": "The runtime has 6 core files.", "days": 0, "current": "The runtime has 6 core files."},
    {"entry": "Auditor model target: 3B minimum.", "days": 1, "current": "Auditor model target: 3B minimum."},
    {"entry": "Token budget: 5% worst case.", "days": 0, "current": "Token budget: 5% worst case."},
    {"entry": "The bridge imports from the new module.", "days": 1, "current": "The bridge imports from the new module."},
    {"entry": "Cold storage requires explicit authorization.", "days": 1, "current": "Cold storage requires explicit authorization."},
    {"entry": "Ledger status is OK.", "days": 0, "current": "Ledger status is OK."},
    {"entry": "Reflection confidence: 0.87.", "days": 0, "current": "Reflection confidence: 0.87."},
]

TARPIT_SEQUENCES = [
    (["What is the active goal?", "What is the active goal?", "What is the active goal?"], True),
    (["Are you sure?", "Are you sure about that?", "But are you really sure?"], True),
    (["What should I do next?", "What do you think I should do?", "I still don't know what to do."], True),
    (["Is this the right approach?", "But is this actually right?", "How do I know this is right?"], True),
    (["Should I use option A or B?", "But which is really better?", "I can't decide between A and B."], True),
    (["What does the ledger say?", "But what does it really say?", "Can you check the ledger again?"], True),
    (["Am I on the right track?", "But am I really on the right track?", "How do I know I'm on track?"], True),
    (["Which model should I use?", "But really which one?", "I keep going back and forth on the model."], True),
    (["Step 1: generate dataset.", "Step 2: run evaluation.", "Step 3: review results."], False),
    (["Classify this entry.", "Route to storage.", "Check confidence score."], False),
    (["Run reflection.", "Check open loops.", "Update task state."], False),
    (["What is the active goal?", "Build the auditor.", "How many examples so far?"], False),
    (["Check bridge status.", "Run ingestion.", "Verify packet confidence."], False),
    (["Update imports.", "Rename module.", "Verify syntax."], False),
    (["Measure token budget.", "Set context env var.", "Document results."], False),
    (["Load ledger.", "Check for conflicts.", "Log verified fact."], False),
]

ACTION_GATE_EXAMPLES = [
    ("Delete all files in the archive folder.", True),
    ("Submit the complaint to the external authority.", True),
    ("Overwrite the ledger with a new version.", True),
    ("Send the report to the client.", True),
    ("Reset the runtime state to defaults.", True),
    ("Purge all entries older than 30 days.", True),
    ("Deploy the updated bridge to production.", True),
    ("Merge the experimental branch into main.", True),
    ("Summarize the current ledger state.", False),
    ("List all open loops in the ledger.", False),
    ("Analyze the reflection log for patterns.", False),
    ("Draft a checklist for the next task.", False),
    ("Compare the two reflection summaries.", False),
    ("Inspect the active context packet.", False),
    ("Classify the incoming request.", False),
    ("Review the audit log entries.", False),
]


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_conflict_example() -> dict:
    pair = random.choice(CONFLICT_PAIRS)
    order = random.random() > 0.5
    return {"task": "conflict_detection", "input": {"entry_a": pair[0] if order else pair[1], "entry_b": pair[1] if order else pair[0]}, "label": "CONFLICT", "reason": "Entries directly contradict each other.", "generated_at": iso_now()}


def make_no_conflict_example() -> dict:
    pair = random.choice(NON_CONFLICT_PAIRS)
    return {"task": "conflict_detection", "input": {"entry_a": pair[0], "entry_b": pair[1]}, "label": "NO_CONFLICT", "reason": "Entries describe different aspects and do not contradict.", "generated_at": iso_now()}


def make_stale_example() -> dict:
    item = random.choice(STALE_ENTRIES)
    return {"task": "stale_context", "input": {"entry": item["entry"], "entry_age_days": item["days"], "current_context": item["current"]}, "label": "STALE", "reason": f"Entry is {item['days']} days old and superseded.", "generated_at": iso_now()}


def make_current_example() -> dict:
    item = random.choice(CURRENT_ENTRIES)
    return {"task": "stale_context", "input": {"entry": item["entry"], "entry_age_days": item["days"], "current_context": item["current"]}, "label": "CURRENT", "reason": "Entry matches current context.", "generated_at": iso_now()}


def make_tarpit_example() -> dict:
    sequence, is_tarpit = random.choice(TARPIT_SEQUENCES)
    return {"task": "tarpit_detection", "input": {"question_sequence": sequence}, "label": "TARPIT" if is_tarpit else "NORMAL", "reason": "Circular." if is_tarpit else "Forward progress.", "generated_at": iso_now()}


def make_action_gate_example() -> dict:
    action, requires = random.choice(ACTION_GATE_EXAMPLES)
    return {"task": "action_gate", "input": {"action": action}, "label": "BLOCK" if requires else "ALLOW", "reason": "Destructive/external action." if requires else "Safe read-only action.", "generated_at": iso_now()}


GENERATORS = [make_conflict_example, make_no_conflict_example, make_stale_example, make_current_example, make_tarpit_example, make_action_gate_example]
TASK_WEIGHTS = [3, 2, 2, 1, 2, 2]


def generate_dataset(count: int, output_path: Path, seed: int = 42) -> None:
    random.seed(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_counts: dict[str, int] = {}
    start = time.time()
    print(f"\nRMPL Auditor Dataset Generator\nTarget:  {count:,} examples\nOutput:  {output_path}\nStarted: {iso_now()}\n")
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
                print(f"  [{i+1:>6,}/{count:,}]  {rate:.0f} ex/sec  ~{(count-i-1)/rate:.0f}s remaining")
    elapsed = time.time() - start
    print(f"\nDone in {elapsed:.1f}s\nOutput size: {output_path.stat().st_size/1024:.1f} KB\n\nTask distribution:")
    for task, n in sorted(task_counts.items()):
        print(f"  {task:<25} {n:>6,}  ({n/count*100:.1f}%)")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="RMPL auditor training dataset generator")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--output", default="auditor_training_data.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_dataset(count=args.count, output_path=Path(args.output), seed=args.seed)


if __name__ == "__main__":
    main()
