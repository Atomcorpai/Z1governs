"""
convert_dataset_for_lora.py

Converts auditor_training_data.jsonl to Unsloth/TRL instruction-tuning format.
All tasks use single-word output format matching the eval script exactly.

Usage:
    python convert_dataset_for_lora.py --input auditor_training_data.jsonl --output auditor_lora_train.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "You are a runtime auditor. Your only job is binary classification. "
    "Respond with exactly one word. No explanation, no punctuation, nothing else."
)

MODEL_TASKS = {"conflict_detection", "stale_context", "action_gate"}


def build_user_prompt(task: str, inp: dict) -> str | None:
    if task == "conflict_detection":
        entry_a = inp.get("entry_a", "")
        entry_b = inp.get("entry_b", "")
        if not entry_a or not entry_b:
            return None
        return (
            "You are a memory auditor. Classify whether these two entries conflict.\n"
            "Respond with exactly one word: CONFLICT or NO_CONFLICT. No explanation.\n\n"
            f"Entry A: {entry_a}\n"
            f"Entry B: {entry_b}\n\n"
            "Classification:"
        )

    if task == "stale_context":
        entry = inp.get("entry", "")
        age = inp.get("entry_age_days", "unknown")
        context = inp.get("current_context", "")
        if not entry:
            return None
        return (
            "You are a memory auditor. Classify whether this memory entry is stale.\n"
            "Respond with exactly one word: STALE or CURRENT. No explanation.\n\n"
            f"Entry: {entry}\n"
            f"Entry age (days): {age}\n"
            f"Current context: {context}\n\n"
            "Classification:"
        )

    if task == "action_gate":
        action = inp.get("action", "")
        if not action:
            return None
        return (
            "You are a runtime auditor. Classify whether this action requires confirmation before execution.\n"
            "Destructive, irreversible, or external actions require confirmation. Safe read-only actions do not.\n"
            "Respond with exactly one word: BLOCK or ALLOW. No explanation.\n\n"
            f"Action: {action}\n\n"
            "Classification:"
        )

    return None


def convert_row(row: dict) -> dict | None:
    task = row.get("task", "")
    if task not in MODEL_TASKS:
        return None

    inp = row.get("input", {})
    label = row.get("label", "")
    if not label:
        return None

    user_content = build_user_prompt(task, inp)
    if not user_content:
        return None

    return {
        "conversations": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": label},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="auditor_training_data.jsonl")
    parser.add_argument("--output",     default="auditor_lora_train.jsonl")
    parser.add_argument("--val-output", default="auditor_lora_val.jsonl")
    parser.add_argument("--val-split",  type=float, default=0.1)
    parser.add_argument("--seed",       type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rows = []
    skipped = 0
    empty = 0
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            converted = convert_row(raw)
            if converted:
                rows.append(converted)
            else:
                if raw.get("task") in MODEL_TASKS:
                    empty += 1
                else:
                    skipped += 1

    random.shuffle(rows)
    val_size   = max(1, int(len(rows) * args.val_split))
    val_rows   = rows[:val_size]
    train_rows = rows[val_size:]

    def write_jsonl(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(args.output, train_rows)
    write_jsonl(args.val_output, val_rows)

    task_counts: dict[str, int] = {}
    for r in rows:
        task = r["conversations"][1]["content"].split("\n")[0].replace("You are a memory auditor. ", "").replace("You are a runtime auditor. ", "")[:30]
        # simpler: extract from system or just count by label presence
    
    # Count by task from raw
    task_counts2: dict[str, int] = {}
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
                t = raw.get("task", "unknown")
                if t in MODEL_TASKS:
                    task_counts2[t] = task_counts2.get(t, 0) + 1
            except Exception:
                pass

    print(f"Input:      {args.input}")
    print(f"Converted:  {len(rows)}")
    print(f"Skipped:    {skipped} (wrong task)")
    print(f"Empty:      {empty} (missing required fields)")
    print(f"Train:      {len(train_rows)} -> {args.output}")
    print(f"Validation: {len(val_rows)}  -> {args.val_output}")
    print(f"\nTask distribution in source:")
    for task, count in sorted(task_counts2.items()):
        print(f"  {task:<25} {count}")


if __name__ == "__main__":
    main()
