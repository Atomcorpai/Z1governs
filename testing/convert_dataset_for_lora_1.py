"""
convert_dataset_for_lora.py

Converts auditor_training_data.jsonl to Unsloth instruction-tuning format.

Input format (one JSON object per line):
  {"task": "conflict_detection", "input": {...}, "label": "CONFLICT"}
  {"task": "action_gate", "input": {"action": "..."}, "label": "BLOCK", "rule_id": "R1"}
  etc.

Output format (ShareGPT / Unsloth chat template):
  {"conversations": [
      {"role": "system", "content": "<system prompt>"},
      {"role": "user",   "content": "<task-specific prompt>"},
      {"role": "assistant", "content": "<label as JSON>"}
  ]}

Usage:
  python convert_dataset_for_lora.py \
      --input  auditor_training_data.jsonl \
      --output auditor_lora_train.jsonl \
      --val-split 0.1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# System prompt (matches governance_seed / runtime_beacon)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a runtime auditor. Your only job is binary classification. "
    "Respond with valid JSON only. No prose before or after. "
    "If you cannot comply: {\"ok\": false, \"error\": \"schema\"}. "
    "Uncertainty defaults to the safer verdict (BLOCK, CONFLICT, TARPIT, STALE)."
)


# ---------------------------------------------------------------------------
# Per-task prompt builders
# ---------------------------------------------------------------------------

def _prompt_conflict_detection(inp: Dict[str, Any]) -> str:
    entry     = inp.get("entry", "")
    context   = inp.get("context", "")
    age_days  = inp.get("entry_age_days", "unknown")
    return (
        "Task: conflict_detection\n"
        f"Entry: {entry}\n"
        f"Current context: {context}\n"
        f"Entry age (days): {age_days}\n\n"
        "Does this entry conflict with the current context?\n"
        "Respond with exactly:\n"
        "{\"task\": \"conflict_detection\", \"verdict\": \"CONFLICT\" | \"NO_CONFLICT\", "
        "\"confidence\": 0.0-1.0, \"evidence\": [\"<= 3 short quotes\"]}"
    )


def _prompt_stale_context(inp: Dict[str, Any]) -> str:
    entry    = inp.get("entry", "")
    age_days = inp.get("entry_age_days", "unknown")
    return (
        "Task: stale_context\n"
        f"Entry: {entry}\n"
        f"Entry age (days): {age_days}\n\n"
        "Is this entry stale?\n"
        "Respond with exactly:\n"
        "{\"task\": \"stale_context\", \"verdict\": \"STALE\" | \"CURRENT\", "
        "\"confidence\": 0.0-1.0, \"evidence\": [\"<= 3 short quotes\"]}"
    )


def _prompt_action_gate(inp: Dict[str, Any]) -> str:
    action = inp.get("action", "")
    return (
        "Task: action_gate\n"
        f"Action: {action}\n\n"
        "Does this action require confirmation before execution?\n"
        "BLOCK: destructive, irreversible, external, or ambiguous.\n"
        "ALLOW: safe, read-only, reversible, unambiguous.\n"
        "If uncertain: BLOCK with confidence <= 0.49 and rule_id R4.\n"
        "Respond with exactly:\n"
        "{\"task\": \"action_gate\", \"verdict\": \"BLOCK\" | \"ALLOW\", "
        "\"rule_id\": \"R0\"|\"R1\"|\"R2\"|\"R3\"|\"R4\", "
        "\"confidence\": 0.0-1.0, \"rationale\": \"<= 240 chars\", "
        "\"evidence\": [\"<= 3 short quotes\"]}"
    )


def _prompt_tarpit_detection(inp: Dict[str, Any]) -> str:
    sequence = inp.get("question_sequence", [])
    seq_str  = " -> ".join(sequence) if sequence else str(inp.get("sequence", ""))
    return (
        "Task: tarpit_detection\n"
        f"Question sequence: {seq_str}\n\n"
        "Is this a circular question loop (tarpit)?\n"
        "Respond with exactly:\n"
        "{\"task\": \"tarpit_detection\", \"verdict\": \"TARPIT\" | \"NORMAL\", "
        "\"confidence\": 0.0-1.0, \"evidence\": [\"<= 3 short quotes\"]}"
    )


PROMPT_BUILDERS = {
    "conflict_detection": _prompt_conflict_detection,
    "stale_context":      _prompt_stale_context,
    "action_gate":        _prompt_action_gate,
    "tarpit_detection":   _prompt_tarpit_detection,
}


# ---------------------------------------------------------------------------
# Label -> assistant response builder
# ---------------------------------------------------------------------------

def _build_assistant_response(task: str, row: Dict[str, Any]) -> Optional[str]:
    label = row.get("label", "")
    inp   = row.get("input", {})

    if task == "conflict_detection":
        verdict = label  # CONFLICT | NO_CONFLICT
        return json.dumps({
            "task": "conflict_detection",
            "verdict": verdict,
            "confidence": 0.9 if verdict == "CONFLICT" else 0.85,
            "evidence": [str(inp.get("entry", ""))[:60]]
        })

    if task == "stale_context":
        verdict = label  # STALE | CURRENT
        return json.dumps({
            "task": "stale_context",
            "verdict": verdict,
            "confidence": 0.9 if verdict == "STALE" else 0.85,
            "evidence": [f"entry_age_days={inp.get('entry_age_days', '?')}"]
        })

    if task == "action_gate":
        verdict  = label  # BLOCK | ALLOW
        rule_id  = row.get("rule_id", "R4")
        action   = str(inp.get("action", ""))
        return json.dumps({
            "task": "action_gate",
            "verdict": verdict,
            "rule_id": rule_id,
            "confidence": 0.97 if verdict == "BLOCK" else 0.85,
            "rationale": f"Action '{action[:60]}' classified as {verdict}.",
            "evidence": [action[:60]]
        })

    if task == "tarpit_detection":
        verdict = label  # TARPIT | NORMAL
        seq = inp.get("question_sequence", [])
        evidence = [seq[0][:60]] if seq else []
        return json.dumps({
            "task": "tarpit_detection",
            "verdict": verdict,
            "confidence": 0.88 if verdict == "TARPIT" else 0.82,
            "evidence": evidence
        })

    return None


# ---------------------------------------------------------------------------
# Row converter
# ---------------------------------------------------------------------------

def convert_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task = row.get("task", "")
    if task not in PROMPT_BUILDERS:
        return None

    inp = row.get("input", {})
    user_content = PROMPT_BUILDERS[task](inp)
    assistant_content = _build_assistant_response(task, row)

    if not assistant_content:
        return None

    return {
        "conversations": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Convert RMPL auditor dataset to Unsloth format")
    parser.add_argument("--input",     default="auditor_training_data.jsonl")
    parser.add_argument("--output",    default="auditor_lora_train.jsonl")
    parser.add_argument("--val-output",default="auditor_lora_val.jsonl")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction of data to use for validation (default 0.1)")
    parser.add_argument("--seed",      type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path  = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        return

    rows = []
    skipped = 0
    with input_path.open("r", encoding="utf-8") as f:
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
                skipped += 1

    random.shuffle(rows)
    val_size   = max(1, int(len(rows) * args.val_split))
    val_rows   = rows[:val_size]
    train_rows = rows[val_size:]

    def write_jsonl(path: str, data: list) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(args.output,     train_rows)
    write_jsonl(args.val_output, val_rows)

    task_counts: Dict[str, int] = {}
    for r in rows:
        task = r["conversations"][1]["content"].split("\n")[0].replace("Task: ", "")
        task_counts[task] = task_counts.get(task, 0) + 1

    print(f"Input:      {input_path} ({len(rows) + skipped} lines)")
    print(f"Converted:  {len(rows)}")
    print(f"Skipped:    {skipped}")
    print(f"Train:      {len(train_rows)} -> {args.output}")
    print(f"Validation: {len(val_rows)}  -> {args.val_output}")
    print(f"\nTask distribution:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task:<25} {count}")


if __name__ == "__main__":
    main()
