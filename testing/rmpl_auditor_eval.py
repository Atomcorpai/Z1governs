"""
rmpl_auditor_eval.py

Evaluates a local Ollama model as an RMPL auditor classifier.
Feed it the dataset from rmpl_auditor_dataset_gen.py and let it run overnight.

Output:
    auditor_eval_results.jsonl  - per-example results
    auditor_eval_summary.json   - accuracy by task, overall score

Usage:
    python rmpl_auditor_eval.py
    python rmpl_auditor_eval.py --model llama3.2:3b --dataset auditor_training_data.jsonl
    python rmpl_auditor_eval.py --limit 500  # quick test run
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "llama3.2:3b"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_DATASET = "auditor_training_data.jsonl"
DEFAULT_RESULTS = "auditor_eval_results.jsonl"
DEFAULT_SUMMARY = "auditor_eval_summary.json"


# ---------------------------------------------------------------------------
# Prompt templates per task
# ---------------------------------------------------------------------------

def build_prompt(example: dict) -> tuple[str, list[str]]:
    """
    Returns (prompt_text, valid_labels).
    Prompt instructs the model to respond with ONLY the label, nothing else.
    """
    task = example["task"]
    inp = example["input"]

    if task == "conflict_detection":
        valid = ["CONFLICT", "NO_CONFLICT"]
        prompt = (
            "You are a memory auditor. Your only job is to classify whether two entries conflict.\n"
            "Respond with exactly one word: CONFLICT or NO_CONFLICT. No explanation.\n\n"
            f"Entry A: {inp['entry_a']}\n"
            f"Entry B: {inp['entry_b']}\n\n"
            "Classification:"
        )

    elif task == "stale_context":
        valid = ["STALE", "CURRENT"]
        prompt = (
            "You are a memory auditor. Your only job is to classify whether a memory entry is stale.\n"
            "Respond with exactly one word: STALE or CURRENT. No explanation.\n\n"
            f"Entry: {inp['entry']}\n"
            f"Entry age (days): {inp['entry_age_days']}\n"
            f"Current context: {inp['current_context']}\n\n"
            "Classification:"
        )

    elif task == "mode_routing":
        valid = ["core_runtime", "work_product", "technical_builds", "life_admin", "other"]
        silo_list = ", ".join(inp["candidate_silos"])
        prompt = (
            "You are a memory auditor. Your only job is to route content to the correct silo.\n"
            f"Available silos: {silo_list}\n"
            "Respond with exactly one silo name from the list above. No explanation.\n\n"
            f"Content: {inp['text']}\n\n"
            "Silo:"
        )

    elif task == "confidence_scoring":
        valid = ["high", "medium", "low"]
        prompt = (
            "You are a memory auditor. Your only job is to score the reliability of a memory entry.\n"
            "Respond with exactly one word: high, medium, or low. No explanation.\n\n"
            f"Entry: {inp['entry']}\n\n"
            "Confidence:"
        )

    elif task == "tarpit_detection":
        valid = ["TARPIT", "NORMAL"]
        sequence = " -> ".join(inp["question_sequence"])
        prompt = (
            "You are a memory auditor. Your only job is to detect circular question loops.\n"
            "Respond with exactly one word: TARPIT or NORMAL. No explanation.\n\n"
            f"Question sequence: {sequence}\n\n"
            "Classification:"
        )

    else:
        valid = []
        prompt = f"Unknown task: {task}"

    return prompt, valid


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def call_ollama(prompt: str, model: str, url: str, timeout: int = 30) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,  # deterministic for eval
            "num_ctx": 2048,
        },
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "").strip()
    except requests.exceptions.ConnectionError:
        return "CONNECTION_ERROR"
    except requests.exceptions.Timeout:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {str(e)}"


def normalize_response(raw: str, valid_labels: list[str]) -> str:
    """Extract the first valid label from model response."""
    cleaned = raw.strip().split("\n")[0].strip().rstrip(".,;:")
    # Exact match first
    if cleaned in valid_labels:
        return cleaned
    # Case-insensitive
    lower = cleaned.lower()
    for label in valid_labels:
        if label.lower() == lower:
            return label
    # Partial match (model added explanation despite instructions)
    for label in valid_labels:
        if label.lower() in lower:
            return label
    return f"INVALID:{cleaned[:40]}"


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_eval(
    dataset_path: Path,
    results_path: Path,
    summary_path: Path,
    model: str,
    ollama_url: str,
    limit: int | None = None,
) -> None:
    examples = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if limit:
        examples = examples[:limit]

    total = len(examples)
    print(f"\nRMPL Auditor Eval")
    print(f"Model:   {model}")
    print(f"Dataset: {dataset_path} ({total:,} examples)")
    print(f"Output:  {results_path}")
    print(f"Started: {iso_now()}\n")

    # Check Ollama is up
    try:
        requests.get(ollama_url.replace("/api/generate", "/api/tags"), timeout=5)
    except Exception:
        print("ERROR: Cannot reach Ollama. Is it running?")
        print(f"  Expected at: {ollama_url}")
        return

    task_correct: dict[str, int] = {}
    task_total: dict[str, int] = {}
    task_invalid: dict[str, int] = {}
    start = time.time()

    with results_path.open("w", encoding="utf-8") as out:
        for i, example in enumerate(examples):
            task = example["task"]
            expected = example["label"]

            prompt, valid_labels = build_prompt(example)
            raw_response = call_ollama(prompt, model, ollama_url)
            predicted = normalize_response(raw_response, valid_labels)

            correct = predicted == expected
            invalid = predicted.startswith("INVALID:")

            task_total[task] = task_total.get(task, 0) + 1
            if correct:
                task_correct[task] = task_correct.get(task, 0) + 1
            if invalid:
                task_invalid[task] = task_invalid.get(task, 0) + 1

            result = {
                "id": example.get("id", i),
                "task": task,
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "invalid_response": invalid,
                "raw_response": raw_response[:200],
                "evaluated_at": iso_now(),
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                elapsed = time.time() - start
                rate = (i + 1) / elapsed
                remaining = (total - i - 1) / rate
                overall_acc = sum(task_correct.values()) / sum(task_total.values()) * 100
                print(f"  [{i+1:>5,}/{total:,}]  acc={overall_acc:.1f}%  {rate:.1f} ex/sec  ~{remaining:.0f}s left")

    # Summary
    elapsed = time.time() - start
    overall_correct = sum(task_correct.values())
    overall_total = sum(task_total.values())
    overall_acc = overall_correct / overall_total * 100 if overall_total else 0

    summary = {
        "model": model,
        "dataset": str(dataset_path),
        "total_examples": overall_total,
        "overall_accuracy": round(overall_acc, 2),
        "elapsed_seconds": round(elapsed, 1),
        "evaluated_at": iso_now(),
        "by_task": {},
    }

    print(f"\n{'='*55}")
    print(f"  RESULTS: {model}")
    print(f"{'='*55}")
    for task in sorted(task_total.keys()):
        n = task_total[task]
        c = task_correct.get(task, 0)
        inv = task_invalid.get(task, 0)
        acc = c / n * 100 if n else 0
        print(f"  {task:<25} {acc:>6.1f}%  ({c}/{n})  invalid={inv}")
        summary["by_task"][task] = {
            "total": n,
            "correct": c,
            "accuracy": round(acc, 2),
            "invalid_responses": inv,
        }

    print(f"{'='*55}")
    print(f"  OVERALL ACCURACY:         {overall_acc:.1f}%  ({overall_correct}/{overall_total})")
    print(f"  Total time:               {elapsed:.1f}s")
    print(f"{'='*55}\n")

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
    print(f"Results saved to: {results_path}\n")

    # Verdict
    if overall_acc >= 80:
        print("VERDICT: 3B auditor is viable. Take this to RunPod.")
    elif overall_acc >= 65:
        print("VERDICT: Marginal. Review invalid responses. May need prompt tuning or larger model.")
    else:
        print("VERDICT: Below threshold. Auditor needs prompt work or model upgrade before RunPod.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RMPL auditor eval against local Ollama model")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="Input JSONL dataset")
    parser.add_argument("--results", default=DEFAULT_RESULTS, help="Output results JSONL")
    parser.add_argument("--summary", default=DEFAULT_SUMMARY, help="Output summary JSON")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama API URL")
    parser.add_argument("--limit", type=int, default=None, help="Limit examples for quick test")
    args = parser.parse_args()

    run_eval(
        dataset_path=Path(args.dataset),
        results_path=Path(args.results),
        summary_path=Path(args.summary),
        model=args.model,
        ollama_url=args.ollama_url,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
