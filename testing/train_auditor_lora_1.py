"""
train_auditor_lora.py

LoRA fine-tune Qwen2.5-3B-Instruct on the RMPL auditor dataset.
Targets conflict_detection and action_gate tasks specifically.

Requirements:
    pip install unsloth --break-system-packages
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
    (or cuda equivalent if using nvidia)

Usage:
    python train_auditor_lora.py
    python train_auditor_lora.py --epochs 3 --output ./auditor_lora_out

After training:
    - Adapter weights saved to --output directory
    - Run eval with: python rmpl_auditor_eval.py --model <merged_model_path>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PATH = r"C:\Users\Adam\RMPL\models\qwen2.5-3b"
DEFAULT_TRAIN_DATA = r"C:\Users\Adam\RMPL\testing\auditor_lora_train.jsonl"
DEFAULT_VAL_DATA   = r"C:\Users\Adam\RMPL\testing\auditor_lora_val.jsonl"
DEFAULT_OUTPUT     = r"C:\Users\Adam\RMPL\models\auditor_lora_out"
DEFAULT_EPOCHS     = 3
DEFAULT_BATCH      = 2       # safe for 16GB VRAM
DEFAULT_GRAD_ACCUM = 4       # effective batch = 8
DEFAULT_LR         = 2e-4
DEFAULT_MAX_SEQ    = 1024    # auditor prompts are short
DEFAULT_LORA_R     = 16
DEFAULT_LORA_ALPHA = 32


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def conversations_to_text(example: dict, tokenizer) -> dict:
    """
    Convert ShareGPT conversation format to a single text string
    using the model's chat template.
    """
    conversations = example["conversations"]
    text = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train RMPL auditor LoRA")
    parser.add_argument("--model",      default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data",   default=DEFAULT_VAL_DATA)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",      type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--grad-accum", type=int,   default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--max-seq",    type=int,   default=DEFAULT_MAX_SEQ)
    parser.add_argument("--lora-r",     type=int,   default=DEFAULT_LORA_R)
    parser.add_argument("--lora-alpha", type=int,   default=DEFAULT_LORA_ALPHA)
    parser.add_argument("--merge",      action="store_true",
                        help="Merge adapter into base model after training")
    args = parser.parse_args()

    # Lazy imports so the script fails loudly if unsloth isn't installed
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ImportError:
        print("ERROR: unsloth not installed.")
        print("  pip install unsloth")
        return

    try:
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
    except ImportError:
        print("ERROR: datasets or trl not installed.")
        print("  pip install datasets trl")
        return

    print(f"\nRMPL Auditor LoRA Training")
    print(f"Model:      {args.model}")
    print(f"Train data: {args.train_data}")
    print(f"Val data:   {args.val_data}")
    print(f"Output:     {args.output}")
    print(f"Epochs:     {args.epochs}")
    print(f"LoRA r/α:   {args.lora_r}/{args.lora_alpha}\n")

    # ------------------------------------------------------------------
    # 1. Load model + tokenizer via Unsloth
    # ------------------------------------------------------------------
    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq,
        dtype=None,           # auto-detect: bf16 on ROCm/RDNA4
        load_in_4bit=False,   # full precision LoRA, 16GB is enough for 3B
    )

    # Apply Qwen2.5 chat template
    tokenizer = get_chat_template(tokenizer, chat_template="qwen-2.5")

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapter
    # ------------------------------------------------------------------
    print("Attaching LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # ------------------------------------------------------------------
    # 3. Load and format datasets
    # ------------------------------------------------------------------
    print("Loading datasets...")
    train_raw = load_jsonl(args.train_data)
    val_raw   = load_jsonl(args.val_data)

    print(f"  Train: {len(train_raw)} examples")
    print(f"  Val:   {len(val_raw)} examples")

    train_ds = Dataset.from_list(train_raw)
    val_ds   = Dataset.from_list(val_raw)

    # Apply chat template to convert conversations -> text
    train_ds = train_ds.map(
        lambda ex: conversations_to_text(ex, tokenizer),
        remove_columns=train_ds.column_names,
    )
    val_ds = val_ds.map(
        lambda ex: conversations_to_text(ex, tokenizer),
        remove_columns=val_ds.column_names,
    )

    # ------------------------------------------------------------------
    # 4. Trainer
    # ------------------------------------------------------------------
    print("\nStarting training...\n")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=SFTConfig(
            output_dir=str(output_path),
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            bf16=True,           # RDNA4 supports bf16
            fp16=False,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",    # no wandb
            dataset_text_field="text",
            max_seq_length=args.max_seq,
            packing=True,        # pack short sequences for efficiency
        ),
    )

    trainer_stats = trainer.train()

    # ------------------------------------------------------------------
    # 5. Save adapter
    # ------------------------------------------------------------------
    adapter_path = output_path / "adapter"
    print(f"\nSaving adapter to {adapter_path}...")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    # ------------------------------------------------------------------
    # 6. Optionally merge and save full model
    # ------------------------------------------------------------------
    if args.merge:
        merged_path = output_path / "merged"
        print(f"Merging adapter into base model -> {merged_path}...")
        model.save_pretrained_merged(
            str(merged_path),
            tokenizer,
            save_method="merged_16bit",
        )
        print(f"Merged model saved to {merged_path}")
        print("\nTo convert to GGUF for Ollama:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_path} --outtype q4_k_m")

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*55}")
    print(f"  Train runtime:  {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"  Train loss:     {trainer_stats.metrics.get('train_loss', 0):.4f}")
    print(f"  Adapter saved:  {adapter_path}")
    if args.merge:
        print(f"  Merged saved:   {merged_path}")
    print(f"{'='*55}")
    print("\nNext: run the auditor eval against the fine-tuned model to get before/after delta.")


if __name__ == "__main__":
    main()
