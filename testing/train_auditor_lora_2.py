"""
train_auditor_lora.py
LoRA fine-tune Qwen2.5-3B-Instruct on the RMPL auditor dataset.
Uses TRL + PEFT directly. No Unsloth.

Usage:
    python train_auditor_lora.py
    python train_auditor_lora.py --merge
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

DEFAULT_MODEL_PATH = r"C:\Users\Adam\RMPL\models\qwen2.5-3b"
DEFAULT_TRAIN_DATA = r"C:\Users\Adam\RMPL\testing\auditor_lora_train.jsonl"
DEFAULT_VAL_DATA   = r"C:\Users\Adam\RMPL\testing\auditor_lora_val.jsonl"
DEFAULT_OUTPUT     = r"C:\Users\Adam\RMPL\models\auditor_lora_out"

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def fmt(example, tokenizer):
    text = tokenizer.apply_chat_template(
        example["conversations"], tokenize=False, add_generation_prompt=False)
    return {"text": text}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL_PATH)
    parser.add_argument("--train-data", default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data",   default=DEFAULT_VAL_DATA)
    parser.add_argument("--output",     default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--batch",      type=int,   default=2)
    parser.add_argument("--grad-accum", type=int,   default=4)
    parser.add_argument("--lr",         type=float, default=2e-4)
    parser.add_argument("--max-seq",    type=int,   default=1024)
    parser.add_argument("--lora-r",     type=int,   default=16)
    parser.add_argument("--lora-alpha", type=int,   default=32)
    parser.add_argument("--merge",      action="store_true")
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("ERROR: No GPU detected.")
        return
    print(f"GPU:  {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB\n")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print(f"Loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model in bf16...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.enable_input_require_grads()

    print("Attaching LoRA adapter...")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nLoading datasets...")
    train_raw = load_jsonl(args.train_data)
    val_raw   = load_jsonl(args.val_data)
    print(f"  Train: {len(train_raw)}")
    print(f"  Val:   {len(val_raw)}")

    train_ds = Dataset.from_list(train_raw).map(lambda ex: fmt(ex, tokenizer), remove_columns=["conversations"])
    val_ds   = Dataset.from_list(val_raw).map(lambda ex: fmt(ex, tokenizer),   remove_columns=["conversations"])

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\nStarting training...\n")
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
            bf16=True,
            fp16=False,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            report_to="none",
            dataset_text_field="text",
            max_seq_length=args.max_seq,
            packing=True,
        ),
    )

    stats = trainer.train()

    adapter_path = output_path / "adapter"
    print(f"\nSaving adapter to {adapter_path}...")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    if args.merge:
        from peft import PeftModel
        merged_path = output_path / "merged"
        print(f"Merging into {merged_path}...")
        base = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
        merged = PeftModel.from_pretrained(base, str(adapter_path))
        merged = merged.merge_and_unload()
        merged.save_pretrained(str(merged_path))
        tokenizer.save_pretrained(str(merged_path))
        print(f"Merged model saved. Convert to GGUF with:")
        print(f"  python llama.cpp/convert_hf_to_gguf.py {merged_path} --outtype q4_k_m")

    print(f"\n{'='*50}")
    print(f"  TRAINING COMPLETE")
    print(f"  Runtime: {stats.metrics.get('train_runtime', 0):.0f}s")
    print(f"  Loss:    {stats.metrics.get('train_loss', 0):.4f}")
    print(f"  Adapter: {adapter_path}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
