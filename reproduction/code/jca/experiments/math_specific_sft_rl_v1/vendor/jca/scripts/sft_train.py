"""SFT training script for one JCA agent (Qwen3-{1.7B|4B|8B}) using TRL + LoRA.

Designed to be launched with `accelerate launch` for 8-GPU DDP.

Key design:
  - Loads per-agent JSONL produced by build_sft_data.py.
  - Tokenizes with the model's chat template; assistant-only loss mask.
  - LoRA adapter on q/k/v/o projections.
  - Detailed logging:
      * per-step train loss (every N steps, configurable)
      * eval loss + token-level accuracy on a held-out split (every M steps)
      * sample generations from a few fixed eval prompts (every M steps)
  - Anti-overfit:
      * Held-out eval split (10% of data)
      * Early stopping on eval loss (patience configurable)
      * Max 3 epochs; cosine LR with warmup
      * Weight decay 0.01

Usage (single agent):
    accelerate launch \\
        --num_processes 8 --multi_gpu \\
        jca/scripts/sft_train.py \\
        --agent A1 \\
        --data jca/sft_data/<tag>/A1.jsonl \\
        --out-dir jca/sft_runs/<tag>/A1
"""
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ============================================================================
# CLI
# ============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA SFT for one JCA agent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Identity
    p.add_argument("--agent", required=True, choices=["A1", "A2", "A3"],
                   help="Which agent to train.")
    p.add_argument("--data", type=Path, required=True,
                   help="Per-agent JSONL from build_sft_data.py")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output dir for checkpoints + logs")
    p.add_argument("--model-name-or-path", default=None,
                   help="HF model path; default = AGENT_CONFIGS[agent].base_model")
    p.add_argument("--init-lora-adapter", type=Path, default=None,
                   help="If set, load this existing LoRA adapter and continue training "
                        "from it instead of creating a new LoRA from scratch.")

    # Data split
    p.add_argument("--eval-fraction", type=float, default=0.1,
                   help="Fraction of data held out for eval / early stopping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--split-group-by",
        choices=["problem_id", "trajectory", "row"],
        default="problem_id",
        help="Keep all rows from a problem or trajectory in the same train/eval split.",
    )

    # LoRA
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # Training hyperparameters
    p.add_argument("--learning-rate", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--num-epochs", type=float, default=2.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If >0, override num-epochs.")
    p.add_argument("--per-device-train-batch-size", type=int, default=4)
    p.add_argument("--per-device-eval-batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation-steps", type=int, default=4)
    p.add_argument("--max-seq-length", type=int, default=4096)
    p.add_argument(
        "--max-tokenization-drop-ratio",
        type=float,
        default=0.05,
        help="Fail if truncation/empty-label filtering drops more than this fraction.",
    )
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--lr-scheduler-type", default="cosine")

    # Anti-overfit
    p.add_argument("--early-stopping-patience", type=int, default=3,
                   help="Stop if eval loss does not improve for this many evals.")
    p.add_argument("--early-stopping-threshold", type=float, default=0.0)
    p.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reload the best checkpoint after training. Disable to avoid end-of-run OOM.",
    )

    # Logging / eval cadence
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--eval-steps", type=int, default=100)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--report-to", default="tensorboard",
                   help="One of: tensorboard, wandb, none, all")

    # Misc
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--torch-compile", action="store_true", default=False)
    p.add_argument("--num-eval-samples-to-generate", type=int, default=4,
                   help="At each eval, generate this many sample completions.")
    p.add_argument("--dataloader-num-workers", type=int, default=0,
                   help="DataLoader workers. Keep 0 to avoid pymp temp-dir cleanup noise.")

    return p.parse_args()


# ============================================================================
# Logging setup
# ============================================================================


def setup_logging(out_dir: Path, agent: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"train_{agent}.log"

    logger = logging.getLogger("jca.sft")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if rerun in same process
    for h in list(logger.handlers):
        logger.removeHandler(h)

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.propagate = False
    is_main = os.environ.get("RANK", "0") == "0"
    if is_main:
        fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    else:
        logger.addHandler(logging.NullHandler())

    return logger


# ============================================================================
# Data loading & assistant-only label masking
# ============================================================================


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def split_examples_grouped(
    examples: List[Dict[str, Any]],
    *,
    eval_fraction: float,
    seed: int,
    group_by: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    if len(examples) < 2:
        raise ValueError("At least two examples are required for a train/eval split")

    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row_idx, example in enumerate(examples):
        if group_by == "row":
            key = f"row::{row_idx}"
        else:
            problem_id = str(example.get("problem_id") or "").strip()
            if not problem_id:
                raise ValueError(
                    f"Example {row_idx} is missing problem_id required by --split-group-by {group_by}"
                )
            if group_by == "problem_id":
                key = f"problem::{problem_id}"
            else:
                trajectory_id = example.get("trajectory_id")
                rollout_idx = example.get("rollout_idx")
                if trajectory_id is not None:
                    key = f"trajectory::{trajectory_id}"
                elif rollout_idx is not None:
                    key = f"trajectory::{problem_id}::{rollout_idx}"
                else:
                    key = f"trajectory::{problem_id}"
        grouped[key].append(example)

    if len(grouped) < 2:
        raise ValueError(
            f"Need at least two {group_by} groups for a leakage-free split; got {len(grouped)}"
        )

    group_keys = sorted(grouped)
    rng = random.Random(seed)
    rng.shuffle(group_keys)

    target_eval_rows = max(8, int(len(examples) * eval_fraction))
    target_eval_rows = min(target_eval_rows, len(examples) - 1)
    eval_keys: set[str] = set()
    eval_rows = 0
    for key in group_keys[:-1]:
        if eval_rows >= target_eval_rows:
            break
        eval_keys.add(key)
        eval_rows += len(grouped[key])

    eval_examples: list[Dict[str, Any]] = []
    train_examples: list[Dict[str, Any]] = []
    for key in group_keys:
        destination = eval_examples if key in eval_keys else train_examples
        destination.extend(grouped[key])

    rng.shuffle(train_examples)
    rng.shuffle(eval_examples)
    stats = {
        "train_rows": len(train_examples),
        "eval_rows": len(eval_examples),
        "train_groups": len(grouped) - len(eval_keys),
        "eval_groups": len(eval_keys),
    }
    return train_examples, eval_examples, stats


def encode_sft_example(
    example: Dict[str, Any],
    tokenizer,
    max_seq_length: int,
) -> Optional[Dict[str, Any]]:
    """Encode the inference prompt and supervised completion independently."""
    try:
        prompt_text = tokenizer.apply_chat_template(
            example["messages"],
            enable_thinking=False,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        prompt_text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=True,
        )

    label = str(example.get("label") or "")
    if not label:
        return None
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer must define eos_token for SFT completion encoding")

    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    completion_ids = tokenizer(
        label + tokenizer.eos_token,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    input_ids = prompt_ids + completion_ids
    if not completion_ids or len(input_ids) >= max_seq_length:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + completion_ids,
    }


def build_dataset(
    examples: List[Dict[str, Any]],
    tokenizer,
    max_seq_length: int,
    logger: logging.Logger,
):
    """Tokenize examples while supervising only the JSON completion."""
    from datasets import Dataset

    encoded = []
    n_skipped = 0
    for ex in examples:
        out = encode_sft_example(ex, tokenizer, max_seq_length)
        if out is None:
            n_skipped += 1
            continue
        encoded.append(out)

    logger.info(f"Encoded {len(encoded)} examples; skipped {n_skipped} "
                f"(too long or empty label).")
    return Dataset.from_list(encoded)


def collate_padded(batch: List[Dict[str, Any]], pad_token_id: int):
    """Right-pad input_ids/attention_mask/labels to max length in the batch."""
    import torch

    max_len = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        out["input_ids"].append(b["input_ids"] + [pad_token_id] * pad_len)
        out["attention_mask"].append(b["attention_mask"] + [0] * pad_len)
        out["labels"].append(b["labels"] + [-100] * pad_len)
    return {k: torch.tensor(v, dtype=torch.long) for k, v in out.items()}


def build_training_arguments(args: argparse.Namespace, out_dir: Path, logger: logging.Logger):
    """Construct TrainingArguments while tolerating HF version API drift."""
    from transformers import TrainingArguments

    params = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "overwrite_output_dir": False,
        "num_train_epochs": args.num_epochs,
        "max_steps": args.max_steps if args.max_steps > 0 else -1,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": args.lr_scheduler_type,
        "gradient_checkpointing": args.gradient_checkpointing,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "logging_dir": str(out_dir / "tb"),
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "eval_steps": args.eval_steps,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "load_best_model_at_end": args.load_best_model_at_end,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "report_to": args.report_to.split(",") if args.report_to != "none" else "none",
        "seed": args.seed,
        "ddp_find_unused_parameters": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
    }

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = "steps"

    supported = {key: value for key, value in kwargs.items() if key in params}
    dropped = sorted(set(kwargs) - set(supported))
    if dropped:
        logger.info(
            "Dropped unsupported TrainingArguments for this transformers version: "
            + ", ".join(dropped)
        )

    return TrainingArguments(**supported)


def build_trainer_kwargs(
    trainer_cls,
    *,
    model,
    targs,
    train_ds,
    eval_ds,
    tokenizer,
    callbacks,
):
    """Build Trainer kwargs across tokenizer/processing_class API versions."""
    params = inspect.signature(trainer_cls.__init__).parameters
    kwargs: Dict[str, Any] = {
        "model": model,
        "args": targs,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": lambda batch: collate_padded(batch, tokenizer.pad_token_id),
        "callbacks": callbacks,
    }
    if "processing_class" in params:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in params:
        kwargs["tokenizer"] = tokenizer
    return kwargs


# ============================================================================
# Custom Trainer subclass for richer logging
# ============================================================================


def make_trainer_class(logger: logging.Logger, eval_examples_for_gen: List[Dict[str, Any]]):
    """Subclass Trainer to log per-step + per-eval extras."""
    from transformers import Trainer

    class JCATrainer(Trainer):
        def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
            super().log(logs, *args, **kwargs)
            # Mirror to our file logger so we capture everything
            step = self.state.global_step
            payload = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
            if payload:
                logger.info(f"step {step}  " + "  ".join(f"{k}={v:.4f}" for k, v in payload.items()))

        def evaluation_loop(self, *args, **kwargs):
            output = super().evaluation_loop(*args, **kwargs)
            # Add a sample-generation block
            if eval_examples_for_gen and os.environ.get("RANK", "0") == "0":
                self._log_sample_generations(eval_examples_for_gen)
            return output

        def _log_sample_generations(self, examples: List[Dict[str, Any]]):
            try:
                model = self.model
                tokenizer = self.processing_class  # newer TRL/HF
            except Exception:
                tokenizer = getattr(self, "tokenizer", None)
            if tokenizer is None:
                return
            model.eval()
            import torch
            for i, ex in enumerate(examples):
                try:
                    prompt = tokenizer.apply_chat_template(
                        ex["messages"], tokenize=False,
                        add_generation_prompt=True, enable_thinking=False,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        ex["messages"], tokenize=False, add_generation_prompt=True,
                    )
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out_ids = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=False,
                        temperature=1.0,
                    )
                gen = tokenizer.decode(
                    out_ids[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True,
                )
                logger.info(
                    f"[eval-sample {i}] step={self.state.global_step}\n"
                    f"  expected: {ex['label'][:200]}\n"
                    f"  generated: {gen[:300]}"
                )

    return JCATrainer


# ============================================================================
# Main
# ============================================================================


def main() -> None:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise SystemExit("--bf16 and --fp16 cannot both be enabled")
    if not 0.0 <= args.max_tokenization_drop_ratio <= 1.0:
        raise SystemExit("--max-tokenization-drop-ratio must be in [0, 1]")
    out_dir = Path(args.out_dir)
    logger = setup_logging(out_dir, args.agent)

    # Resolve base model path
    if args.model_name_or_path is None:
        from jca.src.agents import AGENT_CONFIGS
        args.model_name_or_path = AGENT_CONFIGS[args.agent].base_model
    logger.info(f"Agent: {args.agent}")
    logger.info(f"Base model: {args.model_name_or_path}")
    logger.info(f"Data: {args.data}")
    logger.info(f"Output: {out_dir}")

    # Load + split data
    examples = load_jsonl(args.data)
    logger.info(f"Loaded {len(examples)} examples from {args.data}")

    train_examples, eval_examples, split_stats = split_examples_grouped(
        examples,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        group_by=args.split_group_by,
    )
    logger.info(
        "Split by %s: train=%d rows/%d groups, eval=%d rows/%d groups",
        args.split_group_by,
        split_stats["train_rows"],
        split_stats["train_groups"],
        split_stats["eval_rows"],
        split_stats["eval_groups"],
    )

    # Tokenizer and tokenization preflight
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    import torch

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Tokenizing train set...")
    train_ds = build_dataset(train_examples, tokenizer, args.max_seq_length, logger)
    logger.info("Tokenizing eval set...")
    eval_ds = build_dataset(eval_examples, tokenizer, args.max_seq_length, logger)
    if len(train_ds) == 0 or len(eval_ds) == 0:
        raise RuntimeError(
            "Tokenization left an empty train or eval dataset; increase max sequence "
            "length or rebuild the data."
        )
    encoded_total = len(train_ds) + len(eval_ds)
    tokenization_drop_ratio = 1.0 - encoded_total / len(examples)
    logger.info(
        "Tokenization retained %d/%d examples (drop_ratio=%.4f)",
        encoded_total,
        len(examples),
        tokenization_drop_ratio,
    )
    if tokenization_drop_ratio > args.max_tokenization_drop_ratio:
        raise RuntimeError(
            "Tokenization drop ratio "
            f"{tokenization_drop_ratio:.4f} exceeds "
            f"{args.max_tokenization_drop_ratio:.4f}; increase --max-seq-length."
        )

    # Model and LoRA
    logger.info("Loading model...")
    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    if args.init_lora_adapter is not None:
        from peft import PeftModel
        logger.info(f"Loading existing LoRA adapter from {args.init_lora_adapter}")
        model = PeftModel.from_pretrained(
            model, str(args.init_lora_adapter), is_trainable=True
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    if os.environ.get("RANK", "0") == "0":
        model.print_trainable_parameters()

    # Pick a few eval examples for generation logging
    gen_samples = eval_examples[: args.num_eval_samples_to_generate]

    # TrainingArguments
    from transformers import TrainingArguments, EarlyStoppingCallback

    targs = build_training_arguments(args, out_dir, logger)

    JCATrainer = make_trainer_class(logger, gen_samples)

    trainer = JCATrainer(**build_trainer_kwargs(
        JCATrainer,
        model=model,
        targs=targs,
        train_ds=train_ds,
        eval_ds=eval_ds,
        tokenizer=tokenizer,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    ))

    logger.info("=" * 70)
    logger.info("Starting training")
    logger.info("=" * 70)
    train_result = trainer.train()

    # Final save
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))

    # Final eval
    metrics = trainer.evaluate()
    logger.info(f"Final eval metrics: {metrics}")
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))
        logger.info(f"Final model saved to {final_dir}")
        (final_dir / "final_eval_metrics.json").write_text(
            json.dumps(metrics, indent=2)
        )
        (final_dir / "train_summary.json").write_text(
            json.dumps({
                "global_step": train_result.global_step,
                "training_loss": train_result.training_loss,
                "metrics": train_result.metrics,
            }, indent=2, default=str)
        )

    logger.info("=" * 70)
    logger.info("Training complete.")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
