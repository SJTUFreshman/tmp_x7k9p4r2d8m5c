#!/usr/bin/env python3
"""Resume-capable LoRA SFT entry point using the repository's SFT utilities."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_PARENT = REPO_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
BUNDLED_PACKAGE_PARENT = Path(__file__).resolve().parent / "vendor"
if str(BUNDLED_PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(BUNDLED_PACKAGE_PARENT))

from jca.scripts import sft_train as shared  # noqa: E402


EXPECTED_BUILDER_VERSION = "math_specific_fixed_a1_rollout_sft_v2"


def validate_supervision_contract(examples: list[dict], agent: str) -> None:
    if not examples:
        raise SystemExit("SFT data is empty")
    for row_index, row in enumerate(examples):
        if row.get("builder_version") != EXPECTED_BUILDER_VERSION:
            raise SystemExit(
                "refusing non-rollout MATH SFT data: "
                f"row {row_index} builder_version={row.get('builder_version')!r}"
            )
        if row.get("supervision_source") != "controlled_model_rollout":
            raise SystemExit(
                f"refusing synthetic/official-solution supervision at row {row_index}"
            )
        if row.get("agent_id") != agent:
            raise SystemExit(
                f"role-mismatched SFT row {row_index}: {row.get('agent_id')!r} != {agent}"
            )
        if not str(row.get("sample_type") or "").startswith("rollout_"):
            raise SystemExit(f"non-rollout SFT sample type at row {row_index}")
        if not row.get("trajectory_id") or int(row.get("plan_index", -1)) < 0:
            raise SystemExit(f"missing rollout provenance at row {row_index}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent", required=True, choices=("A1", "A2", "A3"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--eval-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-group-by",
        choices=("problem_id", "trajectory", "row"),
        default="problem_id",
    )
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-epochs", type=float, default=3.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--max-tokenization-drop-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument(
        "--load-best-model-at-end",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=10)
    parser.add_argument("--report-to", default="none")
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--torch-compile", action="store_true", default=False)
    parser.add_argument("--num-eval-samples-to-generate", type=int, default=2)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bf16 and args.fp16:
        raise SystemExit("--bf16 and --fp16 cannot both be enabled")
    if not 0.0 <= args.max_tokenization_drop_ratio <= 1.0:
        raise SystemExit("--max-tokenization-drop-ratio must be in [0, 1]")
    if args.resume_from_checkpoint is not None:
        trainer_state = args.resume_from_checkpoint / "trainer_state.json"
        if not trainer_state.is_file():
            raise SystemExit(f"checkpoint is missing trainer_state.json: {args.resume_from_checkpoint}")

    logger = shared.setup_logging(args.out_dir, args.agent)
    examples = shared.load_jsonl(args.data)
    validate_supervision_contract(examples, args.agent)
    train_examples, eval_examples, split_stats = shared.split_examples_grouped(
        examples,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
        group_by=args.split_group_by,
    )
    logger.info(
        "MATH-specific split: train=%d/%d groups eval=%d/%d groups",
        split_stats["train_rows"],
        split_stats["train_groups"],
        split_stats["eval_rows"],
        split_stats["eval_groups"],
    )

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_ds = shared.build_dataset(
        train_examples,
        tokenizer,
        args.max_seq_length,
        logger,
    )
    eval_ds = shared.build_dataset(
        eval_examples,
        tokenizer,
        args.max_seq_length,
        logger,
    )
    encoded_total = len(train_ds) + len(eval_ds)
    drop_ratio = 1.0 - encoded_total / len(examples)
    if not train_ds or not eval_ds:
        raise RuntimeError("tokenization left an empty train or eval split")
    if drop_ratio > args.max_tokenization_drop_ratio:
        raise RuntimeError(
            f"tokenization drop ratio {drop_ratio:.4f} exceeds "
            f"{args.max_tokenization_drop_ratio:.4f}"
        )
    logger.info("Tokenization retained %d/%d rows", encoded_total, len(examples))

    dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    )
    if os.environ.get("RANK", "0") == "0":
        model.print_trainable_parameters()

    training_args = shared.build_training_arguments(args, args.out_dir, logger)
    trainer_class = shared.make_trainer_class(
        logger,
        eval_examples[: args.num_eval_samples_to_generate],
    )
    callbacks = []
    if args.load_best_model_at_end:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        )
    trainer = trainer_class(
        **shared.build_trainer_kwargs(
            trainer_class,
            model=model,
            targs=training_args,
            train_ds=train_ds,
            eval_ds=eval_ds,
            tokenizer=tokenizer,
            callbacks=callbacks,
        )
    )
    logger.info("Starting MATH-specific SFT; resume=%s", args.resume_from_checkpoint)
    result = trainer.train(
        resume_from_checkpoint=(
            str(args.resume_from_checkpoint)
            if args.resume_from_checkpoint is not None
            else None
        )
    )
    final_dir = args.out_dir / "final"
    trainer.save_model(str(final_dir))
    metrics = trainer.evaluate()
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))
        (final_dir / "final_eval_metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )
        (final_dir / "train_summary.json").write_text(
            json.dumps(
                {
                    "agent": args.agent,
                    "dataset": "MATH",
                    "source_split": "train",
                    "builder_version": EXPECTED_BUILDER_VERSION,
                    "supervision_source": "controlled_model_rollout",
                    "initialization": "fresh LoRA on the role base model",
                    "global_step": result.global_step,
                    "training_loss": result.training_loss,
                    "resume_from_checkpoint": (
                        str(args.resume_from_checkpoint)
                        if args.resume_from_checkpoint is not None
                        else None
                    ),
                    "split_stats": split_stats,
                    "metrics": result.metrics,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    logger.info("Final MATH-specific SFT adapter saved: %s", final_dir)


if __name__ == "__main__":
    main()
