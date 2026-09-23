"""RWR trainer used by the MuSiQue no-SFT-initialization ablation."""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
for path in (REPO_ROOT.parent, SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from rl_train import (  # noqa: E402
    collate_batch,
    compute_rwr_loss,
    load_rollouts,
    setup_logging,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RWR from a fresh base-model LoRA, optionally continuing an adapter.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--agent", required=True, choices=["A1", "A2", "A3"])
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--reward-field", default="reward")
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--kl-coef", type=float, default=0.2)
    parser.add_argument("--num-epochs", type=float, default=2.0)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--report-to", default="tensorboard")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.rollout.is_file():
        raise SystemExit(f"Rollout not found: {args.rollout}")
    if not Path(args.model_name_or_path).is_dir():
        raise SystemExit(f"Base model not found: {args.model_name_or_path}")
    if args.init_adapter is not None:
        weights = args.init_adapter / "adapter_model.safetensors"
        if not weights.is_file():
            raise SystemExit(f"Initial adapter weights not found: {weights}")
    if args.num_epochs <= 0 or args.num_epochs != int(args.num_epochs):
        raise SystemExit("--num-epochs must be a positive integer")


def build_policy_and_reference(args: argparse.Namespace, dtype, logger):
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM

    policy_base = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if args.init_adapter is None:
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
        )
        policy = get_peft_model(policy_base, config)
        initialization = "base"
    else:
        policy = PeftModel.from_pretrained(
            policy_base,
            str(args.init_adapter),
            is_trainable=True,
        )
        initialization = str(args.init_adapter)

    reference = None
    if args.kl_coef > 0:
        logger.info("Loading frozen initialization reference...")
        ref_base = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        if args.init_adapter is None:
            reference = ref_base
        else:
            reference = PeftModel.from_pretrained(
                ref_base,
                str(args.init_adapter),
                is_trainable=False,
            )
        reference.eval()
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    return policy, reference, initialization


def main() -> None:
    args = parse_args()
    validate_args(args)
    records = load_rollouts(args.rollout, args.agent, args.reward_field)
    if not records:
        raise SystemExit(f"No {args.agent} records found in {args.rollout}")
    rewards = [record.reward for record in records]

    if args.dry_run:
        print(json.dumps({
            "agent": args.agent,
            "model": args.model_name_or_path,
            "initialization": str(args.init_adapter) if args.init_adapter else "base",
            "rollout": str(args.rollout),
            "reward_field": args.reward_field,
            "records": len(records),
            "reward_min": min(rewards),
            "reward_max": max(rewards),
            "out_dir": str(args.out_dir),
        }, indent=2))
        return

    out_dir = args.out_dir
    logger = setup_logging(out_dir, args.agent)
    logger.info("Agent: %s", args.agent)
    logger.info("Base model: %s", args.model_name_or_path)
    logger.info("Initialization: %s", args.init_adapter or "fresh base LoRA")
    logger.info("Rollout: %s", args.rollout)
    logger.info("Reward field: %s", args.reward_field)
    logger.info("Loaded %d records", len(records))
    logger.info(
        "Reward stats: mean=%.4f min=%.4f max=%.4f",
        sum(rewards) / len(rewards),
        min(rewards),
        max(rewards),
    )

    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

    accelerator = Accelerator()
    set_seed(args.seed, device_specific=True)
    is_main = accelerator.is_main_process
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    logger.info("Accelerator: rank=%d/%d device=%s", rank, world_size, accelerator.device)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    policy, reference, initialization = build_policy_and_reference(args, dtype, logger)
    if args.gradient_checkpointing:
        policy.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        policy.enable_input_require_grads()
    if is_main:
        policy.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    effective_batch = (
        world_size
        * args.per_device_batch_size
        * args.gradient_accumulation_steps
    )
    steps_per_epoch = max(1, math.ceil(len(records) / effective_batch))
    total_steps = steps_per_epoch * int(args.num_epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    policy, optimizer = accelerator.prepare(policy, optimizer)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    if reference is not None:
        reference = reference.to(accelerator.device)
    logger.info(
        "total_records=%d world_size=%d effective_batch=%d total_steps=%d warmup=%d",
        len(records),
        world_size,
        effective_batch,
        total_steps,
        warmup_steps,
    )

    rng = random.Random(args.seed)
    global_step = 0
    accumulated_loss = 0.0
    accumulated_metrics = defaultdict(float)
    accumulated_count = 0
    optimizer.zero_grad()

    for epoch in range(int(args.num_epochs)):
        rng.shuffle(records)
        padded_size = math.ceil(len(records) / world_size) * world_size
        padded_records = records + records[: padded_size - len(records)]
        rank_records = padded_records[rank::world_size]
        if is_main:
            logger.info(
                "Epoch %d/%d total_n=%d padded_n=%d per_rank_n=%d",
                epoch + 1,
                int(args.num_epochs),
                len(records),
                len(padded_records),
                len(rank_records),
            )

        batch_size = args.per_device_batch_size
        micro_batches = math.ceil(len(rank_records) / batch_size)
        for micro_index, offset in enumerate(range(0, len(rank_records), batch_size)):
            batch = rank_records[offset: offset + batch_size]
            tensors = collate_batch(batch, tokenizer, args.max_seq_length)
            if tensors is None:
                pad_id = tokenizer.pad_token_id
                input_ids = torch.full((1, 8), pad_id, dtype=torch.long)
                attention_mask = torch.ones((1, 8), dtype=torch.long)
                labels = torch.full((1, 8), -100, dtype=torch.long)
                labels[0, -1] = pad_id
                batch_rewards = torch.zeros((1,), dtype=torch.float32)
            else:
                input_ids, attention_mask, labels, batch_rewards = tensors

            input_ids = input_ids.to(accelerator.device)
            attention_mask = attention_mask.to(accelerator.device)
            labels = labels.to(accelerator.device)
            batch_rewards = batch_rewards.to(accelerator.device)
            loss, metrics = compute_rwr_loss(
                policy,
                reference,
                input_ids,
                attention_mask,
                labels,
                batch_rewards,
                kl_coef=args.kl_coef,
            )
            accelerator.backward(loss / args.gradient_accumulation_steps)
            accumulated_loss += loss.item()
            for name, value in metrics.items():
                accumulated_metrics[name] += value
            accumulated_count += 1

            should_step = (
                (micro_index + 1) % args.gradient_accumulation_steps == 0
                or (micro_index + 1) == micro_batches
            )
            if not should_step:
                continue
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(
                    [parameter for parameter in policy.parameters() if parameter.requires_grad],
                    max_norm=1.0,
                )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.logging_steps == 0 and is_main:
                count = max(accumulated_count, 1)
                logger.info(
                    "step %d loss=%.4f kl=%.4f reward_mean=%.3f lr=%.2e",
                    global_step,
                    accumulated_loss / count,
                    accumulated_metrics["kl_loss"] / count,
                    accumulated_metrics["reward_mean"] / count,
                    scheduler.get_last_lr()[0],
                )
            if global_step % args.logging_steps == 0:
                accumulated_loss = 0.0
                accumulated_metrics = defaultdict(float)
                accumulated_count = 0

            if global_step % args.save_steps == 0:
                accelerator.wait_for_everyone()
                if is_main:
                    checkpoint = out_dir / f"checkpoint-{global_step}"
                    accelerator.unwrap_model(policy).save_pretrained(str(checkpoint))
                    tokenizer.save_pretrained(str(checkpoint))
                    checkpoints = sorted(
                        out_dir.glob("checkpoint-*"),
                        key=lambda path: int(path.name.split("-")[1]),
                    )
                    for old_checkpoint in checkpoints[: -args.save_total_limit]:
                        shutil.rmtree(old_checkpoint, ignore_errors=True)
                accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    if is_main:
        final_dir = out_dir / "final"
        accelerator.unwrap_model(policy).save_pretrained(str(final_dir))
        tokenizer.save_pretrained(str(final_dir))
        (final_dir / "rl_train_summary.json").write_text(
            json.dumps({
                "agent": args.agent,
                "global_step": global_step,
                "total_records": len(records),
                "algorithm": "rwr",
                "initialization": initialization,
                "reward_field": args.reward_field,
                "kl_coef": args.kl_coef,
                "world_size": world_size,
            }, indent=2),
            encoding="utf-8",
        )
        logger.info("Final adapter saved: %s", final_dir)
    logger.info("RL training complete.")


if __name__ == "__main__":
    main()

