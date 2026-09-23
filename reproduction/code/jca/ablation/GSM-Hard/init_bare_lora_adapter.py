#!/usr/bin/env python3
"""Create a no-op LoRA adapter for base-model-initialized RL ablations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Initialize a fresh LoRA whose initial policy is functionally equal "
            "to the bare base model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-name-or-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata-kind", default="without_sft_warmup")
    parser.add_argument("--metadata-file", default="ablation_init.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model_name_or_path.is_dir():
        raise SystemExit(f"base model not found: {args.model_name_or_path}")
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output_dir}")
    if args.lora_rank <= 0 or args.lora_alpha <= 0:
        raise SystemExit("LoRA rank and alpha must be positive")
    if not 0.0 <= args.lora_dropout < 1.0:
        raise SystemExit("LoRA dropout must be in [0, 1)")

    import numpy as np
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model_name_or_path), trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_name_or_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=list(TARGET_MODULES),
        bias="none",
        init_lora_weights=True,
    )
    model = get_peft_model(model, config)

    lora_b = [
        parameter.detach()
        for name, parameter in model.named_parameters()
        if "lora_B" in name
    ]
    if not lora_b:
        raise RuntimeError("no LoRA B matrices were created")
    if any(torch.count_nonzero(parameter).item() for parameter in lora_b):
        raise RuntimeError("fresh LoRA is not a no-op: a B matrix is nonzero")

    args.output_dir.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "initial_policy": "bare_base_model_plus_fresh_noop_lora",
        "base_model": str(args.model_name_or_path),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": list(TARGET_MODULES),
        "seed": args.seed,
        "lora_b_zero_verified": True,
    }
    if args.metadata_kind == "without_sft_warmup":
        metadata["ablation"] = args.metadata_kind
    else:
        metadata["purpose"] = args.metadata_kind
    (args.output_dir / args.metadata_file).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
