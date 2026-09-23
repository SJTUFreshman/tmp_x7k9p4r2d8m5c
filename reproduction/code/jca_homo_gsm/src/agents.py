"""Agent configuration: 3 heterogeneous Qwen3 base models + LoRA + system prompt.

Per design.md §1.1 / §2.1: same prompt template, only AGENT_ID differs.

Public API:
    AGENT_CONFIGS: dict[str, AgentConfig]
    AGENT_IDS: list[str]
    START_AGENT: str
    render_system_prompt(agent_id) -> str
    load_agent_with_lora(agent_id, lora_ckpt=None) -> model
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================================
# Agent specs
# ============================================================================


@dataclass(frozen=True)
class AgentConfig:
    id: str                          # "A1" / "A2" / "A3"
    base_model: str                  # HuggingFace path
    lora_rank: int = 16


AGENT_CONFIGS: Dict[str, AgentConfig] = {
    "A1": AgentConfig(id="A1", base_model="Qwen/Qwen3-1.5B-Instruct"),
    "A2": AgentConfig(id="A2", base_model="Qwen/Qwen3-4B-Instruct"),
    "A3": AgentConfig(id="A3", base_model="Qwen/Qwen3-8B-Instruct"),
}

AGENT_IDS: List[str] = ["A1", "A2", "A3"]
START_AGENT: str = "A1"


# ============================================================================
# Prompt rendering
# ============================================================================


_PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "agent_system.md"
_DISTILL_PROMPT_TEMPLATE_PATH = Path(__file__).parent.parent / "prompts" / "distill_system.md"


def _format_other_agents(agent_id: str) -> str:
    """E.g. for A1 -> 'A2 and A3'."""
    others = [a for a in AGENT_IDS if a != agent_id]
    if len(others) == 2:
        return f"{others[0]} and {others[1]}"
    return ", ".join(others)


def _format_a_list() -> str:
    """E.g. 'A1, A2, A3' (full list, ID order)."""
    return ", ".join(AGENT_IDS)


def render_system_prompt(agent_id: str) -> str:
    """Load agent_system.md and fill placeholders.

    Per design.md §2.1: 3 agents share the same template byte-for-byte
    except {AGENT_ID} and {OTHER_AGENTS}. NO capacity / role / hierarchy
    information is leaked.
    """
    if agent_id not in AGENT_CONFIGS:
        raise KeyError(f"Unknown agent_id: {agent_id}")

    template_text = _PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    template = _extract_first_code_block(template_text)
    return (
        template
        .replace("{AGENT_ID}", agent_id)
        .replace("{OTHER_AGENTS}", _format_other_agents(agent_id))
    )


def render_distill_system_prompt(
    agent_id: str,
    *,
    demo_mode: str = "exploratory",
) -> str:
    """Load distill_system.md and fill placeholders.

    Used ONLY for self-distillation data generation. The model trained from
    these demos sees `render_system_prompt(...)` (the deployment prompt) at
    inference time. Behaviors encouraged here are baked into the LoRA.

    `demo_mode` ∈ {"agree", "disagree", "refine", "exploratory"} — passed in
    to bias one trajectory toward a particular verification mode for
    diversity across demos.
    """
    if agent_id not in AGENT_CONFIGS:
        raise KeyError(f"Unknown agent_id: {agent_id}")

    template_text = _DISTILL_PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    template = _extract_first_code_block(template_text)
    return (
        template
        .replace("{AGENT_ID}", agent_id)
        .replace("{A_LIST}", _format_a_list())
        .replace("{OTHER_AGENTS}", _format_other_agents(agent_id))
        .replace("{DEMO_MODE}", demo_mode)
    )


def _extract_first_code_block(markdown_text: str) -> str:
    """Return the body of the first fenced code block in a markdown file."""
    match = re.search(r"```(?:[^\n`]*)\n?(.*?)```", markdown_text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No fenced prompt block found in {_PROMPT_TEMPLATE_PATH}")
    return match.group(1).strip()


# ============================================================================
# Model loading
# ============================================================================


def load_agent_with_lora(
    agent_id: str,
    lora_ckpt: Optional[str] = None,
    *,
    device_map: str = "auto",
):
    """Load base + (optionally pre-trained) LoRA adapter for a given agent.

    NOTE: At init time (lora_ckpt is None), B matrix is zero so the model
    behaves identically to the base. See design.md §6.1 for why SFT warm-up
    is required before RL.
    """
    if agent_id not in AGENT_CONFIGS:
        raise KeyError(f"Unknown agent_id: {agent_id}")

    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, PeftModel, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "load_agent_with_lora requires transformers and peft. "
            "Install them before loading real agents."
        ) from exc

    cfg = AGENT_CONFIGS[agent_id]
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        device_map=device_map,
        torch_dtype="auto",
        trust_remote_code=True,
    )

    if lora_ckpt is not None:
        model = PeftModel.from_pretrained(model, lora_ckpt)
    else:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_rank * 2,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, lora_config)

    return model, tokenizer
