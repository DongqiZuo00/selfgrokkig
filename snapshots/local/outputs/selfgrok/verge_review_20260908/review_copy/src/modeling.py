from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from common import MODEL_REPO_ID, MODEL_ROOT


def assert_mistral_backbone() -> None:
    identity = f"{MODEL_REPO_ID} {MODEL_ROOT}".lower()
    if "qwen" in identity or "mistral" not in identity:
        raise RuntimeError(f"only the frozen Mistral backbone is permitted: {identity}")


def load_tokenizer(padding_side: str = "left"):
    assert_mistral_backbone()
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_ROOT),
        padding_side=padding_side,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_text_model(
    *,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "cuda:0",
    attn_implementation: str = "sdpa",
) -> Any:
    """Load only the causal-language component of the frozen Ministral checkpoint."""
    assert_mistral_backbone()
    config = AutoConfig.from_pretrained(str(MODEL_ROOT))
    if config.model_type != "mistral3":
        return AutoModelForCausalLM.from_pretrained(
            str(MODEL_ROOT),
            dtype=dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )

    from transformers import Mistral3ForConditionalGeneration

    return Mistral3ForConditionalGeneration.from_pretrained(
        str(MODEL_ROOT),
        dtype=dtype,
        device_map=device_map,
        attn_implementation=attn_implementation,
    )


def lora_target_modules(configured: str) -> str:
    """Restrict LoRA to the language tower of the text-only Ministral run."""
    if configured != "all-linear":
        return configured
    return (
        r"model\.language_model\.layers\.\d+\."
        r"(?:self_attn\.(?:q_proj|k_proj|v_proj|o_proj)|"
        r"mlp\.(?:gate_proj|up_proj|down_proj))"
    )


def assert_adapter_backbone(adapter: Path) -> None:
    config_path = adapter / "adapter_config.json"
    if not config_path.exists():
        raise RuntimeError(f"adapter config is missing: {config_path}")
    if "qwen" in config_path.read_text(encoding="utf-8").lower():
        raise RuntimeError(f"refusing prohibited Qwen adapter: {adapter}")
