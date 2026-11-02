from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"
    return tokenizer


def load_student(config: dict[str, Any], device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["student"],
        local_files_only=True,
        trust_remote_code=bool(config["models"].get("trust_remote_code", False)),
        dtype=torch.bfloat16,
        attn_implementation=config["teacher"].get("attention_backend", "sdpa"),
    ).to(device)
    model.config.use_cache = False
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def load_ema_teacher(config: dict[str, Any], device: torch.device):
    model = AutoModelForCausalLM.from_pretrained(
        config["models"]["ema_initial_teacher"],
        local_files_only=True,
        trust_remote_code=bool(config["models"].get("trust_remote_code", False)),
        dtype=torch.bfloat16,
        attn_implementation=config["teacher"].get("attention_backend", "sdpa"),
    ).to(device)
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    return model


def vocab_mapping(path: str) -> dict[str, int]:
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    return tokenizer.get_vocab()


def assert_vocabulary_alignment(config: dict[str, Any]) -> None:
    student = vocab_mapping(config["models"]["student_tokenizer"])
    teacher = vocab_mapping(config["models"]["teacher_tokenizer"])
    if student != teacher:
        raise ValueError("student/teacher token-to-id maps differ; exact vocabulary KL is undefined")


def model_identity(path: str) -> dict[str, Any]:
    config_path = Path(path) / "config.json"
    with config_path.open(encoding="utf-8") as handle:
        model_config = json.load(handle)
    return {
        "path": str(Path(path).resolve()),
        "model_type": model_config.get("model_type"),
        "architectures": model_config.get("architectures"),
        "vocab_size": model_config.get("vocab_size"),
        "hidden_size": model_config.get("hidden_size"),
        "num_hidden_layers": model_config.get("num_hidden_layers"),
    }

