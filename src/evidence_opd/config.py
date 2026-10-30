from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    base_ref = current.pop("_base_", None)
    if base_ref is None:
        return current
    base_path = (config_path.parent / base_ref).resolve()
    return _merge(load_config(base_path), current)


def validate_config(config: dict[str, Any]) -> None:
    if config.get("evaluation", {}).get("enabled", False):
        raise ValueError("evaluation must remain disabled for this training matrix")
    checkpoint = config.get("checkpoint", {})
    if not checkpoint.get("weights_only", False):
        raise ValueError("intermediate checkpoints must be student weights only")
    if int(checkpoint.get("project_budget_bytes", 0)) <= 0:
        raise ValueError("a positive checkpoint project budget is required")
    if config["teacher"]["tp_size"] != 2:
        raise ValueError("the frozen teacher backend is defined for TP=2")
    if config["communication"]["backend"] != "nccl":
        raise ValueError("full-vocabulary logits must use NCCL, not CPU staging")
    if config["communication"].get("cpu_staging", True):
        raise ValueError("CPU staging is forbidden for exact teacher logits")
    if int(config["communication"].get("kl_chunk_tokens", 0)) <= 0:
        raise ValueError("KL chunk token count must be positive")
    if int(config["communication"].get("timeout_seconds", 600)) <= 0:
        raise ValueError("distributed timeout must be positive")
    if config["training"]["model_dtype"] != "bf16":
        raise ValueError("all nine main runs must use BF16 models")
    if config["training"]["kl_dtype"] != "fp32":
        raise ValueError("KL arithmetic must use FP32")
    if int(config["data"]["student_max_batch_tokens"]) <= 0:
        raise ValueError("student token budget must be positive")
    if int(config["data"]["teacher_max_batch_tokens"]) <= 0:
        raise ValueError("teacher token budget must be positive")
    if config.get("pipeline", {}).get("enabled", False):
        raise ValueError("async pipeline remains gated until later correctness testing")
    if config.get("vllm", {}).get("student_rollout", {}).get("enabled", False):
        raise ValueError("vLLM rollout remains gated until profiling after the stable backend")
    if config.get("vllm", {}).get("teacher_forced_prefix", {}).get("enabled", False):
        raise ValueError("vLLM teacher scoring is not part of the stable exact-KL backend")
