from __future__ import annotations

from typing import Any


def teacher_prompt_key(config: dict[str, Any]) -> str:
    mode = config["method"].get("teacher_prompt", "normal")
    mapping = {
        "normal": "teacher_normal_ids",
        "pi_ids": "teacher_pi_ids",
        "pi_span": "teacher_pi_span_ids",
    }
    try:
        return mapping[mode]
    except KeyError as error:
        raise ValueError(f"unknown teacher prompt mode: {mode}") from error


def is_ema(config: dict[str, Any]) -> bool:
    return config["method"]["family"] == "ema_teacher"

