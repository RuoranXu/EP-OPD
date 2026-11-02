from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any


def directory_bytes(path: str | Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            total += (Path(root) / filename).stat().st_size
    return total


def planned_checkpoint_steps(total_steps: int, fractions: list[float]) -> set[int]:
    return {
        max(1, min(total_steps - 1, round(total_steps * float(fraction))))
        for fraction in fractions
        if total_steps > 1 and 0.0 < float(fraction) < 1.0
    }


def ensure_storage_budget(
    output_root: str | Path,
    expected_new_bytes: int,
    project_budget_bytes: int,
    minimum_filesystem_free_bytes: int,
) -> dict[str, int]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    used = directory_bytes(root)
    disk = shutil.disk_usage(root)
    if used + expected_new_bytes > project_budget_bytes:
        raise RuntimeError(
            f"checkpoint plan exceeds project cap: used={used}, new={expected_new_bytes}, "
            f"cap={project_budget_bytes}"
        )
    if disk.free - expected_new_bytes < minimum_filesystem_free_bytes:
        raise RuntimeError(
            f"checkpoint plan violates filesystem reserve: free={disk.free}, "
            f"new={expected_new_bytes}, reserve={minimum_filesystem_free_bytes}"
        )
    return {
        "output_bytes_existing": used,
        "expected_new_bytes": expected_new_bytes,
        "project_budget_bytes": project_budget_bytes,
        "filesystem_free_bytes": disk.free,
        "minimum_filesystem_free_bytes": minimum_filesystem_free_bytes,
    }


def save_student_weights(model, tokenizer, destination: str | Path, metadata: dict[str, Any]) -> None:
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {destination}")
    temporary = destination.parent / f".{destination.name}.incomplete"
    if temporary.exists():
        raise FileExistsError(f"incomplete checkpoint already exists: {temporary}")
    temporary.mkdir(parents=True)
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="4GB")
    tokenizer.save_pretrained(temporary)
    with (temporary / "checkpoint_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.rename(destination)

