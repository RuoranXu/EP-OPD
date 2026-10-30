#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/"src"),str(PROJECT)]

from evidence_opd.config import load_config
from evidence_opd.distributed import close_distributed, initialize_distributed
from evidence_opd.train import run_training


def assert_gpu_mapping() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must explicitly select exactly three GPUs")
    normalized = [item.strip() for item in visible.split(",")]
    if len(normalized) != 3 or len(set(normalized)) != 3 or not all(
        item.isdigit() for item in normalized
    ):
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must contain three unique physical GPU indices; "
            f"received {visible!r}"
        )
    excluded = {
        item.strip()
        for item in os.environ.get("SCI_OPD_EXCLUDED_GPUS", "").split(",")
        if item.strip()
    }
    selected_excluded = sorted(set(normalized) & excluded)
    if selected_excluded:
        raise RuntimeError(f"selected excluded physical GPUs: {','.join(selected_excluded)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if (PROJECT / "run_controls" / (config["run"]["name"] + ".disabled")).exists():
        print(f"[skip] {config['run']['name']} disabled by user", flush=True)
        return
    if not os.environ.get('EP_OPD_VERL_WORKER'):assert_gpu_mapping()
    timeout_seconds = int(config.get("communication", {}).get("timeout_seconds", 600))
    context = initialize_distributed(
        timeout_seconds=timeout_seconds,
        eager_nccl_init=config.get("communication", {}).get("eager_nccl_init", True),
    )
    try:
        run_training(config, context)
    except BaseException:
        # Do not enter a success barrier after one rank has already failed.
        import torch.distributed as dist

        if dist.is_initialized():
            dist.destroy_process_group()
        raise
    else:
        close_distributed()


if __name__ == "__main__":
    main()
