from __future__ import annotations

import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import yaml

from evidence_opd.config import validate_config
from evidence_opd.data import ReplayBatch, TokenizedCorpus, collate_replay, read_batch_manifest
from evidence_opd.distributed import DistributedContext
from evidence_opd.losses import exact_kl_per_token
from evidence_opd.methods import is_ema, teacher_prompt_key
from evidence_opd.models import (
    assert_vocabulary_alignment,
    load_ema_teacher,
    load_student,
    load_tokenizer,
    model_identity,
)
from evidence_opd.profiling import JsonlMetrics
from evidence_opd.rollout import generate_responses
from evidence_opd.teacher.hf_tp_worker import create_teacher_mesh, load_tp_teacher
from evidence_opd.teacher.nccl_transport import (
    broadcast_replay_batch,
    receive_logits,
    send_logits,
    token_chunks,
    update_ema_across_ranks,
)
from evidence_opd.teacher.response_logits import selected_response_hidden
from evidence_opd.train.checkpoint import (
    directory_bytes,
    ensure_storage_budget,
    planned_checkpoint_steps,
    save_student_weights,
)
from evidence_opd.train.optimizer import build_optimizer, build_scheduler


def _synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


@torch.inference_mode()
def _entropy_selection(
    model,
    prompts: list[list[int]],
    responses: torch.Tensor,
    response_mask: torch.Tensor,
    pad_id: int,
    device: torch.device,
    policy_version: int,
    fraction: float,
    chunk_tokens: int,
) -> torch.Tensor:
    if fraction >= 1.0:
        return response_mask.bool()
    was_training = model.training
    model.eval()
    batch = collate_replay(
        prompts,
        responses,
        response_mask,
        pad_id,
        device,
        policy_version,
    )
    hidden = selected_response_hidden(model, batch)
    flat_entropy = torch.empty(hidden.shape[0], dtype=torch.float32, device=device)
    for start, end in token_chunks(hidden.shape[0], chunk_tokens):
        logp = torch.log_softmax(model.lm_head(hidden[start:end]).float(), dim=-1)
        flat_entropy[start:end] = -(logp.exp() * logp).sum(dim=-1)
    entropies = torch.full(response_mask.shape, -torch.inf, dtype=torch.float32, device=device)
    coordinates = batch.selected_mask.nonzero(as_tuple=False)
    entropies[coordinates[:, 0], coordinates[:, 1]] = flat_entropy
    selected = torch.zeros_like(response_mask, dtype=torch.bool, device=device)
    active_mask = response_mask.to(device=device, dtype=torch.bool)
    for row in range(active_mask.shape[0]):
        active = active_mask[row].nonzero(as_tuple=False).flatten()
        if active.numel() == 0:
            continue
        keep = max(1, int(math.ceil(active.numel() * fraction)))
        local = torch.topk(entropies[row, active], keep, largest=True, sorted=False).indices
        selected[row, active[local]] = True
    model.train(was_training)
    return selected


def _worker_stats(
    context: DistributedContext, elapsed_ms: float, peak_gb: float
) -> None:
    if context.rank in (1, 2):
        stats = torch.tensor([elapsed_ms, peak_gb], dtype=torch.float64, device=context.device)
        dist.send(stats, dst=0)


def _receive_worker_stats(context: DistributedContext) -> dict[str, float]:
    values: dict[str, float] = {}
    for rank in (1, 2):
        stats = torch.empty(2, dtype=torch.float64, device=context.device)
        dist.recv(stats, src=rank)
        values[f"time/teacher_rank{rank - 1}_ms"] = float(stats[0].item())
        values[f"memory/teacher_rank{rank - 1}_peak_gb"] = float(stats[1].item())
    return values


def _save_run_manifest(
    config: dict[str, Any], output_dir: Path, storage: dict[str, int], total_steps: int
) -> None:
    visible = [
        value.strip()
        for value in os.environ.get("EP_OPD_GROUP_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",")
        if value.strip()
    ]
    manifest = {
        "run": config["run"]["name"],
        "physical_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "physical_gpu_mapping": {
            "rank0_student": visible[0] if len(visible) > 0 else None,
            "rank1_teacher": visible[1] if len(visible) > 1 else None,
            "rank2_teacher": visible[2] if len(visible) > 2 else None,
        },
        "student": model_identity(config["models"]["student"]),
        "teacher": model_identity(
            config["models"]["ema_initial_teacher"] if is_ema(config) else config["models"]["teacher"]
        ),
        "total_optimizer_steps": total_steps,
        "storage_plan": storage,
        "evaluation_enabled": False,
    }
    with (output_dir / "run_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def _checkpoint(
    student,
    tokenizer,
    output_dir: Path,
    name: str,
    run_name: str,
    optimizer_step: int,
    policy_version: int,
) -> None:
    save_student_weights(
        student,
        tokenizer,
        output_dir / name,
        {
            "run": run_name,
            "optimizer_step": optimizer_step,
            "policy_version": policy_version,
            "weights_only": True,
            "optimizer_state_saved": False,
            "evaluation_run": False,
        },
    )


def run_training(config: dict[str, Any], context: DistributedContext) -> None:
    validate_config(config)
    ema_method = is_ema(config)
    chunk_tokens = int(config["communication"]["kl_chunk_tokens"])
    run_name = config["run"]["name"]
    output_dir = Path(config["checkpoint"]["output_root"]) / run_name

    student = tokenizer = optimizer = scheduler = corpus = metrics = None
    if context.is_student:
        assert_vocabulary_alignment(config)
        tokenizer = load_tokenizer(config["models"]["student_tokenizer"])
        student = load_student(config, context.device)
        corpus = TokenizedCorpus(config["data"]["tokenized_path"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / "final_student").exists():
            raise FileExistsError(f"completed run already exists: {output_dir / 'final_student'}")
        metrics = JsonlMetrics(output_dir / "metrics.jsonl")

    teacher = None
    teacher_mesh = None
    if ema_method:
        if context.is_teacher_leader:
            teacher = load_ema_teacher(config, context.device)
    else:
        teacher_mesh = create_teacher_mesh()
        teacher = load_tp_teacher(config, context, teacher_mesh)

    base_manifest = read_batch_manifest(config["data"]["batch_manifest"])
    epochs = int(config["training"]["epochs"])
    repeats = 1 if config["data"].get("manifest_includes_epochs", False) else epochs
    manifest = [dict(row) for _epoch in range(repeats) for row in base_manifest]
    max_steps = config["training"].get("max_optimizer_steps")
    if max_steps is not None:
        manifest = [row for row in manifest if int(row["optimizer_step"]) < int(max_steps)]
    if not manifest:
        raise ValueError("batch manifest selected zero microbatches")
    # Re-number optimizer-step groups after epoch replication while preserving order.
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    for index, row in enumerate(manifest):
        row["runtime_optimizer_step"] = index // accumulation
    group_counts = Counter(int(row["runtime_optimizer_step"]) for row in manifest)
    total_optimizer_steps = max(group_counts) + 1
    milestone_steps = planned_checkpoint_steps(
        total_optimizer_steps, [float(v) for v in config["checkpoint"]["milestone_fractions"]]
    )

    if context.is_student:
        checkpoint_bytes = directory_bytes(config["models"]["student"])
        expected_new = checkpoint_bytes * (len(milestone_steps) + 1)
        storage = ensure_storage_budget(
            config["checkpoint"]["output_root"],
            expected_new,
            int(config["checkpoint"]["project_budget_bytes"]),
            int(config["checkpoint"]["minimum_filesystem_free_bytes"]),
        )
        storage.update(
            {
                "checkpoint_bytes_estimate": checkpoint_bytes,
                "checkpoints_this_run": len(milestone_steps) + 1,
            }
        )
        _save_run_manifest(config, output_dir, storage, total_optimizer_steps)
        optimizer = build_optimizer(student, config)
        scheduler = build_scheduler(
            optimizer, total_optimizer_steps, float(config["training"].get("warmup_ratio", 0.0)),
            schedule=config["training"].get("schedule", "linear"),
        )
        optimizer.zero_grad(set_to_none=True)

    dist.barrier()
    policy_version = 0
    completed_optimizer_steps = 0
    prompt_key = teacher_prompt_key(config)
    entropy_fraction = float(config["method"].get("entropy_fraction", 1.0))
    divergence = config["method"]["divergence"]
    temperature = float(config["method"].get("temperature", 1.0))
    temperature_scale = bool(config["method"].get("temperature_squared_scale", False))
    seed = int(config["experiment"]["seed"])
    vocab_size = int(model_identity(config["models"]["student"])["vocab_size"])

    for microbatch_index, manifest_row in enumerate(manifest):
        runtime_step = int(manifest_row["runtime_optimizer_step"])
        group_size = group_counts[runtime_step]
        source_teacher_batch: ReplayBatch | None = None
        student_batch: ReplayBatch | None = None
        rollout_ms = entropy_ms = 0.0
        generated_tokens = 0

        if context.is_student:
            print(
                f"[progress] run={run_name} microbatch={microbatch_index} "
                f"optimizer_step={completed_optimizer_steps} phase=rollout",
                flush=True,
            )
            rows = corpus.get_many(manifest_row["ids"])
            student_prompts = [row["student_prompt_ids"] for row in rows]
            teacher_prompts = [row[prompt_key] for row in rows]
            _synchronize(context.device)
            started = time.perf_counter()
            responses, response_mask = generate_responses(
                student,
                tokenizer,
                student_prompts,
                config,
                context.device,
                seed + 1_000_003 * policy_version + microbatch_index,
            )
            _synchronize(context.device)
            rollout_ms = (time.perf_counter() - started) * 1000.0
            generated_tokens = int(response_mask.sum().item())

            started = time.perf_counter()
            selected_mask = _entropy_selection(
                student,
                student_prompts,
                responses,
                response_mask,
                tokenizer.pad_token_id,
                context.device,
                policy_version,
                entropy_fraction,
                chunk_tokens,
            )
            _synchronize(context.device)
            entropy_ms = (time.perf_counter() - started) * 1000.0
            student_batch = collate_replay(
                student_prompts,
                responses,
                response_mask,
                tokenizer.pad_token_id,
                context.device,
                policy_version,
                selected_mask,
            )
            source_teacher_batch = collate_replay(
                teacher_prompts,
                responses,
                response_mask,
                tokenizer.pad_token_id,
                context.device,
                policy_version,
                selected_mask,
            )

        teacher_batch = broadcast_replay_batch(context, source_teacher_batch)
        if teacher_batch.policy_version != policy_version:
            raise RuntimeError(
                f"stale on-policy item: item={teacher_batch.policy_version}, current={policy_version}"
            )
        selected_tokens = teacher_batch.selected_tokens
        if selected_tokens <= 0:
            raise RuntimeError("microbatch selected zero response positions")
        _synchronize(context.device)
        phase_started = time.perf_counter()

        teacher_hidden = None
        if (not ema_method and context.is_teacher) or (ema_method and context.is_teacher_leader):
            with torch.inference_mode():
                teacher_hidden = selected_response_hidden(teacher, teacher_batch)

        student_hidden = None
        student_hidden_gradient = None
        if context.is_student:
            student_hidden = selected_response_hidden(student, student_batch)
            # Each KL chunk projects only a small slice through lm_head.  Detach
            # those slices below, accumulate their hidden-state gradients here,
            # and traverse the expensive Transformer backbone exactly once.
            student_hidden_gradient = torch.empty_like(student_hidden)

        total_loss = 0.0
        transfer_wait_ms = 0.0
        for start, end in token_chunks(selected_tokens, chunk_tokens):
            teacher_logits = None
            if (not ema_method and context.is_teacher) or (ema_method and context.is_teacher_leader):
                with torch.inference_mode():
                    teacher_logits = teacher.lm_head(teacher_hidden[start:end])
                if context.is_teacher_leader:
                    send_logits(teacher_logits)
            if context.is_student:
                projection_hidden = student_hidden[start:end].detach().requires_grad_(True)
                student_logits = student.lm_head(projection_hidden)
                receive_started = time.perf_counter()
                teacher_received = receive_logits(end - start, vocab_size, context.device)
                transfer_wait_ms += (time.perf_counter() - receive_started) * 1000.0
                per_token = exact_kl_per_token(
                    student_logits,
                    teacher_received,
                    divergence=divergence,
                    temperature=temperature,
                    temperature_squared_scale=temperature_scale,
                )
                chunk_loss = per_token.sum() / float(selected_tokens * group_size)
                chunk_loss.backward()
                if projection_hidden.grad is None:
                    raise RuntimeError("lm_head backward did not produce hidden-state gradients")
                student_hidden_gradient[start:end].copy_(projection_hidden.grad)
                total_loss += float(per_token.detach().sum().item())
                del projection_hidden, student_logits, teacher_received, per_token, chunk_loss
            if teacher_logits is not None:
                del teacher_logits

        if context.is_student:
            # This single call replaces one full-backbone traversal per KL
            # chunk. lm_head parameter gradients were already accumulated by
            # the chunk-local backward calls above.
            student_hidden.backward(student_hidden_gradient)

        _synchronize(context.device)
        worker_elapsed_ms = (time.perf_counter() - phase_started) * 1000.0
        worker_peak_gb = torch.cuda.max_memory_allocated(context.device) / (1024**3)
        _worker_stats(context, worker_elapsed_ms, worker_peak_gb)

        worker_stats: dict[str, float] = {}
        if context.is_student:
            worker_stats = _receive_worker_stats(context)
            step_loss = total_loss / float(selected_tokens)
            is_boundary = (
                microbatch_index + 1 == len(manifest)
                or int(manifest[microbatch_index + 1]["runtime_optimizer_step"]) != runtime_step
            )
            optimizer_ms = 0.0
            if is_boundary:
                _synchronize(context.device)
                optimizer_started = time.perf_counter()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    student.parameters(), float(config["training"]["max_grad_norm"])
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                _synchronize(context.device)
                optimizer_ms = (time.perf_counter() - optimizer_started) * 1000.0
                completed_optimizer_steps += 1
                policy_version += 1
            else:
                grad_norm = None

            row = {
                "run": run_name,
                "microbatch_index": microbatch_index,
                "optimizer_step": completed_optimizer_steps,
                "policy_version": policy_version,
                "batch_ids": manifest_row["ids"],
                "loss/kl": step_loss,
                "loss/divergence": divergence,
                "loss/temperature": temperature,
                "rollout/generated_tokens": generated_tokens,
                "train/selected_tokens": selected_tokens,
                "train/entropy_fraction": entropy_fraction,
                "train/kl_chunk_tokens": chunk_tokens,
                "train/backbone_backward_calls": 1,
                "time/rollout_ms": rollout_ms,
                "time/entropy_selection_ms": entropy_ms,
                "time/student_replay_backward_ms": worker_elapsed_ms,
                "time/logits_transfer_wait_ms": transfer_wait_ms,
                "time/optimizer_ms": optimizer_ms,
                "perf/generated_tokens_per_sec": generated_tokens / max(rollout_ms / 1000.0, 1e-9),
                "memory/student_peak_gb": torch.cuda.max_memory_allocated(context.device) / (1024**3),
                "optimizer/lr": scheduler.get_last_lr()[0],
                "optimizer/grad_norm": (
                    float(grad_norm.item()) if grad_norm is not None else None
                ),
                **worker_stats,
            }
            metrics.write(row)
        else:
            is_boundary = (
                microbatch_index + 1 == len(manifest)
                or int(manifest[microbatch_index + 1]["runtime_optimizer_step"]) != runtime_step
            )

        if is_boundary:
            if context.is_student:
                print(
                    f"[progress] run={run_name} microbatch={microbatch_index} "
                    f"optimizer_step={completed_optimizer_steps} phase=boundary_barrier",
                    flush=True,
                )
            synchronized_version = torch.tensor(
                [policy_version if context.is_student else -1],
                dtype=torch.long,
                device=context.device,
            )
            dist.broadcast(synchronized_version, src=0)
            policy_version = int(synchronized_version.item())
            if ema_method:
                if context.rank == 0:
                    update_ema_across_ranks(student, context, float(config["method"]["ema_decay"]))
                elif context.rank == 1:
                    update_ema_across_ranks(teacher, context, float(config["method"]["ema_decay"]))
            dist.barrier()
            if context.is_student:
                print(
                    f"[progress] run={run_name} microbatch={microbatch_index} "
                    f"optimizer_step={completed_optimizer_steps} phase=checkpoint",
                    flush=True,
                )
            if context.is_student and completed_optimizer_steps in milestone_steps:
                _checkpoint(
                    student,
                    tokenizer,
                    output_dir,
                    f"checkpoint-step-{completed_optimizer_steps:05d}",
                    run_name,
                    completed_optimizer_steps,
                    policy_version,
                )
            dist.barrier()
            if context.is_student:
                print(
                    f"[progress] run={run_name} microbatch={microbatch_index} "
                    f"optimizer_step={completed_optimizer_steps} phase=next_microbatch",
                    flush=True,
                )

        del teacher_batch, teacher_hidden, student_hidden, student_hidden_gradient
        if context.is_student:
            del student_batch, source_teacher_batch, responses, response_mask

    if context.is_student:
        _checkpoint(
            student,
            tokenizer,
            output_dir,
            "final_student",
            run_name,
            completed_optimizer_steps,
            policy_version,
        )
    dist.barrier()
