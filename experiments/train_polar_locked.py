"""CUDA final train+validation refits governed exclusively by a sealed selection lock."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from train_polar_benchmark import atomic_checkpoint, predict, seed_worker, set_seed, verify_images

from hac.polar import sha256_file
from hac.polar_benchmark import (
    atomic_json,
    canonical_hash,
    check_completed,
    environment_evidence,
    lock_json,
    utc_now,
)
from hac.polar_benchmark_features import _cache_lock
from hac.polar_locked_neural import (
    FINAL_FIXED,
    FINAL_MODELS,
    FINAL_STAGE,
    FinalFitImages,
    build_final_model,
    final_foundation,
    final_preprocessing,
    load_locked_development,
    parameter_evidence,
    read_fit_lock,
    replay_sample,
    validate_replay,
)
from hac.polar_training import optimizer_parameter_groups, warmup_cosine_scheduler


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--task", choices=("polar4", "polar9"), required=True)
    parser.add_argument("--model-kind", choices=FINAL_MODELS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 52, 62), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def final_request(args, lock, task_spec, fit, frame, model, foundation, processor, loading):
    return {
        **FINAL_FIXED,
        "stage": FINAL_STAGE,
        "role": "engineering_smoke" if args.smoke else FINAL_STAGE,
        "smoke": args.smoke,
        "task": args.task,
        "model_kind": args.model_kind,
        "seed": args.seed,
        "classes": task_spec["classes"],
        "class_names": task_spec["class_names"],
        "epochs": 1 if args.smoke else fit["epochs"],
        "locked_production_epochs": fit["epochs"],
        "epoch_selection": fit["epoch_selection"],
        "schedule_horizon_epochs": fit["schedule_horizon_epochs"],
        "selection": "none_fixed_locked_epoch_budget",
        "validation_selection": False,
        "early_stopping": False,
        "initialization": "original_foundation_only",
        "lock_path": str(args.lock.resolve()),
        "lock_sha256": sha256_file(args.lock),
        "selection_lock_sha256": sha256_file(args.lock),
        "development_manifest": task_spec["development_manifest"],
        "development_manifest_sha256": task_spec["development_manifest_sha256"],
        "train_rows": len(frame),
        "original_split_counts": {
            str(key): int(value) for key, value in frame.split.value_counts().items()
        },
        "development_row_order_sha256": canonical_hash(frame.image_id.astype(str).tolist()),
        "fit_scope": "development_smoke" if args.smoke else "all_original_train_plus_validation",
        "loss": "unweighted_cross_entropy",
        "label_smoothing": 0.0,
        "mixup": 0.0,
        "precision": "cuda_bfloat16",
        "workers": args.workers,
        "epoch_rng": "seed_plus_10000_times_epoch_reset; loader_seed_same",
        "gradient_accumulation": "mean_batch_loss_over_actual_accumulation_window",
        "foundation": foundation,
        "foundation_loading": loading,
        "preprocessing": final_preprocessing(args.model_kind, processor),
        "parameters": parameter_evidence(model, args.model_kind),
        "implementation": lock["implementation"],
        "environment": environment_evidence(),
        "test_rows_read": 0,
        "test_labels_read": False,
    }


def resume_integrity(output: Path, request: dict):
    last, sidecar = output / "last.pt", output / "last_checkpoint.json"
    if not last.exists():
        if sidecar.exists():
            raise RuntimeError("Final-fit resume sidecar exists without a checkpoint")
        return None
    if not sidecar.exists():
        raise RuntimeError("Final-fit resume checkpoint lacks integrity evidence")
    evidence = json.loads(sidecar.read_text(encoding="utf-8"))
    if (
        evidence.get("request_sha256") != canonical_hash(request)
        or evidence.get("lock_sha256") != request["lock_sha256"]
        or evidence.get("sha256") != sha256_file(last)
        or evidence.get("history_sha256") != sha256_file(output / "history.csv")
    ):
        raise RuntimeError("Final-fit resume checkpoint or history integrity mismatch")
    return evidence


def schedule_steps(rows: int, horizon_epochs: int = 20) -> int:
    if horizon_epochs != 20 or rows < 1:
        raise ValueError("The unchanged 20-epoch schedule horizon is mandatory")
    return (
        math.ceil(math.ceil(rows / FINAL_FIXED["batch_size"]) / FINAL_FIXED["accumulation"])
        * horizon_epochs
    )


def _run_locked(args, lock, task_spec, fit):
    frame = load_locked_development(task_spec, smoke=args.smoke)
    verify_images(frame)
    set_seed(args.seed)
    snapshot, foundation, processor = final_foundation(args.model_kind)
    model, loading = build_final_model(args.model_kind, task_spec["classes"], snapshot)
    request = final_request(
        args, lock, task_spec, fit, frame, model, foundation, processor, loading
    )
    output = args.output_dir.resolve()
    if previous := check_completed(output, request):
        print(f"Verified completed final refit: {output}", flush=True)
        return previous
    lock_json(output / "request.json", request)
    model = model.to("cuda")
    training_set = FinalFitImages(
        frame, model_kind=args.model_kind, processor=processor, training=True
    )
    sample = replay_sample(frame)
    replay_loader = DataLoader(
        FinalFitImages(sample, model_kind=args.model_kind, processor=processor, training=False),
        batch_size=FINAL_FIXED["batch_size"] * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            head_lr=FINAL_FIXED["head_lr"],
            backbone_lr=FINAL_FIXED["backbone_lr"],
            weight_decay=FINAL_FIXED["weight_decay"],
        )
    )
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=schedule_steps(len(frame), request["schedule_horizon_epochs"]),
        warmup_fraction=FINAL_FIXED["warmup_fraction"],
    )
    amp_dtype = torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    criterion = nn.CrossEntropyLoss()
    history, next_epoch = [], 1
    if resume_integrity(output, request):
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if (
            checkpoint["request_sha256"] != canonical_hash(request)
            or checkpoint["lock_sha256"] != request["lock_sha256"]
            or not 1 <= checkpoint["epoch"] <= request["epochs"]
        ):
            raise RuntimeError("Final-fit resume payload identity differs from its request")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        history, next_epoch = checkpoint["history"], checkpoint["epoch"] + 1
        del checkpoint
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    print(
        f"CUDA locked final refit {args.task}/{args.model_kind}/seed{args.seed}: {len(frame)} pooled development rows; fixed epochs={request['epochs']}; schedule horizon=20",
        flush=True,
    )
    for epoch in range(next_epoch, request["epochs"] + 1):
        epoch_started = time.perf_counter()
        epoch_seed = args.seed + 10000 * epoch
        set_seed(epoch_seed)
        loader = DataLoader(
            training_set,
            batch_size=FINAL_FIXED["batch_size"],
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(epoch_seed),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_rows = 0.0, 0
        for batch_index, (pixels, targets) in enumerate(loader):
            pixels, targets = (
                pixels.to("cuda", non_blocking=True),
                targets.to("cuda", non_blocking=True),
            )
            window_start = (batch_index // FINAL_FIXED["accumulation"]) * FINAL_FIXED[
                "accumulation"
            ]
            divisor = min(FINAL_FIXED["accumulation"], len(loader) - window_start)
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = criterion(model(pixels), targets)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    "Nonfinite final-fit loss; no automatic retry or model substitution"
                )
            scaler.scale(loss / divisor).backward()
            if (batch_index + 1) % FINAL_FIXED["accumulation"] == 0 or batch_index + 1 == len(
                loader
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), FINAL_FIXED["gradient_clip"], error_if_nonfinite=True
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            total_loss += float(loss.detach()) * len(targets)
            total_rows += len(targets)
            if batch_index % 50 == 0:
                elapsed = time.perf_counter() - epoch_started
                atomic_json(
                    output / "progress.json",
                    {
                        "status": "TRAINING",
                        "stage": FINAL_STAGE,
                        "epoch": epoch,
                        "epochs": request["epochs"],
                        "batch": batch_index + 1,
                        "batches": len(loader),
                        "examples_seen": total_rows,
                        "examples_per_second": total_rows / max(elapsed, 1e-6),
                        "epoch_eta_seconds": elapsed
                        * (len(loader) - batch_index - 1)
                        / (batch_index + 1),
                        "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                        "test_rows_read": 0,
                        "utc": utc_now(),
                    },
                )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "train_rows": total_rows,
            "epoch_seconds": time.perf_counter() - epoch_started,
            "learning_rates": json.dumps([group["lr"] for group in optimizer.param_groups]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        atomic_checkpoint(
            output / "last.pt",
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "request_sha256": canonical_hash(request),
                "lock_sha256": request["lock_sha256"],
                "epoch": epoch,
                "history": history,
            },
        )
        atomic_json(
            output / "last_checkpoint.json",
            {
                "sha256": sha256_file(output / "last.pt"),
                "request_sha256": canonical_hash(request),
                "lock_sha256": request["lock_sha256"],
                "history_sha256": sha256_file(output / "history.csv"),
            },
        )
        atomic_json(
            output / "progress.json",
            {
                "status": "EPOCH_COMPLETE",
                **row,
                "epochs": request["epochs"],
                "test_rows_read": 0,
                "utc": utc_now(),
            },
        )
        print(row, flush=True)
    if len(history) != request["epochs"] or [row["epoch"] for row in history] != list(
        range(1, request["epochs"] + 1)
    ):
        raise RuntimeError("Final refit did not execute its complete locked epoch sequence")
    resume_integrity(output, request)
    original_labels, original_probabilities = predict(model, replay_loader, amp_dtype)
    original_probabilities /= original_probabilities.sum(axis=1, keepdims=True)
    np.savez_compressed(
        output / "dev_replay_predictions.npz",
        image_ids=sample.image_id.to_numpy(dtype=str),
        labels=original_labels,
        probabilities=original_probabilities,
    )
    atomic_checkpoint(
        output / "final.pt",
        {
            "model": model.state_dict(),
            "request_sha256": canonical_hash(request),
            "lock_sha256": request["lock_sha256"],
            "epoch": request["epochs"],
            "model_kind": args.model_kind,
            "task": args.task,
            "seed": args.seed,
        },
    )
    atomic_json(
        output / "final_checkpoint.json",
        {
            "sha256": sha256_file(output / "final.pt"),
            "request_sha256": canonical_hash(request),
            "lock_sha256": request["lock_sha256"],
        },
    )
    final = torch.load(output / "final.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(final["model"], strict=True)
    del final
    labels, probabilities = predict(model, replay_loader, amp_dtype)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    audit = validate_replay(
        sample.image_id.to_numpy(dtype=str),
        labels,
        probabilities,
        output / "dev_replay_predictions.npz",
    )
    atomic_json(
        output / "replay_audit.json",
        {
            **audit,
            "checkpoint_sha256": sha256_file(output / "final.pt"),
            "sample_rows": len(sample),
        },
    )
    artifact_names = (
        "final.pt",
        "final_checkpoint.json",
        "request.json",
        "replay_audit.json",
        "dev_replay_predictions.npz",
        "history.csv",
        "last.pt",
        "last_checkpoint.json",
    )
    summary = {
        "status": "COMPLETE",
        "stage": FINAL_STAGE,
        "role": request["role"],
        "lock_sha256": request["lock_sha256"],
        "selection_lock_sha256": request["lock_sha256"],
        "request_sha256": canonical_hash(request),
        "model_kind": args.model_kind,
        "seed": args.seed,
        "task": args.task,
        "epochs_completed": len(history),
        "schedule_horizon_epochs": 20,
        "train_rows": len(frame),
        "test_rows_read": 0,
        "test_labels_read": False,
        "selection": "none_fixed_locked_epoch_budget",
        "elapsed_this_session_seconds": time.perf_counter() - started,
        "artifacts": {name: sha256_file(output / name) for name in artifact_names},
        "completed_utc": utc_now(),
    }
    atomic_json(output / "summary.json", summary)
    atomic_json(
        output / "progress.json",
        {
            "status": "COMPLETE",
            "stage": FINAL_STAGE,
            "epochs_completed": len(history),
            "train_rows": len(frame),
            "test_rows_read": 0,
            "utc": utc_now(),
        },
    )
    return summary


def run_training(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: refusing CPU final training")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "The locked CUDA bfloat16 precision is unavailable; no silent precision substitution"
        )
    if args.workers < 0:
        raise ValueError("Workers cannot be negative")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    lock, task_spec, fit = read_fit_lock(
        args.lock,
        task=args.task,
        model_kind=args.model_kind,
        seed=args.seed,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )
    if not args.smoke and args.workers != lock["neural_training"].get("workers"):
        raise ValueError("Production data-loader workers must match the final selection lock")
    with _cache_lock(args.output_dir.resolve()):
        return _run_locked(args, lock, task_spec, fit)


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
