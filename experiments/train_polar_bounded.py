"""Fixed CUDA partial adaptation: two modern foundations, no test access or sweeps."""

from __future__ import annotations

import argparse
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from train_polar_benchmark import atomic_checkpoint, predict, seed_worker, set_seed, verify_images

from hac.polar import image_view, sha256_file
from hac.polar_benchmark import (
    atomic_json,
    canonical_hash,
    check_completed,
    environment_evidence,
    implementation_evidence,
    label_names,
    load_development,
    lock_json,
    probability_metrics,
    utc_now,
)
from hac.polar_bounded_models import (
    BOUNDED_MODELS,
    build_bounded_model,
    build_person_transform,
    foundation_evidence,
    preprocessing_evidence,
)
from hac.polar_training import optimizer_parameter_groups, warmup_cosine_scheduler

FIXED = {
    "head_lr": 0.001,
    "backbone_lr": 5e-6,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "batch_size": 16,
    "accumulation": 4,
    "max_epochs": 20,
    "patience": 4,
    "minimum_epochs": 3,
    "warmup_fraction": 0.1,
    "gradient_clip": 1.0,
    "replay_probability_tolerance": 1e-5,
}
SELECTED_ARTIFACTS = ("best.pt", "validation_predictions.npz", "best_metrics.json", "history.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", type=int, choices=(4, 9), required=True)
    parser.add_argument("--model-kind", choices=BOUNDED_MODELS, required=True)
    parser.add_argument("--seed", type=int, choices=(42, 52, 62), default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def load_training_manifest(path: Path, num_classes: int, *, smoke: bool) -> pd.DataFrame:
    splits = pd.read_csv(path, usecols=["split"], dtype=str, keep_default_na=False)["split"]
    if set(splits) != {"train", "val"}:
        raise ValueError("Only train/val accepted; test rejected before parsing labels")
    frame = load_development(path, num_classes=num_classes)
    bbox = ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
    if not set(bbox).issubset(frame):
        raise ValueError("Person-preserving training requires audited bounding boxes")
    values = frame[bbox].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values[:, 2:] <= values[:, :2]).any():
        raise ValueError("Invalid or nonfinite person boxes")
    if (
        "image_sha256" in frame
        and "sha256" in frame
        and not frame.image_sha256.eq(frame.sha256).all()
    ):
        raise ValueError("Manifest image-byte hash aliases disagree")
    if smoke:
        counts = frame.groupby(["split", "label_index"]).size()
        if (counts < 2).any():
            raise ValueError("Engineering smoke requires two examples per class and split")
        frame = (
            frame.groupby(["split", "label_index"], group_keys=False)
            .head(2)
            .sort_values("image_id", ignore_index=True)
        )
    return frame


class BoundedImages(Dataset):
    def __init__(self, frame: pd.DataFrame, *, training: bool, processor: dict):
        self.rows = frame.to_dict("records")
        self.transform = build_person_transform(processor, training=training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        import hashlib

        row = self.rows[index]
        encoded = Path(row["image_path"]).read_bytes()
        expected = row.get("image_sha256", row.get("sha256"))
        if not expected or hashlib.sha256(encoded).hexdigest() != expected:
            raise RuntimeError("Audited training image bytes changed")
        with Image.open(io.BytesIO(encoded)) as image:
            pixels = self.transform(image_view(image.convert("RGB"), row, "person_context_25"))
        return pixels, int(row["label_index"])


def request_evidence(args, frame, foundation, processor, model, loading):
    root = Path(__file__).resolve().parents[1]
    implementation = implementation_evidence(root)
    implementation["experiments/train_polar_bounded.py"] = sha256_file(Path(__file__))
    return {
        **FIXED,
        "max_epochs": 1 if args.smoke else FIXED["max_epochs"],
        "manifest_sha256": sha256_file(args.manifest),
        "classes": args.classes,
        "class_names": label_names(frame),
        "role": "engineering_smoke" if args.smoke else "development_adaptation",
        "recipe": "person_preserving_model_normalization",
        "seed": args.seed,
        "model_kind": args.model_kind,
        "initialization": "original_foundation_only",
        "foundation": foundation,
        "foundation_loading": loading,
        "preprocessing": preprocessing_evidence(processor),
        "parameters": model.parameter_evidence(),
        "train_rows": int(frame.split.eq("train").sum()),
        "validation_rows": int(frame.split.eq("val").sum()),
        "loss": "unweighted_cross_entropy",
        "label_smoothing": 0.0,
        "mixup": 0.0,
        "workers": args.workers,
        "selection": "validation_macro_f1_then_nll",
        "epoch_rng": "seed_plus_10000_times_epoch_reset; loader_seed_same",
        "gradient_accumulation": "mean_batch_loss_over_actual_accumulation_window",
        "test_rows_read": 0,
        "test_labels_read": False,
        "environment": environment_evidence(),
        "implementation": implementation,
    }


def validate_resume_integrity(output: Path, request: dict) -> dict | None:
    last, sidecar = output / "last.pt", output / "last_checkpoint.json"
    if not last.exists():
        if sidecar.exists():
            raise RuntimeError("Resume integrity sidecar exists without its checkpoint")
        return None
    if not sidecar.exists():
        raise RuntimeError("Resume checkpoint has no integrity sidecar")
    integrity = json.loads(sidecar.read_text(encoding="utf-8"))
    if integrity.get("sha256") != sha256_file(last) or integrity.get(
        "request_sha256"
    ) != canonical_hash(request):
        raise RuntimeError("Resume checkpoint integrity mismatch")
    selected = integrity.get("selected_artifacts", {})
    if set(selected) != set(SELECTED_ARTIFACTS):
        raise RuntimeError("Resume selected-artifact inventory differs from the fixed contract")
    for name, digest in selected.items():
        if sha256_file(output / name) != digest:
            raise RuntimeError(f"Selected checkpoint artifact drift: {name}")
    return integrity


def check_replay(saved_path, image_ids, labels, probabilities):
    with np.load(saved_path, allow_pickle=False) as saved:
        if (
            probabilities.shape != saved["probabilities"].shape
            or not np.isfinite(probabilities).all()
        ):
            raise RuntimeError("Selected checkpoint replay has invalid probability dimensions")
        delta = float(np.max(np.abs(probabilities - saved["probabilities"])))
        identity = np.array_equal(np.asarray(image_ids, dtype=str), saved["image_ids"])
        label_match = np.array_equal(labels, saved["labels"])
        prediction_match = np.array_equal(
            probabilities.argmax(axis=1), saved["probabilities"].argmax(axis=1)
        )
    if (
        not identity
        or not label_match
        or not prediction_match
        or delta > FIXED["replay_probability_tolerance"]
    ):
        raise RuntimeError("Selected checkpoint failed fixed validation replay")
    return {
        "status": "PASS",
        "maximum_probability_difference": delta,
        "tolerance": FIXED["replay_probability_tolerance"],
        "image_ids_identical": bool(identity),
        "labels_identical": bool(label_match),
        "predictions_identical": bool(prediction_match),
        "test_rows_read": 0,
    }


def run_training(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: refusing CPU training")
    if args.workers < 0:
        raise ValueError("Workers cannot be negative")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    frame = load_training_manifest(args.manifest, args.classes, smoke=args.smoke)
    verify_images(frame)
    snapshot, foundation, processor = foundation_evidence(args.model_kind)
    model, loading = build_bounded_model(args.model_kind, args.classes, snapshot)
    request = request_evidence(args, frame, foundation, processor, model, loading)
    output = args.output_dir.resolve()
    if previous := check_completed(output, request):
        print(
            f"Verified completed bounded run: {output}; best epoch {previous['best_epoch']}",
            flush=True,
        )
        return previous
    lock_json(output / "request.json", request)
    model = model.to("cuda")
    train = frame.loc[frame.split.eq("train")].copy()
    validation = frame.loc[frame.split.eq("val")].copy()
    names = request["class_names"]
    train_set = BoundedImages(train, training=True, processor=processor)
    val_loader = DataLoader(
        BoundedImages(validation, training=False, processor=processor),
        batch_size=FIXED["batch_size"] * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(
            model,
            head_lr=FIXED["head_lr"],
            backbone_lr=FIXED["backbone_lr"],
            weight_decay=FIXED["weight_decay"],
        )
    )
    batches = math.ceil(len(train_set) / FIXED["batch_size"])
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=math.ceil(batches / FIXED["accumulation"]) * request["max_epochs"],
        warmup_fraction=FIXED["warmup_fraction"],
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    criterion = nn.CrossEntropyLoss()
    best_epoch, best_f1, best_nll, bad_epochs, next_epoch, history = 0, -1.0, float("inf"), 0, 1, []
    if validate_resume_integrity(output, request):
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["request_sha256"] != canonical_hash(request):
            raise RuntimeError("Resume request mismatch")
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        best_epoch, best_f1, best_nll = (
            checkpoint["best_epoch"],
            checkpoint["best_f1"],
            checkpoint["best_nll"],
        )
        bad_epochs, next_epoch, history = (
            checkpoint["bad_epochs"],
            checkpoint["epoch"] + 1,
            checkpoint["history"],
        )
        del checkpoint
    started = time.perf_counter()
    print(
        f"CUDA bounded adaptation {args.model_kind}; classes={args.classes}; seed={args.seed}; train={len(train)}; val={len(validation)}",
        flush=True,
    )
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(next_epoch, request["max_epochs"] + 1):
        if bad_epochs >= FIXED["patience"] and epoch > FIXED["minimum_epochs"]:
            break
        epoch_started = time.perf_counter()
        epoch_seed = args.seed + 10000 * epoch
        set_seed(epoch_seed)
        train_loader = DataLoader(
            train_set,
            batch_size=FIXED["batch_size"],
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(epoch_seed),
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_rows = 0.0, 0
        for batch_index, (pixels, targets) in enumerate(train_loader):
            pixels, targets = (
                pixels.to("cuda", non_blocking=True),
                targets.to("cuda", non_blocking=True),
            )
            window_start = (batch_index // FIXED["accumulation"]) * FIXED["accumulation"]
            divisor = min(FIXED["accumulation"], len(train_loader) - window_start)
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = criterion(model(pixels), targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; refusing continuation")
            scaler.scale(loss / divisor).backward()
            if (batch_index + 1) % FIXED["accumulation"] == 0 or batch_index + 1 == len(
                train_loader
            ):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), FIXED["gradient_clip"], error_if_nonfinite=True
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
                        "epoch": epoch,
                        "max_epochs": request["max_epochs"],
                        "batch": batch_index + 1,
                        "batches": len(train_loader),
                        "examples_seen": total_rows,
                        "examples_per_second": total_rows / max(elapsed, 1e-6),
                        "epoch_eta_seconds": elapsed
                        * (len(train_loader) - batch_index - 1)
                        / (batch_index + 1),
                        "cuda_peak_bytes": torch.cuda.max_memory_allocated(),
                        "utc": utc_now(),
                    },
                )
        labels, probabilities = predict(model, val_loader, amp_dtype)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        metrics = probability_metrics(labels, probabilities, names)
        improved = metrics["macro_f1"] > best_f1 + 1e-12 or (
            abs(metrics["macro_f1"] - best_f1) <= 1e-12 and metrics["log_loss"] < best_nll
        )
        if improved:
            best_epoch, best_f1, best_nll, bad_epochs = (
                epoch,
                metrics["macro_f1"],
                metrics["log_loss"],
                0,
            )
            atomic_checkpoint(
                output / "best.pt",
                {
                    "model": model.state_dict(),
                    "request_sha256": canonical_hash(request),
                    "epoch": epoch,
                    "metrics": metrics,
                },
            )
            np.savez_compressed(
                output / "validation_predictions.npz",
                image_ids=validation.image_id.to_numpy(dtype=str),
                labels=labels,
                probabilities=probabilities,
            )
            atomic_json(output / "best_metrics.json", metrics)
        else:
            bad_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total_rows,
            "validation_macro_f1": metrics["macro_f1"],
            "validation_nll": metrics["log_loss"],
            "epoch_seconds": time.perf_counter() - epoch_started,
            "best_epoch": best_epoch,
            "bad_epochs": bad_epochs,
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
                "epoch": epoch,
                "best_epoch": best_epoch,
                "best_f1": best_f1,
                "best_nll": best_nll,
                "bad_epochs": bad_epochs,
                "history": history,
            },
        )
        atomic_json(
            output / "last_checkpoint.json",
            {
                "sha256": sha256_file(output / "last.pt"),
                "request_sha256": canonical_hash(request),
                "selected_artifacts": {
                    name: sha256_file(output / name) for name in SELECTED_ARTIFACTS
                },
            },
        )
        atomic_json(output / "progress.json", {"status": "EPOCH_COMPLETE", **row, "utc": utc_now()})
        print(row, flush=True)
    if not (output / "best.pt").is_file():
        raise RuntimeError("No validated checkpoint produced")
    validate_resume_integrity(output, request)
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    if selected["request_sha256"] != canonical_hash(request) or selected["epoch"] != best_epoch:
        raise RuntimeError("Selected checkpoint does not belong to this locked run")
    model.load_state_dict(selected["model"], strict=True)
    del selected
    replay_labels, replay_probabilities = predict(model, val_loader, amp_dtype)
    replay_probabilities /= replay_probabilities.sum(axis=1, keepdims=True)
    audit = check_replay(
        output / "validation_predictions.npz",
        validation.image_id.to_numpy(dtype=str),
        replay_labels,
        replay_probabilities,
    )
    atomic_json(
        output / "replay_audit.json",
        {**audit, "checkpoint_sha256": sha256_file(output / "best.pt")},
    )
    artifacts = {
        name: sha256_file(output / name)
        for name in (*SELECTED_ARTIFACTS, "last.pt", "last_checkpoint.json", "replay_audit.json")
    }
    summary = {
        "status": "COMPLETE",
        "role": request["role"],
        "model_kind": args.model_kind,
        "request_sha256": canonical_hash(request),
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "epochs_completed": len(history),
        "elapsed_this_session_seconds": time.perf_counter() - started,
        "artifacts": artifacts,
        "test_rows_read": 0,
        "completed_utc": utc_now(),
    }
    atomic_json(output / "summary.json", summary)
    return summary


def main():
    run_training(parse_args())


if __name__ == "__main__":
    main()
