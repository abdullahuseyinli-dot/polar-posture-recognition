"""CUDA-only, resumable nine/four-class DINOv2 adaptation from foundation weights.

This is a new development experiment. It cannot open a test manifest, initialize
from an old train+validation checkpoint, or overwrite any retained result.
"""

from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hac.augmentations import (
    build_aspect_preserving_eval_transform,
    build_eval_transform,
    build_person_train_transform,
    build_train_transform,
)
from hac.config import ModelConfig
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
from hac.polar_models import DINO_MODEL_SPECS, build_polar_model
from hac.polar_training import optimizer_parameter_groups, warmup_cosine_scheduler


class BenchmarkImages(Dataset):
    def __init__(self, frame, *, training, recipe):
        self.rows = frame.to_dict("records")
        if recipe == "historical_mild":
            self.transform = build_train_transform("mild") if training else build_eval_transform()
        elif recipe == "person_preserving":
            self.transform = (
                build_person_train_transform("person_safe_mild")
                if training
                else build_aspect_preserving_eval_transform()
            )
        else:
            raise ValueError("Unknown locked view/augmentation recipe")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["image_path"]) as image:
            pixels = self.transform(image_view(image.convert("RGB"), row, "person_context_25"))
        return pixels, int(row["label_index"])


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_checkpoint(path, value):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def foundation_evidence():
    specification = DINO_MODEL_SPECS["dinov2_base"]
    root = Path(
        snapshot_download(
            repo_id=specification["model_id"],
            revision=specification["revision"],
            allow_patterns=["config.json", "model.safetensors"],
            local_files_only=True,
        )
    )
    files = {
        path.name: sha256_file(path)
        for path in root.iterdir()
        if path.suffix in {".safetensors", ".bin", ".json"}
    }
    if "config.json" not in files or not any(
        name.endswith((".safetensors", ".bin")) for name in files
    ):
        raise RuntimeError("Pinned foundation checkpoint is incomplete")
    return {**specification, "files": files}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", type=int, choices=(4, 9), required=True)
    parser.add_argument("--recipe", choices=("historical_mild", "person_preserving"), required=True)
    parser.add_argument("--seed", type=int, choices=(42, 52, 62), default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def verify_images(frame):
    column = "image_sha256" if "image_sha256" in frame else "sha256"
    if column not in frame:
        raise ValueError("Training requires byte-audited image hashes")
    for row in frame.to_dict("records"):
        if sha256_file(row["image_path"]) != row[column]:
            raise RuntimeError(f"Image bytes drifted: {row['image_id']}")


@torch.inference_mode()
def predict(model, loader, amp_dtype):
    model.eval()
    values, labels = [], []
    for pixels, targets in loader:
        with torch.autocast("cuda", dtype=amp_dtype):
            logits = model(pixels.to("cuda", non_blocking=True))
        values.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(targets.numpy())
    return np.concatenate(labels), np.concatenate(values).astype(np.float64)


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required: refusing CPU training")
    if (
        min(args.batch_size, args.accumulation, args.max_epochs, args.patience) < 1
        or args.workers < 0
    ):
        raise ValueError("Invalid training budget")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    set_seed(args.seed)
    frame = load_development(args.manifest, num_classes=args.classes)
    if args.smoke:
        frame = (
            frame.groupby(["split", "label_index"], group_keys=False)
            .head(2)
            .sort_values("image_id", ignore_index=True)
        )
    names = label_names(frame)
    train = frame.loc[frame.split.eq("train")].copy()
    validation = frame.loc[frame.split.eq("val")].copy()
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    request = {
        "manifest_sha256": sha256_file(args.manifest),
        "classes": args.classes,
        "class_names": names,
        "role": "engineering_smoke" if args.smoke else "development_adaptation",
        "recipe": args.recipe,
        "seed": args.seed,
        "model_kind": "dinov2_base",
        "initialization": "original_foundation_only",
        "foundation": foundation_evidence(),
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows_read": 0,
        "top_blocks": 4,
        "head_lr": 0.001,
        "backbone_lr": 0.000005,
        "weight_decay": 0.0001,
        "dropout": 0.1,
        "loss": "unweighted_cross_entropy",
        "label_smoothing": 0.0,
        "mixup": 0.0,
        "gradient_clip": 1.0,
        "batch_size": args.batch_size,
        "accumulation": args.accumulation,
        "max_epochs": 1 if args.smoke else args.max_epochs,
        "patience": args.patience,
        "workers": args.workers,
        "selection": "validation_macro_f1_then_nll",
        "minimum_epochs": 3,
        "epoch_rng": "seed_plus_10000_times_epoch_reset; loader_seed_same",
        "environment": environment_evidence(),
        "implementation": implementation_evidence(root),
    }
    if previous := check_completed(output, request):
        print(f"Verified completed run: {output}; best epoch {previous['best_epoch']}", flush=True)
        return
    lock_json(output / "request.json", request)
    verify_images(frame)
    config = ModelConfig(
        model_kind="dinov2_base",
        augmentation_strength="mild",
        batch_size=args.batch_size,
        head_lr=0.001,
        backbone_lr=5e-6,
        weight_decay=1e-4,
        dropout=0.1,
        unfreeze_strategy="top_blocks",
        top_n_blocks=4,
    )
    model = build_polar_model(config, num_classes=args.classes, pretrained=True).to("cuda")
    train_set = BenchmarkImages(train, training=True, recipe=args.recipe)
    val_loader = DataLoader(
        BenchmarkImages(validation, training=False, recipe=args.recipe),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
    )
    optimizer = torch.optim.AdamW(
        optimizer_parameter_groups(model, head_lr=0.001, backbone_lr=5e-6, weight_decay=1e-4)
    )
    batches = math.ceil(len(train_set) / args.batch_size)
    max_epochs = request["max_epochs"]
    scheduler = warmup_cosine_scheduler(
        optimizer,
        total_steps=math.ceil(batches / args.accumulation) * max_epochs,
        warmup_fraction=0.10,
    )
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    criterion = nn.CrossEntropyLoss()
    best_epoch, best_f1, best_nll, bad_epochs, next_epoch, history = 0, -1.0, float("inf"), 0, 1, []
    last = output / "last.pt"
    checkpoint_integrity = output / "last_checkpoint.json"
    if last.exists():
        import json

        if not checkpoint_integrity.exists():
            raise RuntimeError("Resume checkpoint has no integrity sidecar")
        integrity = json.loads(checkpoint_integrity.read_text(encoding="utf-8"))
        if integrity.get("sha256") != sha256_file(last) or integrity.get(
            "request_sha256"
        ) != canonical_hash(request):
            raise RuntimeError("Resume checkpoint integrity mismatch")
        for name, digest in integrity.get("selected_artifacts", {}).items():
            if sha256_file(output / name) != digest:
                raise RuntimeError(f"Selected checkpoint artifact drift: {name}")
        if not integrity.get("selected_artifacts"):
            raise RuntimeError("Resume checkpoint lacks selected-model integrity evidence")
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if checkpoint["request_sha256"] != canonical_hash(request):
            raise RuntimeError("Resume request mismatch")
        model.load_state_dict(checkpoint["model"])
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
        f"CUDA training {args.classes} classes, {len(train)} train / {len(validation)} validation, {args.recipe}, seed {args.seed}",
        flush=True,
    )
    for epoch in range(next_epoch, max_epochs + 1):
        if bad_epochs >= args.patience and epoch > 3:
            break
        epoch_started = time.perf_counter()
        epoch_seed = args.seed + 10000 * epoch
        set_seed(epoch_seed)
        generator = torch.Generator().manual_seed(epoch_seed)
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_loss, total_rows = 0.0, 0
        for batch_index, (pixels, targets) in enumerate(train_loader):
            pixels, targets = (
                pixels.to("cuda", non_blocking=True),
                targets.to("cuda", non_blocking=True),
            )
            window_start = (batch_index // args.accumulation) * args.accumulation
            divisor = min(args.accumulation, len(train_loader) - window_start)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits = model(pixels)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; refusing continuation")
            scaler.scale(loss / divisor).backward()
            if (batch_index + 1) % args.accumulation == 0 or batch_index + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
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
                        "max_epochs": max_epochs,
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
            last,
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
            checkpoint_integrity,
            {
                "sha256": sha256_file(last),
                "request_sha256": canonical_hash(request),
                "selected_artifacts": {
                    name: sha256_file(output / name)
                    for name in (
                        "best.pt",
                        "validation_predictions.npz",
                        "best_metrics.json",
                        "history.csv",
                    )
                },
            },
        )
        atomic_json(output / "progress.json", {"status": "EPOCH_COMPLETE", **row, "utc": utc_now()})
        print(row, flush=True)
    if not (output / "best.pt").is_file():
        raise RuntimeError("No validated checkpoint produced")
    selected = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    if selected["request_sha256"] != canonical_hash(request) or selected["epoch"] != best_epoch:
        raise RuntimeError("Selected checkpoint does not belong to this locked run")
    model.load_state_dict(selected["model"])
    del selected
    replay_labels, replay_probabilities = predict(model, val_loader, amp_dtype)
    replay_probabilities /= replay_probabilities.sum(axis=1, keepdims=True)
    with np.load(output / "validation_predictions.npz", allow_pickle=False) as saved:
        delta = float(np.max(np.abs(replay_probabilities - saved["probabilities"])))
        label_match = bool(np.array_equal(replay_labels, saved["labels"]))
        prediction_match = bool(
            np.array_equal(
                replay_probabilities.argmax(axis=1), saved["probabilities"].argmax(axis=1)
            )
        )
    if not label_match or not prediction_match or delta > 1e-5:
        raise RuntimeError(
            "Selected checkpoint failed predeclared validation replay (1e-5 probability tolerance, exact labels/predictions)"
        )
    atomic_json(
        output / "replay_audit.json",
        {
            "status": "PASS",
            "maximum_probability_difference": delta,
            "tolerance": 1e-5,
            "labels_identical": label_match,
            "predictions_identical": prediction_match,
            "checkpoint_sha256": sha256_file(output / "best.pt"),
            "test_rows_read": 0,
        },
    )
    artifacts = {
        name: sha256_file(output / name)
        for name in (
            "best.pt",
            "validation_predictions.npz",
            "best_metrics.json",
            "history.csv",
            "last.pt",
            "last_checkpoint.json",
            "replay_audit.json",
        )
    }
    atomic_json(
        output / "summary.json",
        {
            "status": "COMPLETE",
            "role": request["role"],
            "request_sha256": canonical_hash(request),
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_f1,
            "epochs_completed": len(history),
            "elapsed_this_session_seconds": time.perf_counter() - started,
            "artifacts": artifacts,
            "test_rows_read": 0,
            "completed_utc": utc_now(),
        },
    )


if __name__ == "__main__":
    main()
