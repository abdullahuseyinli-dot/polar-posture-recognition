"""Immutable final-fit contracts and inference loading for locked POLAR models.

Training sees only the original development manifest. A separate evaluator owns
the post-fit test barrier; this module never opens a test manifest or chooses an
epoch, threshold, ensemble, or model from test outcomes.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset

from hac.augmentations import build_aspect_preserving_eval_transform, build_person_train_transform
from hac.models import configure_dinov2
from hac.polar import image_view, sha256_file
from hac.polar_benchmark import canonical_hash, check_completed, label_names, load_development
from hac.polar_bounded_models import (
    build_bounded_model,
    build_person_transform,
    preprocessing_evidence,
)
from hac.polar_bounded_models import foundation_evidence as bounded_foundation_evidence
from hac.polar_models import DINO_MODEL_SPECS, PinnedDinoV2Classifier

FINAL_MODELS = ("dinov2_base", "siglip2_base", "convnextv2_base")
FINAL_STAGE = "final_train_plus_validation_fit"
FINAL_FIXED = {
    "batch_size": 16,
    "accumulation": 4,
    "head_lr": 0.001,
    "backbone_lr": 5e-6,
    "weight_decay": 1e-4,
    "dropout": 0.1,
    "warmup_fraction": 0.1,
    "gradient_clip": 1.0,
    "schedule_horizon_epochs": 20,
    "replay_probability_tolerance": 1e-5,
}
DINO_FOUNDATION_HASHES = {
    "config.json": "f7ff4cfa73d2f70647dbf6950541ad25d73082d54c2e7e9bded160c7656b2a70",
    "model.safetensors": "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841",
    "preprocessor_config.json": "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
}


def required_source_paths() -> tuple[str, ...]:
    return (
        "src/hac/polar_locked_neural.py",
        "experiments/train_polar_locked.py",
        "src/hac/polar_bounded_models.py",
        "src/hac/polar_benchmark_features.py",
        "src/hac/polar_benchmark.py",
        "src/hac/polar_training.py",
        "src/hac/polar_models.py",
        "src/hac/models.py",
        "src/hac/config.py",
        "src/hac/augmentations.py",
        "src/hac/polar.py",
        "src/hac/polar_features.py",
        "src/hac/vcoco_v3_representations.py",
        "src/hac/metrics.py",
        "experiments/train_polar_benchmark.py",
    )


def verify_implementation(implementation: dict, root: Path) -> None:
    if not isinstance(implementation, dict) or not set(required_source_paths()).issubset(
        implementation
    ):
        raise ValueError("Final selection lock does not bind the complete neural implementation")
    root = root.resolve()
    for relative, digest in implementation.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Locked source implementation drift: {relative}")


def read_fit_lock(
    path: Path,
    *,
    task: str,
    model_kind: str,
    seed: int,
    output_dir: Path,
    smoke: bool = False,
    root: Path | None = None,
) -> tuple[dict, dict, dict]:
    lock = json.loads(path.read_text(encoding="utf-8"))
    if lock.get("status") != "POLAR_FINAL_EVALUATION_LOCKED":
        raise ValueError("Final fitting requires a sealed final-evaluation lock")
    if task not in {"polar4", "polar9"} or model_kind not in FINAL_MODELS:
        raise ValueError("Unknown locked task or neural model")
    if root is None:
        root = Path(__file__).resolve().parents[2]
    verify_implementation(lock.get("implementation"), root)
    task_spec = lock["tasks"][task]
    if task_spec["classes"] != int(task.removeprefix("polar")):
        raise ValueError("Locked task/class count mismatch")
    names = task_spec["class_names"]
    if (
        not all(isinstance(name, str) and name for name in names)
        or len(names) != task_spec["classes"]
        or len(set(names)) != len(names)
    ):
        raise ValueError("Locked class names must be complete and unique")
    manifest = Path(task_spec["development_manifest"])
    if not manifest.is_absolute():
        raise ValueError("Locked development manifest must be absolute")
    fit = task_spec["neural_fits"][model_kind]
    if fit.get("seeds") != [42, 52, 62] or seed not in fit["seeds"]:
        raise ValueError("Only the three locked seeds are admissible")
    epochs = fit.get("epochs")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
        raise ValueError("Locked final epochs must be an integer from one through twenty")
    if (
        fit.get("epoch_selection") != "median_of_three_development_best_epochs"
        or fit.get("schedule_horizon_epochs") != 20
    ):
        raise ValueError(
            "Final refits require the locked development median and unchanged 20-epoch schedule horizon"
        )
    training = lock.get("neural_training", {})
    expected_training = {
        key: FINAL_FIXED[key]
        for key in (
            "batch_size",
            "accumulation",
            "head_lr",
            "backbone_lr",
            "weight_decay",
            "dropout",
            "gradient_clip",
            "warmup_fraction",
            "schedule_horizon_epochs",
        )
    }
    expected_training.update(
        {
            "loss": "cross_entropy",
            "early_stopping": False,
            "precision": "cuda_bfloat16",
            "view": "person_context_25",
            "augmentation": "person_safe_mild",
        }
    )
    if any(training.get(key) != value for key, value in expected_training.items()):
        raise ValueError("Final-training settings differ from the unchanged development recipe")
    output_root = Path(lock.get("output_root", path.resolve().parent))
    if not output_root.is_absolute() or output_root.resolve() != path.resolve().parent:
        raise ValueError("Final output root must be the absolute selection-lock parent")
    pattern = fit["output_dir_pattern"].format(task=task, model_kind=model_kind, seed=seed)
    if Path(pattern).is_absolute():
        raise ValueError("Final output pattern must be relative to the locked output root")
    expected = (output_root / pattern).resolve()
    if not expected.is_relative_to(output_root.resolve()):
        raise ValueError("Final output pattern escapes its locked output root")
    actual = output_dir.resolve()
    if smoke:
        if (
            not actual.is_relative_to(output_root.resolve())
            or actual == expected
            or not any(
                "smoke" in part.lower() for part in actual.relative_to(output_root.resolve()).parts
            )
        ):
            raise ValueError(
                "Engineering smoke must use a separate smoke directory under the locked root"
            )
    elif actual != expected:
        raise ValueError("Final production output differs from its immutable locked destination")
    return lock, task_spec, fit


def load_locked_development(task_spec: dict, *, smoke: bool = False) -> pd.DataFrame:
    manifest = Path(task_spec["development_manifest"])
    splits = pd.read_csv(manifest, usecols=["split"], dtype=str, keep_default_na=False)["split"]
    if set(splits) != {"train", "val"}:
        raise ValueError("Final fitting rejects test rows before parsing any labels")
    if sha256_file(manifest) != task_spec["development_manifest_sha256"]:
        raise RuntimeError("Locked development manifest bytes changed")
    frame = load_development(manifest, num_classes=task_spec["classes"])
    if len(frame) != task_spec["development_rows"]:
        raise RuntimeError("Development row count differs from the final selection lock")
    if label_names(frame) != task_spec["class_names"]:
        raise RuntimeError("Development class mapping differs from the final selection lock")
    boxes = ["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]
    if not set(boxes).issubset(frame):
        raise ValueError("Final person-preserving fitting requires audited boxes")
    coordinates = frame[boxes].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all() or (coordinates[:, 2:] <= coordinates[:, :2]).any():
        raise ValueError("Invalid final-fit person boxes")
    if "image_sha256" not in frame and "sha256" not in frame:
        raise ValueError("Final fitting requires audited source image hashes")
    if (
        "image_sha256" in frame
        and "sha256" in frame
        and not frame.image_sha256.eq(frame.sha256).all()
    ):
        raise ValueError("Image-hash aliases disagree")
    if smoke:
        if (frame.groupby(["split", "label_index"]).size() < 2).any():
            raise ValueError("Smoke requires two examples per class and original split")
        frame = (
            frame.groupby(["split", "label_index"], group_keys=False)
            .head(2)
            .sort_values("image_id", ignore_index=True)
        )
    return frame


def replay_sample(frame: pd.DataFrame) -> pd.DataFrame:
    if set(frame.split) - {"train", "val"}:
        raise ValueError("Checkpoint replay must use development examples only")
    sample = (
        frame.sort_values("image_id")
        .groupby("label_index", group_keys=False)
        .head(2)
        .sort_values("image_id", ignore_index=True)
    )
    if not sample.groupby("label_index").size().eq(2).all():
        raise ValueError("Replay requires exactly two development examples per class")
    return sample


class LockedDinoClassifier(PinnedDinoV2Classifier):
    """Historical DINO classifier forward with injected, strictly verified foundation."""

    def __init__(self, backbone: nn.Module, num_classes: int):
        nn.Module.__init__(self)
        self.backbone = backbone
        self.backbone_name = DINO_MODEL_SPECS["dinov2_base"]["model_id"]
        self.backbone_revision = DINO_MODEL_SPECS["dinov2_base"]["revision"]
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(int(backbone.config.hidden_size), num_classes)
        configure_dinov2(self, "top_blocks", 4)


def final_foundation(model_kind: str) -> tuple[Path, dict, dict]:
    if model_kind != "dinov2_base":
        return bounded_foundation_evidence(model_kind)
    from huggingface_hub import snapshot_download

    specification = DINO_MODEL_SPECS[model_kind]
    snapshot = Path(
        snapshot_download(
            repo_id=specification["model_id"],
            revision=specification["revision"],
            local_files_only=True,
            allow_patterns=list(DINO_FOUNDATION_HASHES),
        )
    )
    files = {}
    for filename, digest in DINO_FOUNDATION_HASHES.items():
        source = snapshot / filename
        if not source.is_file() or sha256_file(source) != digest:
            raise RuntimeError(f"Original DINO foundation changed: {filename}")
        files[filename] = {"sha256": digest, "bytes": source.stat().st_size}
    return (
        snapshot,
        {**specification, "files": files},
        json.loads((snapshot / "preprocessor_config.json").read_text(encoding="utf-8")),
    )


def build_final_model(model_kind: str, num_classes: int, snapshot: Path) -> tuple[nn.Module, dict]:
    if model_kind != "dinov2_base":
        return build_bounded_model(model_kind, num_classes, snapshot)
    from transformers import AutoModel

    backbone, loading = AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    if any(
        loading.get(key)
        for key in ("missing_keys", "mismatched_keys", "unexpected_keys", "error_msgs")
    ):
        raise RuntimeError("DINO foundation did not load exactly; no hidden-weight substitutions")
    return LockedDinoClassifier(backbone, num_classes), {
        "missing_keys": [],
        "mismatched_keys": [],
        "unexpected_nonvision_keys": [],
    }


def final_transform(model_kind: str, processor: dict, *, training: bool):
    if model_kind == "dinov2_base":
        return (
            build_person_train_transform("person_safe_mild", 224)
            if training
            else build_aspect_preserving_eval_transform(224)
        )
    if model_kind not in FINAL_MODELS:
        raise ValueError("Unknown final-fit model")
    return build_person_transform(processor, training=training)


def final_preprocessing(model_kind: str, processor: dict) -> dict:
    if model_kind == "dinov2_base":
        return {
            "view": "person_context_25",
            "image_size": 224,
            "recipe": "historical_phase1_person_preserving",
            "training": "build_person_train_transform(person_safe_mild,224)",
            "evaluation": "build_aspect_preserving_eval_transform(224)",
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "padding_and_affine_fill": [124, 116, 104],
        }
    return preprocessing_evidence(processor)


def parameter_evidence(model: nn.Module, model_kind: str) -> dict:
    if hasattr(model, "parameter_evidence"):
        return model.parameter_evidence()
    trainable = {
        name: parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    total = sum(parameter.numel() for parameter in model.parameters())
    return {
        "scope": "historical_top_four_blocks_and_all_backbone_norms",
        "model_kind": model_kind,
        "trainable_names_and_counts": trainable,
        "trainable_parameters": sum(trainable.values()),
        "frozen_parameters": total - sum(trainable.values()),
        "total_parameters": total,
        "training_mode": "unchanged_historical_dino_classifier",
    }


class FinalFitImages(Dataset):
    def __init__(self, frame: pd.DataFrame, *, model_kind: str, processor: dict, training: bool):
        if set(frame.split) - {"train", "val"}:
            raise ValueError("Final-fit and replay datasets never accept test rows")
        self.rows = frame.to_dict("records")
        self.transform = final_transform(model_kind, processor, training=training)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        encoded = Path(row["image_path"]).read_bytes()
        if hashlib.sha256(encoded).hexdigest() != row.get("image_sha256", row.get("sha256")):
            raise RuntimeError("Locked development image bytes changed")
        with Image.open(io.BytesIO(encoded)) as image:
            pixels = self.transform(image_view(image.convert("RGB"), row, "person_context_25"))
        return pixels, int(row["label_index"])


def validate_replay(image_ids, labels, probabilities, saved_path: Path) -> dict:
    with np.load(saved_path, allow_pickle=False) as saved:
        if (
            probabilities.shape != saved["probabilities"].shape
            or not np.isfinite(probabilities).all()
        ):
            raise RuntimeError("Final checkpoint replay returned invalid probabilities")
        delta = float(np.max(np.abs(probabilities - saved["probabilities"])))
        identities = np.array_equal(np.asarray(image_ids, dtype=str), saved["image_ids"])
        labels_match = np.array_equal(labels, saved["labels"])
        predictions_match = np.array_equal(
            probabilities.argmax(axis=1), saved["probabilities"].argmax(axis=1)
        )
    if (
        not identities
        or not labels_match
        or not predictions_match
        or delta > FINAL_FIXED["replay_probability_tolerance"]
    ):
        raise RuntimeError("Fixed final checkpoint failed development-only replay")
    return {
        "status": "PASS",
        "maximum_probability_difference": delta,
        "tolerance": FINAL_FIXED["replay_probability_tolerance"],
        "image_ids_identical": bool(identities),
        "labels_identical": bool(labels_match),
        "predictions_identical": bool(predictions_match),
        "sample_selection": "first_two_per_class_by_image_id_from_development_only",
        "test_rows_read": 0,
    }


def validate_final_run(run_dir: Path, *, expected_lock_sha256: str) -> tuple[dict, dict]:
    request_path = run_dir / "request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if (
        request.get("lock_sha256") != expected_lock_sha256
        or request.get("selection_lock_sha256") != expected_lock_sha256
        or request.get("role") != FINAL_STAGE
        or request.get("smoke") is not False
    ):
        raise RuntimeError("Only an exact locked production final refit is admissible")
    if (
        request.get("test_rows_read") != 0
        or request.get("selection") != "none_fixed_locked_epoch_budget"
    ):
        raise RuntimeError("Final-fit request violates the no-test/no-selection contract")
    verify_implementation(request["implementation"], Path(__file__).resolve().parents[2])
    summary = check_completed(run_dir, request)
    if (
        summary is None
        or summary.get("stage") != FINAL_STAGE
        or summary.get("role") != FINAL_STAGE
        or summary.get("lock_sha256") != expected_lock_sha256
        or summary.get("selection_lock_sha256") != expected_lock_sha256
        or summary.get("epochs_completed") != request["epochs"]
    ):
        raise RuntimeError("The final refit is incomplete or violates its locked epoch budget")
    if any(
        summary.get(key) != request.get(key)
        for key in (
            "task",
            "model_kind",
            "seed",
            "train_rows",
            "test_rows_read",
            "test_labels_read",
            "selection",
            "schedule_horizon_epochs",
        )
    ):
        raise RuntimeError("Final completion metadata differs from the immutable request")
    required = {
        "final.pt",
        "final_checkpoint.json",
        "request.json",
        "replay_audit.json",
        "dev_replay_predictions.npz",
        "history.csv",
        "last.pt",
        "last_checkpoint.json",
    }
    if not required.issubset(summary.get("artifacts", {})):
        raise RuntimeError("Final completion evidence is incomplete")
    replay = json.loads((run_dir / "replay_audit.json").read_text(encoding="utf-8"))
    if (
        replay.get("status") != "PASS"
        or replay.get("test_rows_read") != 0
        or replay.get("predictions_identical") is not True
        or replay.get("checkpoint_sha256") != sha256_file(run_dir / "final.pt")
    ):
        raise RuntimeError("Final checkpoint replay was not passed")
    sidecar = json.loads((run_dir / "final_checkpoint.json").read_text(encoding="utf-8"))
    if (
        sidecar.get("sha256") != sha256_file(run_dir / "final.pt")
        or sidecar.get("request_sha256") != canonical_hash(request)
        or sidecar.get("lock_sha256") != expected_lock_sha256
    ):
        raise RuntimeError("Final checkpoint integrity differs from its request")
    return request, summary


def load_final_model(
    run_dir: Path, *, expected_lock_sha256: str, device: str = "cuda"
) -> tuple[nn.Module, object, dict]:
    """Post-barrier evaluator loader. It does not read any image or test label."""
    request, _ = validate_final_run(Path(run_dir), expected_lock_sha256=expected_lock_sha256)
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA inference requested but unavailable; no CPU fallback")
    snapshot, foundation, processor = final_foundation(request["model_kind"])
    if (
        foundation != request["foundation"]
        or final_preprocessing(request["model_kind"], processor) != request["preprocessing"]
    ):
        raise RuntimeError("Final inference foundation or preprocessing differs from training")
    model, loading = build_final_model(request["model_kind"], request["classes"], snapshot)
    if (
        loading != request["foundation_loading"]
        or parameter_evidence(model, request["model_kind"]) != request["parameters"]
    ):
        raise RuntimeError("Final inference loading contract changed")
    checkpoint = torch.load(Path(run_dir) / "final.pt", map_location="cpu", weights_only=True)
    if (
        checkpoint["request_sha256"] != canonical_hash(request)
        or checkpoint["lock_sha256"] != expected_lock_sha256
        or checkpoint["epoch"] != request["epochs"]
        or any(checkpoint.get(key) != request[key] for key in ("task", "model_kind", "seed"))
    ):
        raise RuntimeError("Final checkpoint payload identity mismatch")
    model.load_state_dict(checkpoint["model"], strict=True)
    return (
        model.to(target).eval(),
        final_transform(request["model_kind"], processor, training=False),
        request,
    )
