"""Locked final RBF refits and label-free held-out feature extraction.

Development-only helpers remain restrictive and untouched.  Held-out pixels are
accessible here only through the independently verified all-fits/test-access
barrier; no held-out labels are accepted by this module.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader

from hac.polar import sha256_file
from hac.polar_benchmark import (
    aligned_features,
    atomic_json,
    canonical_hash,
    environment_evidence,
    label_names,
    load_development,
    lock_json,
    utc_now,
)
from hac.polar_benchmark_features import (
    BBOX_COLUMNS,
    MODEL_SPECS,
    BenchmarkFeatureDataset,
    FeatureRequest,
    _cache_lock,
    _resume_chunks,
    _save_chunk,
    _verify_artifacts,
    _verify_source_rows,
    prepare_model_and_transform,
    require_cuda,
)
from hac.polar_benchmark_features import (
    CACHE_STATUS as DEVELOPMENT_CACHE_STATUS,
)
from hac.polar_benchmark_heads import fit_group_calibrated_rbf

LOCKED_CACHE_STATUS = "POLAR_LOCKED_TEST_FEATURE_CACHE_COMPLETE"
HEAD_MODELS = ("dinov2_base", "siglip2_base", "convnextv2_base", "dinov3_base")
HEAD_VIEWS = ("full_frame", "person_context_10")
HEAD_STAGE = "final_train_plus_validation_fit"
INFERENCE_COLUMNS = {
    "image_id",
    "image_path",
    "split",
    "image_sha256",
    *BBOX_COLUMNS,
}


def required_source_paths(root: Path) -> list[Path]:
    """Implementation files that a final-selection lock must bind."""
    names = (
        "src/hac/polar_locked_features.py",
        "src/hac/polar_locked_evaluation.py",
        "src/hac/polar_benchmark.py",
        "src/hac/polar_benchmark_heads.py",
        "src/hac/polar_benchmark_features.py",
        "src/hac/polar_features.py",
        "src/hac/polar_models.py",
        "src/hac/vcoco_v3_representations.py",
        "src/hac/polar.py",
        "src/hac/augmentations.py",
        "experiments/fit_polar_locked_head.py",
        "experiments/cache_polar_locked_test_features.py",
    )
    return [root / name for name in names]


def _resolve(root: Path, value: str) -> Path:
    return (root / value).resolve()


def _read_selection_lock(path: Path) -> dict:
    # Imported lazily to keep the CPU unit tests independent of orchestration.
    from hac.polar_locked_evaluation import verify_selection_lock

    return verify_selection_lock(path)


def locked_head_spec(
    selection_lock: Path, task: str, model_kind: str, output_dir: Path
) -> tuple[dict, dict, dict]:
    selection_lock = selection_lock.resolve()
    lock = _read_selection_lock(selection_lock)
    root = Path(lock["output_root"]).resolve()
    if root != selection_lock.parent:
        raise ValueError("Selection lock must reside in its declared output root")
    if task not in ("polar4", "polar9") or model_kind not in HEAD_MODELS:
        raise ValueError("Unknown locked final-head task or backbone")
    task_spec = lock["tasks"][task]
    if task_spec["classes"] != int(task.removeprefix("polar")):
        raise ValueError("Locked task class count is inconsistent")
    head_spec = task_spec["head_fits"][model_kind]
    if (
        head_spec.get("C") != 10.0
        or head_spec.get("gamma") != "1/d"
        or head_spec.get("class_weight", "missing") is not None
        or head_spec.get("views") != list(HEAD_VIEWS)
        or head_spec.get("calibration_folds") != 5
        or head_spec.get("seed") != 42
    ):
        raise ValueError("Final head differs from the selected unweighted, grouped calibrated RBF")
    expected_output = _resolve(root, head_spec["output_dir"])
    if not expected_output.is_relative_to(root) or expected_output != output_dir.resolve():
        raise ValueError("Output directory is not the exact locked final-head destination")
    return lock, task_spec, head_spec


def calibration_partition(
    frame: pd.DataFrame, *, folds: int = 5, seed: int = 42
) -> tuple[list, pd.DataFrame, list[dict]]:
    labels = frame.label_index.to_numpy(dtype=int)
    groups = frame.source_group.to_numpy(dtype=str)
    splits = list(
        StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed).split(
            np.zeros((len(frame), 1)), labels, groups
        )
    )
    assignment = np.full(len(frame), -1, dtype=int)
    details = []
    for fold, (training, calibration) in enumerate(splits):
        if set(groups[training]) & set(groups[calibration]):
            raise ValueError("Calibration groups overlap fitting groups")
        if set(labels[training]) != set(labels) or set(labels[calibration]) != set(labels):
            raise ValueError("Calibration fold is missing a task class")
        if (assignment[calibration] != -1).any():
            raise ValueError("A row appears in multiple calibration partitions")
        assignment[calibration] = fold
        details.append(
            {
                "fold": fold,
                "fitting_rows": len(training),
                "calibration_rows": len(calibration),
                "fitting_groups": len(set(groups[training])),
                "calibration_groups": len(set(groups[calibration])),
                "group_overlap": 0,
                "fitting_image_ids_sha256": canonical_hash(frame.iloc[training].image_id.tolist()),
                "calibration_image_ids_sha256": canonical_hash(
                    frame.iloc[calibration].image_id.tolist()
                ),
            }
        )
    if (assignment < 0).any():
        raise ValueError("Some rows never receive a calibration role")
    table = frame[["image_id", "source_group", "label_index", "split"]].copy()
    table["calibration_fold"] = assignment
    return splits, table, details


def head_fit_diagnostics(model, splits: list, *, feature_dimensions: int) -> list[dict]:
    if len(model.calibrated_classifiers_) != 5:
        raise RuntimeError("The fitted calibration ensemble must contain exactly five estimators")
    rows = []
    for index, (calibrated, (training, calibration)) in enumerate(
        zip(model.calibrated_classifiers_, splits, strict=True)
    ):
        scaler = calibrated.estimator.named_steps["standardscaler"]
        kernel_head = calibrated.estimator.named_steps["cudakernelsvc"]
        solver = kernel_head.estimator_
        if int(solver.fit_status_) != 0:
            raise RuntimeError("A final libsvm fit did not converge")
        if solver.class_weight is not None or float(solver.C) != 10.0:
            raise RuntimeError("Final libsvm weighting/regularization changed")
        if np.asarray(scaler.n_samples_seen_).max() != len(training):
            raise RuntimeError("Scaler was not fitted solely on its calibration-training partition")
        if kernel_head.reference_.shape != (len(training), feature_dimensions):
            raise RuntimeError("Fitted RBF reference rows or feature dimension changed")
        if kernel_head.gamma_ != 1.0 / feature_dimensions:
            raise RuntimeError("Fitted kernel gamma changed")
        rows.append(
            {
                "fold": index,
                "fit_status": int(solver.fit_status_),
                "solver_iterations": np.asarray(solver.n_iter_, dtype=int).tolist(),
                "support_vectors_per_class": np.asarray(solver.n_support_, dtype=int).tolist(),
                "class_weights": np.asarray(solver.class_weight_, dtype=float).tolist(),
                "scaler_training_rows": int(np.asarray(scaler.n_samples_seen_).max()),
                "reference_rows": len(kernel_head.reference_),
                "calibration_rows": len(calibration),
                "gamma": kernel_head.gamma_,
                "scaler_mean_sha256": canonical_hash(scaler.mean_.tolist()),
                "scaler_scale_sha256": canonical_hash(scaler.scale_.tolist()),
            }
        )
    return rows


def _probabilities(model, features: np.ndarray, classes: int) -> np.ndarray:
    if not np.array_equal(model.classes_, np.arange(classes)):
        raise RuntimeError("Fitted head class order differs from the locked task")
    result = np.asarray(model.predict_proba(features), dtype=np.float64)
    if (
        result.shape != (len(features), classes)
        or not np.isfinite(result).all()
        or (result < 0).any()
    ):
        raise RuntimeError("Invalid final-head probabilities")
    sums = result.sum(axis=1, keepdims=True)
    if not np.allclose(sums, 1.0, atol=1e-7, rtol=0):
        raise RuntimeError("Final-head probabilities are not normalized")
    return result / sums


def _verified_head_summary(directory: Path, expected_lock_sha256: str) -> dict:
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("selection_lock_sha256") != expected_lock_sha256
    ):
        raise RuntimeError("Final head is incomplete or belongs to another selection lock")
    if request.get("selection_lock_sha256") != expected_lock_sha256 or summary.get(
        "request_sha256"
    ) != canonical_hash(request):
        raise RuntimeError("Final head request has drifted")
    if summary.get("test_rows_read") != 0 or summary.get("test_labels_read") is not False:
        raise RuntimeError("Final head was not fitted exclusively on development data")
    if request.get("role") != HEAD_STAGE or request.get("smoke") is not False:
        raise RuntimeError("Final head request is not a production train+validation refit")
    for name in (
        "role",
        "stage",
        "task",
        "model_kind",
        "classes",
        "class_names",
        "training_rows",
        "feature_dimensions",
    ):
        if summary.get(name) != request.get(name):
            raise RuntimeError(f"Final head summary differs from its request in {name}")
    if summary.get("reload_prediction_replay", {}).get("passed") is not True:
        raise RuntimeError("Final head checkpoint lacks a successful reload replay")
    artifacts = summary.get("artifacts", {})
    if not {
        "head.joblib",
        "reload_replay.npz",
        "replay_audit.json",
        "calibration_assignments.csv",
        "calibration_folds.json",
    } <= set(artifacts):
        raise RuntimeError("Final head is missing required audit artifacts")
    _verify_artifacts(directory, artifacts)
    return summary


def fit_locked_head(selection_lock: Path, task: str, model_kind: str, output_dir: Path) -> dict:
    """Exclusive writer entry point; validate destination before creating it."""
    locked_head_spec(selection_lock, task, model_kind, output_dir)
    with _cache_lock(output_dir):
        return _fit_locked_head(selection_lock, task, model_kind, output_dir)


def _fit_locked_head(selection_lock: Path, task: str, model_kind: str, output_dir: Path) -> dict:
    """Refit the locked head on train+validation; never open its test manifest."""
    started = time.perf_counter()
    lock, task_spec, head_spec = locked_head_spec(selection_lock, task, model_kind, output_dir)
    lock_digest = sha256_file(selection_lock)
    output_dir = output_dir.resolve()
    if (output_dir / "summary.json").exists():
        return _verified_head_summary(output_dir, lock_digest)
    manifest_path = _resolve(selection_lock.parent, task_spec["development_manifest"])
    if sha256_file(manifest_path) != task_spec["development_manifest_sha256"]:
        raise RuntimeError("Final fitting manifest differs from the selection lock")
    frame = load_development(manifest_path, num_classes=task_spec["classes"])
    if label_names(frame) != task_spec["class_names"]:
        raise RuntimeError("Development class names differ from the selection lock")
    require_cuda()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    feature_root = _resolve(selection_lock.parent, lock["development_feature_root"])
    values, feature_evidence = [], {}
    for view in HEAD_VIEWS:
        matrix, feature_evidence[view] = aligned_features(feature_root / model_kind / view, frame)
        values.append(matrix)
    features = np.concatenate(values, axis=1)
    del values
    splits, assignments, folds = calibration_partition(frame)
    request = {
        "stage": HEAD_STAGE,
        "role": HEAD_STAGE,
        "smoke": False,
        "fit_scope": "all_original_train_plus_validation",
        "selection_lock_sha256": lock_digest,
        "task": task,
        "model_kind": model_kind,
        "class_names": task_spec["class_names"],
        "classes": task_spec["classes"],
        "development_manifest_sha256": sha256_file(manifest_path),
        "training_rows": len(frame),
        "train_rows": len(frame),
        "training_split_counts": {
            name: int(count) for name, count in frame.groupby("split").size().items()
        },
        "feature_dimensions": features.shape[1],
        "head_spec": head_spec,
        "feature_evidence": feature_evidence,
        "environment": environment_evidence(),
        "test_rows_read": 0,
        "test_labels_read": False,
        "reload_prediction_tolerance": 1e-7,
    }
    lock_json(output_dir / "request.json", request)
    if (output_dir / "head.joblib").exists():
        raise RuntimeError(
            "An unfinished final head checkpoint already exists; preserve it for audit"
        )
    print(
        json.dumps(
            {
                "event": "LOCKED_HEAD_FIT_START",
                "task": task,
                "model_kind": model_kind,
                "rows": len(frame),
                "dimensions": features.shape[1],
                "utc": utc_now(),
            }
        ),
        flush=True,
    )
    fit_started = time.perf_counter()
    model = fit_group_calibrated_rbf(
        features,
        frame.label_index.to_numpy(dtype=int),
        frame.source_group.to_numpy(dtype=str),
        seed=42,
        folds=5,
    )
    fit_seconds = time.perf_counter() - fit_started
    diagnostics = head_fit_diagnostics(model, splits, feature_dimensions=features.shape[1])
    replay_indices = np.unique(np.linspace(0, len(frame) - 1, min(32, len(frame)), dtype=int))
    replay_features = features[replay_indices].copy()
    expected = _probabilities(model, replay_features, task_spec["classes"])
    temporary = output_dir / "head.joblib.tmp"
    joblib.dump(model, temporary, compress=1)
    temporary.replace(output_dir / "head.joblib")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = joblib.load(output_dir / "head.joblib")
    replayed = _probabilities(reloaded, replay_features, task_spec["classes"])
    maximum_difference = float(np.max(np.abs(expected - replayed)))
    replay_passed = maximum_difference <= request["reload_prediction_tolerance"] and np.array_equal(
        expected.argmax(axis=1), replayed.argmax(axis=1)
    )
    if not replay_passed:
        raise RuntimeError("Final checkpoint failed its locked reload prediction replay")
    np.savez_compressed(
        output_dir / "reload_replay.npz",
        image_ids=frame.iloc[replay_indices].image_id.to_numpy(dtype=str),
        expected_probabilities=expected,
        reloaded_probabilities=replayed,
    )
    atomic_json(
        output_dir / "replay_audit.json",
        {
            "status": "PASS",
            "predictions_identical": True,
            "selection_lock_sha256": lock_digest,
            "checkpoint_sha256": sha256_file(output_dir / "head.joblib"),
            "rows": len(replay_indices),
            "scope": "deterministic development rows only",
            "maximum_absolute_probability_difference": maximum_difference,
            "probability_tolerance": request["reload_prediction_tolerance"],
            "test_rows_read": 0,
            "test_labels_read": False,
        },
    )
    assignments.to_csv(output_dir / "calibration_assignments.csv", index=False, lineterminator="\n")
    atomic_json(
        output_dir / "calibration_folds.json",
        {
            "partitions": folds,
            "fits": diagnostics,
            "scaling": "independent StandardScaler inside each fitting fold",
            "calibration": "sigmoid on disjoint source groups",
            "folds": 5,
        },
    )
    summary = {
        "status": "COMPLETE",
        "stage": HEAD_STAGE,
        "role": HEAD_STAGE,
        "selection_lock_sha256": lock_digest,
        "request_sha256": canonical_hash(request),
        "task": task,
        "model_kind": model_kind,
        "classes": task_spec["classes"],
        "class_names": task_spec["class_names"],
        "training_rows": len(frame),
        "train_rows": len(frame),
        "development_manifest_sha256": request["development_manifest_sha256"],
        "feature_dimensions": features.shape[1],
        "fit_seconds": fit_seconds,
        "runtime_seconds": time.perf_counter() - started,
        "optimization": {
            "kernel": "CUDA_FP32_without_TF32",
            "optimizer": "CPU_libsvm",
            "calibration": "CPU_sigmoid",
            "class_weight": None,
            "all_folds_converged": True,
        },
        "reload_prediction_replay": {
            "passed": True,
            "rows": len(replay_indices),
            "maximum_absolute_difference": maximum_difference,
            "tolerance": request["reload_prediction_tolerance"],
            "argmax_matches": True,
            "scope": "deterministic development rows only; not a performance measurement",
        },
        "artifacts": {
            name: sha256_file(output_dir / name)
            for name in (
                "head.joblib",
                "reload_replay.npz",
                "replay_audit.json",
                "calibration_assignments.csv",
                "calibration_folds.json",
            )
        },
        "test_rows_read": 0,
        "test_labels_read": False,
        "completed_utc": utc_now(),
    }
    atomic_json(output_dir / "summary.json", summary)
    return summary


def predict_locked_head(
    head_directory: Path, features: np.ndarray, *, expected_lock_sha256: str
) -> np.ndarray:
    """Predict with a hash-verified final head; class order comes from its lock."""
    summary = _verified_head_summary(head_directory, expected_lock_sha256)
    features = np.asarray(features, dtype=np.float32)
    if (
        features.ndim != 2
        or features.shape[1] != summary["feature_dimensions"]
        or not np.isfinite(features).all()
    ):
        raise ValueError("Inference features differ from the locked head dimensions")
    model = joblib.load(head_directory / "head.joblib")
    return _probabilities(model, features, summary["classes"])


def read_locked_inference_manifest(path: Path, access: dict) -> pd.DataFrame:
    """Check the access receipt before reading even the label-free CSV header."""
    if (
        path.resolve() != Path(access["inference_manifest"]).resolve()
        or sha256_file(path) != access["inference_manifest_sha256"]
    ):
        raise RuntimeError("Label-free inference manifest differs from the verified access gate")
    columns = set(pd.read_csv(path, nrows=0).columns)
    if columns != INFERENCE_COLUMNS:
        raise ValueError("Held-out extraction accepts only the exact label-free inference schema")
    frame = pd.read_csv(
        path,
        dtype={"image_id": str, "image_path": str, "split": str, "image_sha256": str},
        keep_default_na=False,
    )
    if len(frame) != access["inference_rows"] or frame.empty or set(frame.split) != {"test"}:
        raise ValueError("Held-out inference cohort count or split differs from the gate")
    if (
        frame.image_id.duplicated().any()
        or frame.image_path.duplicated().any()
        or frame.image_id.str.strip().eq("").any()
    ):
        raise ValueError("Held-out inference identities/paths must be unique and nonempty")
    if not frame.image_sha256.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Held-out inference pixels must be bound to SHA256 digests")
    frame["image_path"] = frame.image_path.map(lambda value: str((path.parent / value).resolve()))
    if frame.image_path.duplicated().any():
        raise ValueError("Held-out inference paths resolve to duplicate images")
    if not frame.image_path.map(lambda value: Path(value).is_file()).all():
        raise FileNotFoundError("A locked held-out image is missing")
    boxes = frame[list(BBOX_COLUMNS)].apply(pd.to_numeric, errors="raise")
    if (
        not np.isfinite(boxes.to_numpy()).all()
        or (boxes.bbox_xmax <= boxes.bbox_xmin).any()
        or (boxes.bbox_ymax <= boxes.bbox_ymin).any()
        or (boxes[["bbox_xmin", "bbox_ymin"]] < 0).any().any()
    ):
        raise ValueError("Invalid locked inference bounding boxes")
    frame[list(BBOX_COLUMNS)] = boxes
    return frame


def reference_development_contract(cache: Path, model_kind: str, view: str) -> dict:
    contract = json.loads((cache / "contract.json").read_text(encoding="utf-8"))
    provenance = json.loads((cache / "provenance.json").read_text(encoding="utf-8"))
    if provenance.get("status") != DEVELOPMENT_CACHE_STATUS or provenance.get(
        "contract_sha256"
    ) != canonical_hash(contract):
        raise RuntimeError("Reference development feature cache is incomplete or altered")
    if (
        contract.get("scope") != "development"
        or contract.get("device_type") != "cuda"
        or contract.get("test_rows_read") != 0
        or contract.get("test_labels_read") is not False
    ):
        raise RuntimeError("Reference features are not full CUDA development-only features")
    if (
        contract.get("model_kind") != model_kind
        or contract.get("view") != view
        or contract.get("model") != MODEL_SPECS[model_kind]
    ):
        raise RuntimeError("Reference feature model/view specification changed")
    if not {"features.npy", "rows.csv"} <= set(provenance.get("artifact_sha256", {})):
        raise RuntimeError("Reference feature cache lacks pixel/feature integrity records")
    _verify_artifacts(cache, provenance["artifact_sha256"])
    return contract


@torch.inference_mode()
def _cache_locked_with_runtime(
    request: FeatureRequest,
    frame: pd.DataFrame,
    model: nn.Module,
    transform,
    checkpoint: dict,
    preprocess: dict,
    device: torch.device,
    access: dict,
    reference: dict,
) -> dict:
    """Tiny CPU fixtures exercise resumability; the production entry requires CUDA."""
    import transformers

    started = time.perf_counter()
    if (
        set(frame) != INFERENCE_COLUMNS
        or set(frame.split) != {"test"}
        or len(frame) != access["inference_rows"]
        or request.max_images is not None
        or request.allow_download
    ):
        raise ValueError(
            "Locked extraction requires the full label-free cohort and offline weights"
        )
    if checkpoint != reference["checkpoint"] or preprocess != reference["preprocess"]:
        raise RuntimeError(
            "Held-out feature extraction does not match development model/preprocessing"
        )
    output = request.output_dir
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "scope": "locked_test_inference",
        "selection_lock_sha256": access["selection_lock_sha256"],
        "barrier_sha256": access["barrier_sha256"],
        "manifest_sha256": access["inference_manifest_sha256"],
        "rows": len(frame),
        "model_kind": request.model_kind,
        "model": MODEL_SPECS[request.model_kind],
        "view": request.view,
        "checkpoint": checkpoint,
        "preprocess": preprocess,
        "reference_development_contract_sha256": canonical_hash(reference),
        "batch_size": request.batch_size,
        "chunk_rows": request.chunk_rows,
        "autocast_dtype": request.autocast_dtype,
        "device_type": device.type,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "test_rows_read": len(frame),
        "test_labels_read": False,
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "polar_locked_features.py",
                "polar_benchmark_features.py",
                "polar_features.py",
                "polar_models.py",
                "vcoco_v3_representations.py",
                "polar.py",
                "augmentations.py",
            )
        },
    }
    contract_hash = canonical_hash(contract)
    provenance_path = output / "provenance.json"
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            existing.get("status") != LOCKED_CACHE_STATUS
            or existing.get("contract_sha256") != contract_hash
        ):
            raise RuntimeError("Existing held-out feature cache belongs to another locked contract")
        _verify_artifacts(output, existing["artifact_sha256"])
        if (
            canonical_hash(json.loads((output / "contract.json").read_text(encoding="utf-8")))
            != contract_hash
        ):
            raise RuntimeError("Existing held-out cache contract bytes changed")
        _verify_source_rows(pd.read_csv(output / "rows.csv", dtype={"image_id": str}), frame)
        return existing
    lock_json(output / "contract.json", contract)
    chunks = _resume_chunks(output, contract_hash, frame)
    position = chunks[-1]["end"] if chunks else 0
    resumed_rows = position
    loader = DataLoader(
        BenchmarkFeatureDataset(frame.iloc[position:], view=request.view, transform=transform),
        batch_size=request.batch_size,
        shuffle=False,
        num_workers=request.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=request.workers > 0,
    )
    model = model.to(device).eval().requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pending, hashes = [], []
    dimensions = chunks[0]["shape"][1] if chunks else None
    extraction_started = time.perf_counter()
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=getattr(torch, request.autocast_dtype),
            enabled=device.type == "cuda",
        ):
            features = model(pixels).float().cpu().numpy()
        if features.ndim != 2 or len(features) != len(pixels) or not np.isfinite(features).all():
            raise RuntimeError("Invalid locked feature output")
        if dimensions is not None and features.shape[1] != dimensions:
            raise RuntimeError("Locked feature dimensionality changed between chunks")
        dimensions = features.shape[1]
        if device.type == "cuda" and dimensions != MODEL_SPECS[request.model_kind]["dimensions"]:
            raise RuntimeError("Pinned backbone returned the wrong representation dimension")
        pending.append(features)
        hashes.extend(batch["image_sha256"])
        if len(hashes) >= request.chunk_rows or position + len(hashes) == len(frame):
            chunk = _save_chunk(
                output, contract_hash, frame, position, np.concatenate(pending), hashes
            )
            chunks.append(chunk)
            position = chunk["end"]
            pending, hashes = [], []
            rate = (position - resumed_rows) / max(time.perf_counter() - extraction_started, 1e-9)
            progress = {
                "status": "LOCKED_TEST_EXTRACTING",
                "model_kind": request.model_kind,
                "view": request.view,
                "rows_complete": position,
                "rows_total": len(frame),
                "images_per_second": rate,
                "estimated_seconds_remaining": (len(frame) - position) / max(rate, 1e-9),
            }
            atomic_json(output / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    if position != len(frame) or not chunks:
        raise RuntimeError("Locked extraction did not produce the complete cohort")
    temporary = output / "features.npy.tmp"
    combined = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(frame), dimensions)
    )
    rows = []
    for chunk in chunks:
        combined[chunk["start"] : chunk["end"]] = np.load(
            output / "chunks" / chunk["features_file"], mmap_mode="r", allow_pickle=False
        )
        rows.append(pd.read_csv(output / "chunks" / chunk["rows_file"], dtype={"image_id": str}))
    combined.flush()
    del combined
    temporary.replace(output / "features.npy")
    pd.concat(rows, ignore_index=True).to_csv(output / "rows.csv", index=False, lineterminator="\n")
    artifacts = {name: sha256_file(output / name) for name in ("features.npy", "rows.csv")}
    provenance = {
        **contract,
        "status": LOCKED_CACHE_STATUS,
        "contract_sha256": contract_hash,
        "feature_shape": [len(frame), dimensions],
        "feature_dtype": "float32",
        "artifact_sha256": artifacts,
        "features_sha256": artifacts["features.npy"],
        "rows_sha256": artifacts["rows.csv"],
        "resumed_rows": resumed_rows,
        "runtime_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "completed_utc": utc_now(),
    }
    atomic_json(provenance_path, provenance)
    atomic_json(
        output / "progress.json", {"status": LOCKED_CACHE_STATUS, "rows_complete": len(frame)}
    )
    return provenance


def cache_locked_test_features(selection_lock: Path, run_dir: Path) -> dict:
    from hac.polar_locked_evaluation import verify_test_access

    access = verify_test_access(selection_lock, run_dir)
    lock = _read_selection_lock(selection_lock)
    if Path(lock["output_root"]).resolve() != run_dir.resolve():
        raise ValueError("Held-out feature destination differs from the selection lock")
    manifest = Path(access["inference_manifest"])
    frame = read_locked_inference_manifest(manifest, access)
    device = require_cuda()
    models = {model for task in lock["tasks"].values() for model in task["head_fits"]}
    if models != set(HEAD_MODELS):
        raise ValueError("The locked four-backbone challenger panel changed")
    root = _resolve(selection_lock.parent, lock["development_feature_root"])
    summaries = {}
    for model_kind in HEAD_MODELS:
        for view in HEAD_VIEWS:
            # Reverify the immutable completed-fit barrier before every model/view.
            renewed = verify_test_access(selection_lock, run_dir)
            if renewed != access:
                raise RuntimeError("Held-out access receipt changed during extraction")
            reference = reference_development_contract(root / model_kind / view, model_kind, view)
            request = FeatureRequest(
                manifest=manifest,
                output_dir=run_dir / "test_features" / model_kind / view,
                model_kind=model_kind,
                view=view,
                preprocess=reference["preprocess"]["policy"],
                batch_size=reference["batch_size"],
                workers=4,
                chunk_rows=reference["chunk_rows"],
                cache_dir=Path(lock["hf_cache_dir"]) if lock.get("hf_cache_dir") else None,
                allow_download=False,
                max_images=None,
                autocast_dtype=reference["autocast_dtype"],
            )
            request.validate()
            torch.set_float32_matmul_precision(reference["matmul_precision"])
            if request.autocast_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
                raise RuntimeError("CUDA device does not support the locked extraction dtype")
            with _cache_lock(request.output_dir):
                model, transform, checkpoint, preprocess = prepare_model_and_transform(request)
                summaries[f"{model_kind}/{view}"] = _cache_locked_with_runtime(
                    request,
                    frame,
                    model,
                    transform,
                    checkpoint,
                    preprocess,
                    device,
                    access,
                    reference,
                )
                del model, transform
            gc.collect()
            torch.cuda.empty_cache()
    result = {
        "status": "COMPLETE",
        "selection_lock_sha256": access["selection_lock_sha256"],
        "barrier_sha256": access["barrier_sha256"],
        "inference_manifest_sha256": access["inference_manifest_sha256"],
        "rows": len(frame),
        "test_labels_read": False,
        "caches": {
            key: {
                "contract_sha256": value["contract_sha256"],
                "features_sha256": value["features_sha256"],
                "rows_sha256": value["rows_sha256"],
            }
            for key, value in summaries.items()
        },
        "completed_utc": utc_now(),
    }
    atomic_json(run_dir / "test_features" / "summary.json", result)
    return result


def load_aligned_test_features(
    cache_root: Path, model_kind: str, ids, *, expected_lock_sha256: str
) -> np.ndarray:
    if model_kind not in HEAD_MODELS:
        raise ValueError("Unknown locked feature backbone")
    identifiers = [str(value) for value in ids]
    if not identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("Inference identifiers must be nonempty and unique")
    matrices, fingerprints = [], []
    for view in HEAD_VIEWS:
        directory = cache_root / model_kind / view
        contract = json.loads((directory / "contract.json").read_text(encoding="utf-8"))
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        if provenance.get("status") != LOCKED_CACHE_STATUS or provenance.get(
            "contract_sha256"
        ) != canonical_hash(contract):
            raise RuntimeError("Held-out features are incomplete or their contract changed")
        if (
            contract.get("selection_lock_sha256") != expected_lock_sha256
            or contract.get("scope") != "locked_test_inference"
            or contract.get("test_labels_read") is not False
            or contract.get("device_type") != "cuda"
        ):
            raise RuntimeError("Held-out features lack the required locked CUDA provenance")
        if contract.get("model_kind") != model_kind or contract.get("view") != view:
            raise RuntimeError("Held-out feature model/view mismatch")
        if contract.get("model") != MODEL_SPECS[model_kind]:
            raise RuntimeError("Held-out feature representation specification changed")
        if not {"features.npy", "rows.csv"} <= set(provenance.get("artifact_sha256", {})):
            raise RuntimeError("Held-out feature integrity records are missing")
        _verify_artifacts(directory, provenance["artifact_sha256"])
        rows = pd.read_csv(directory / "rows.csv", dtype={"image_id": str, "image_sha256": str})
        if (
            set(rows) != {"row", "image_id", "split", "image_sha256"}
            or rows.image_id.duplicated().any()
            or set(rows.split) != {"test"}
        ):
            raise RuntimeError(
                "Held-out feature rows contain labels, duplicates or the wrong split"
            )
        if (
            len(rows) != contract.get("rows")
            or len(rows) != contract.get("test_rows_read")
            or not np.array_equal(rows.row.to_numpy(), np.arange(len(rows)))
        ):
            raise RuntimeError("Held-out feature rows are incomplete or out of order")
        if not set(identifiers) <= set(rows.image_id):
            raise RuntimeError("Held-out feature cache is missing requested identities")
        positions = pd.Series(np.arange(len(rows)), index=rows.image_id).loc[identifiers].to_numpy()
        features = np.load(directory / "features.npy", mmap_mode="r", allow_pickle=False)
        if features.dtype != np.float32 or features.shape != (
            len(rows),
            MODEL_SPECS[model_kind]["dimensions"],
        ):
            raise RuntimeError("Held-out feature array dimensions or dtype changed")
        if provenance.get("feature_shape") != list(features.shape):
            raise RuntimeError("Held-out feature shape differs from completion evidence")
        selected = np.asarray(features[positions], dtype=np.float32)
        if not np.isfinite(selected).all():
            raise RuntimeError("Held-out feature values are nonfinite")
        matrices.append(selected)
        fingerprints.append(
            (
                contract["manifest_sha256"],
                contract["barrier_sha256"],
                rows.iloc[positions].image_sha256.tolist(),
            )
        )
    if fingerprints[0] != fingerprints[1]:
        raise RuntimeError(
            "Held-out full-frame/crop caches refer to different source pixels or access barriers"
        )
    return np.concatenate(matrices, axis=1)
