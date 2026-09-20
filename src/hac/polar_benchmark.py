"""Fail-closed contracts for the new, separate POLAR competitive benchmark.

Historical result files are inputs only. Development selection never accepts test
rows. A lock describes the actual new implementation, not a historical replay.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from hac.metrics import classification_metrics
from hac.polar import sha256_file


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def lock_json(path: Path, value: dict) -> None:
    """Create once, or verify an identical existing lock; never silently amend."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError(f"Existing lock differs: {path}")
        return
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def implementation_evidence(root: Path) -> dict[str, str]:
    paths = sorted((root / "src" / "hac").glob("*.py"))
    paths += sorted((root / "experiments").glob("*polar_benchmark*.py"))
    paths += sorted((root / "tools").glob("*polar_benchmark*.py"))
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def validate_development(frame: pd.DataFrame, *, num_classes: int) -> pd.DataFrame:
    required = {"image_id", "image_path", "split", "label", "label_index", "source_group"}
    if missing := required - set(frame):
        raise ValueError(f"Missing development fields: {sorted(missing)}")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError("Empty or missing-valued development data")
    output = frame.copy()
    output["image_id"] = output.image_id.astype(str)
    output["source_group"] = output.source_group.astype(str)
    if output.image_id.duplicated().any():
        raise ValueError("Duplicate development image IDs")
    if set(output.split) != {"train", "val"}:
        raise ValueError("Development accepts exactly train and val; test is forbidden")
    indices = pd.to_numeric(output.label_index, errors="raise")
    if not np.equal(indices, np.floor(indices)).all():
        raise ValueError("Class indices must be integers")
    output["label_index"] = indices.astype(int)
    for split in ("train", "val"):
        if set(output.loc[output.split.eq(split), "label_index"]) != set(range(num_classes)):
            raise ValueError(f"{split} must include exactly all {num_classes} classes")
    if (output.groupby("label_index").label.nunique() != 1).any() or (
        output.groupby("label").label_index.nunique() != 1
    ).any():
        raise ValueError("Class name/index mapping is not one to one")
    if (output.groupby("source_group").split.nunique() > 1).any():
        raise ValueError("A detected source group crosses train/validation")
    return output.sort_values("image_id", ignore_index=True)


def load_development(path: Path, *, num_classes: int) -> pd.DataFrame:
    return validate_development(
        pd.read_csv(path, dtype={"image_id": str, "source_group": str}), num_classes=num_classes
    )


def label_names(frame: pd.DataFrame) -> list[str]:
    return (
        frame[["label_index", "label"]].drop_duplicates().sort_values("label_index").label.tolist()
    )


def probability_metrics(labels: np.ndarray, probabilities: np.ndarray, names: list[str]) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(labels), len(names)):
        raise ValueError("Probability dimensions do not match the declared task")
    if set(labels) != set(range(len(names))):
        raise ValueError("Evaluation must contain every declared class")
    metrics = classification_metrics(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    return {
        **metrics,
        "rows": int(len(labels)),
        "errors": int((predictions != labels).sum()),
        "class_names": names,
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=range(len(names))
        ).tolist(),
        "per_class": classification_report(
            labels,
            predictions,
            labels=range(len(names)),
            target_names=names,
            output_dict=True,
            zero_division=0,
        ),
    }


def environment_evidence() -> dict:
    import importlib.metadata
    import sys

    import torch

    versions = {}
    for package in (
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "huggingface-hub",
        "pillow",
    ):
        versions[package] = importlib.metadata.version(package)
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def check_completed(output: Path, request: dict) -> dict | None:
    marker = output / "summary.json"
    if not marker.exists():
        return None
    previous = json.loads(marker.read_text(encoding="utf-8"))
    if previous.get("status") != "COMPLETE" or previous.get("request_sha256") != canonical_hash(
        request
    ):
        raise RuntimeError(f"Completed output does not match request: {output}")
    for relative, digest in previous.get("artifacts", {}).items():
        path = (output / relative).resolve()
        if (
            not path.is_relative_to(output.resolve())
            or not path.is_file()
            or sha256_file(path) != digest
        ):
            raise RuntimeError(f"Completed artifact drift: {relative}")
    if not previous.get("artifacts"):
        raise RuntimeError("Completed run has no artifact integrity records")
    return previous


def aligned_features(cache_dir: Path, manifest: pd.DataFrame) -> tuple[np.ndarray, dict]:
    from hac.polar_benchmark_features import CACHE_STATUS

    provenance_path = cache_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    rows_path, values_path = cache_dir / "rows.csv", cache_dir / "features.npy"
    contract = json.loads((cache_dir / "contract.json").read_text(encoding="utf-8"))
    if provenance.get("status") != CACHE_STATUS or provenance.get(
        "contract_sha256"
    ) != canonical_hash(contract):
        raise ValueError("Incomplete or altered feature-cache contract")
    if (
        contract.get("scope") != "development"
        or contract.get("test_rows_read") != 0
        or contract.get("test_labels_read") is not False
    ):
        raise ValueError("Only full development-only feature caches are admissible")
    if contract.get("device_type") != "cuda":
        raise ValueError("Production caches must have been extracted on CUDA")
    if (
        contract.get("model_kind") != cache_dir.parent.name
        or contract.get("view") != cache_dir.name
    ):
        raise ValueError("Feature cache model/view does not match its requested location")
    for path, key in ((rows_path, "rows_sha256"), (values_path, "features_sha256")):
        if provenance.get(key) != sha256_file(path):
            raise ValueError(f"Feature-cache bytes do not match extraction evidence: {path.name}")
    rows = pd.read_csv(rows_path, dtype={"image_id": str})
    if rows.image_id.duplicated().any() or not set(rows.split).issubset({"train", "val"}):
        raise ValueError("Invalid feature-pool rows or test contamination")
    features = np.load(values_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or len(rows) != len(features):
        raise ValueError("Feature-pool shape drift")
    by_id = rows.set_index("image_id")
    if not set(manifest.image_id).issubset(set(by_id.index)):
        raise ValueError("Feature cache is missing task rows")
    selected = by_id.loc[manifest.image_id]
    if selected.split.tolist() != manifest.split.tolist():
        raise ValueError("Feature-pool split drift")
    hash_column = "image_sha256" if "image_sha256" in manifest else "sha256"
    if hash_column not in manifest or "image_sha256" not in selected:
        raise ValueError("Both features and task data must bind image byte hashes")
    if selected.image_sha256.tolist() != manifest[hash_column].tolist():
        raise ValueError("Feature extraction image bytes differ from audited task data")
    order = pd.Series(np.arange(len(rows)), index=rows.image_id).loc[manifest.image_id].to_numpy()
    values = np.asarray(features[order], dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite feature values")
    evidence = {
        "cache": str(cache_dir.resolve()),
        "provenance_sha256": sha256_file(provenance_path),
        "features_sha256": sha256_file(values_path),
        "rows_sha256": sha256_file(rows_path),
        "provenance": provenance,
    }
    return values, evidence
