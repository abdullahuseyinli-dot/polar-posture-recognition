import json

import numpy as np
import pandas as pd
import pytest

from hac.polar import sha256_file
from hac.polar_benchmark import (
    atomic_json,
    canonical_hash,
    check_completed,
    lock_json,
    probability_metrics,
    validate_development,
)


def manifest(classes=9):
    return pd.DataFrame(
        [
            {
                "image_id": f"{split}-{label}",
                "image_path": f"{split}-{label}.jpg",
                "split": split,
                "label": f"class-{label}",
                "label_index": label,
                "source_group": f"{split}-{label}",
            }
            for split in ("train", "val")
            for label in range(classes)
        ]
    )


def test_nine_class_contract():
    assert len(validate_development(manifest(), num_classes=9)) == 18


@pytest.mark.parametrize("mutation", ["test", "duplicate", "source", "mapping", "fractional"])
def test_development_rejects_leakage_and_drift(mutation):
    frame = manifest()
    if mutation == "test":
        frame.loc[0, "split"] = "test"
    elif mutation == "duplicate":
        frame.loc[0, "image_id"] = frame.loc[1, "image_id"]
    elif mutation == "source":
        frame.loc[0, "source_group"] = frame.loc[9, "source_group"]
    elif mutation == "mapping":
        frame.loc[0, "label"] = "wrong-name"
    else:
        frame["label_index"] = frame.label_index.astype(float)
        frame.loc[0, "label_index"] = 0.5
    with pytest.raises(ValueError):
        validate_development(frame, num_classes=9)


def test_metrics_generic_classes():
    report = probability_metrics(np.arange(9), np.eye(9), [str(x) for x in range(9)])
    assert report["macro_f1"] == 1.0
    assert report["errors"] == 0
    assert len(report["confusion_matrix"]) == 9


def test_lock_refuses_mutation(tmp_path):
    path = tmp_path / "lock.json"
    lock_json(path, {"selection": "validation-only"})
    lock_json(path, {"selection": "validation-only"})
    with pytest.raises(RuntimeError):
        lock_json(path, {"selection": "test"})


def test_completed_integrity(tmp_path):
    request = {"task": "nine"}
    artifact = tmp_path / "artifact.json"
    atomic_json(artifact, {"ok": True})
    atomic_json(
        tmp_path / "summary.json",
        {
            "status": "COMPLETE",
            "request_sha256": canonical_hash(request),
            "artifacts": {artifact.name: sha256_file(artifact)},
        },
    )
    assert check_completed(tmp_path, request)["status"] == "COMPLETE"
    atomic_json(artifact, {"ok": False})
    with pytest.raises(RuntimeError):
        check_completed(tmp_path, request)


def test_completed_rejects_path_escape(tmp_path):
    request = {"task": "nine"}
    atomic_json(
        tmp_path / "summary.json",
        {
            "status": "COMPLETE",
            "request_sha256": canonical_hash(request),
            "artifacts": {"../outside": "fake"},
        },
    )
    with pytest.raises(RuntimeError):
        check_completed(tmp_path, request)
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "COMPLETE"
