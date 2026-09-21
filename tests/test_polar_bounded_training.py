"""Contracts for fixed budgets, label gating, integrity, and exact prediction replay."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import train_polar_bounded as training

from hac.polar import sha256_file
from hac.polar_benchmark import canonical_hash


def manifest(tmp_path, classes=4):
    rows = []
    for split in ("train", "val"):
        for label in range(classes):
            for example in range(3):
                key = f"{split}-{label}-{example}"
                rows.append(
                    {
                        "image_id": key,
                        "source_group": key,
                        "image_path": str(tmp_path / f"{key}.png"),
                        "split": split,
                        "label": f"class_{label}",
                        "label_index": label,
                        "bbox_xmin": 0,
                        "bbox_ymin": 0,
                        "bbox_xmax": 8,
                        "bbox_ymax": 16,
                        "image_sha256": "a" * 64,
                    }
                )
    path = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_fixed_cli_has_no_sweep_or_initialization_switch(tmp_path):
    args = training.parse_args(
        [
            "--manifest",
            str(tmp_path / "x.csv"),
            "--output-dir",
            str(tmp_path / "run"),
            "--classes",
            "9",
            "--model-kind",
            "siglip2_base",
            "--seed",
            "52",
            "--smoke",
        ]
    )
    assert args.smoke and args.seed == 52
    assert training.FIXED["batch_size"] == 16 and training.FIXED["accumulation"] == 4
    assert training.FIXED["max_epochs"] == 20 and training.FIXED["patience"] == 4
    with pytest.raises(SystemExit):
        training.parse_args(
            [
                "--manifest",
                "x",
                "--output-dir",
                "y",
                "--classes",
                "4",
                "--model-kind",
                "siglip2_base",
                "--head-lr",
                "0.2",
            ]
        )


def test_test_manifest_rejected_before_any_labels(tmp_path, monkeypatch):
    path = manifest(tmp_path)
    data = pd.read_csv(path)
    data.loc[0, "split"] = "test"
    data.to_csv(path, index=False)
    calls = []
    original = pd.read_csv

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    with pytest.raises(ValueError, match="before parsing labels"):
        training.load_training_manifest(path, 4, smoke=False)
    assert len(calls) == 1 and calls[0]["usecols"] == ["split"]


@pytest.mark.parametrize("classes", [4, 9])
def test_smoke_has_exactly_two_examples_per_class_per_split(tmp_path, classes):
    frame = training.load_training_manifest(manifest(tmp_path, classes), classes, smoke=True)
    assert len(frame) == classes * 4
    assert frame.groupby(["split", "label_index"]).size().eq(2).all()


def test_cuda_guard_precedes_any_training_or_file_read(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing CPU"):
        training.run_training(SimpleNamespace())


def test_resume_requires_hash_and_complete_selected_artifacts(tmp_path):
    request = {"model": "fixture"}
    assert training.validate_resume_integrity(tmp_path, request) is None
    (tmp_path / "last.pt").write_bytes(b"fixture")
    with pytest.raises(RuntimeError, match="no integrity"):
        training.validate_resume_integrity(tmp_path, request)
    selected = {}
    for name in training.SELECTED_ARTIFACTS:
        (tmp_path / name).write_bytes(name.encode())
        selected[name] = sha256_file(tmp_path / name)
    sidecar = {
        "sha256": sha256_file(tmp_path / "last.pt"),
        "request_sha256": canonical_hash(request),
        "selected_artifacts": selected,
    }
    (tmp_path / "last_checkpoint.json").write_text(json.dumps(sidecar), encoding="utf-8")
    assert training.validate_resume_integrity(tmp_path, request) == sidecar
    (tmp_path / "best.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact drift"):
        training.validate_resume_integrity(tmp_path, request)


def test_replay_requires_exact_identity_labels_and_decisions(tmp_path):
    path = tmp_path / "predictions.npz"
    ids, labels = np.array(["a", "b"]), np.array([0, 1])
    probabilities = np.array([[0.7, 0.3], [0.2, 0.8]])
    np.savez(path, image_ids=ids, labels=labels, probabilities=probabilities)
    audit = training.check_replay(path, ids, labels, probabilities.copy())
    assert audit["status"] == "PASS" and audit["maximum_probability_difference"] == 0
    for altered_ids, altered_labels, altered_probabilities in [
        (ids[::-1], labels, probabilities),
        (ids, labels[::-1], probabilities),
        (ids, labels, probabilities[:, ::-1]),
        (ids, labels, probabilities + 2e-5),
    ]:
        with pytest.raises(RuntimeError, match="fixed validation replay"):
            training.check_replay(path, altered_ids, altered_labels, altered_probabilities)


def test_image_dataset_checks_bytes_before_decoding(tmp_path):
    path = tmp_path / "image.png"
    path.write_bytes(b"changed image")
    frame = pd.DataFrame([{"image_path": str(path), "image_sha256": "0" * 64, "label_index": 0}])
    dataset = training.BoundedImages(
        frame, training=False, processor={"image_mean": [0.5] * 3, "image_std": [0.5] * 3}
    )
    with pytest.raises(RuntimeError, match="image bytes changed"):
        dataset[0]


def test_request_records_new_trainer_source_and_smoke_role(tmp_path, monkeypatch):
    path = manifest(tmp_path)
    frame = training.load_training_manifest(path, 4, smoke=True)
    args = SimpleNamespace(
        manifest=path, classes=4, model_kind="siglip2_base", seed=42, workers=0, smoke=True
    )
    monkeypatch.setattr(training, "environment_evidence", lambda: {"fixture": True})
    model = SimpleNamespace(parameter_evidence=lambda: {"scope": "fixture"})
    result = training.request_evidence(
        args, frame, {"pinned": True}, {"image_mean": [0.5] * 3, "image_std": [0.5] * 3}, model, {}
    )
    assert result["role"] == "engineering_smoke" and result["max_epochs"] == 1
    assert result["test_rows_read"] == 0 and result["test_labels_read"] is False
    assert "experiments/train_polar_bounded.py" in result["implementation"]
    assert "src/hac/polar_bounded_models.py" in result["implementation"]
