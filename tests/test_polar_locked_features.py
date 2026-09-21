"""CPU-only fixtures for final-fit isolation and locked label-free test inference."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.metrics.pairwise import rbf_kernel
from torch import nn
from torchvision import transforms

from hac.polar import sha256_file
from hac.polar_benchmark import canonical_hash
from hac.polar_benchmark_features import MODEL_SPECS, FeatureRequest, read_development_manifest
from hac.polar_benchmark_heads import cuda_rbf_kernel, fit_group_calibrated_rbf
from hac.polar_locked_features import (
    HEAD_MODELS,
    HEAD_VIEWS,
    INFERENCE_COLUMNS,
    LOCKED_CACHE_STATUS,
    _cache_locked_with_runtime,
    cache_locked_test_features,
    calibration_partition,
    fit_locked_head,
    load_aligned_test_features,
    locked_head_spec,
    predict_locked_head,
    read_locked_inference_manifest,
    required_source_paths,
)


class TinyFeatureModel(nn.Module):
    def forward(self, pixels):
        return pixels.mean(dim=(2, 3))


@pytest.fixture
def inference(tmp_path):
    records = []
    for index in range(5):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (12, 10), (20 * index, 40, 120)).save(path)
        records.append(
            {
                "image_id": f"i{index}",
                "image_path": str(path),
                "split": "test",
                "image_sha256": sha256_file(path),
                "bbox_xmin": 1,
                "bbox_ymin": 1,
                "bbox_xmax": 10,
                "bbox_ymax": 9,
            }
        )
    path = tmp_path / "inference_manifest.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    access = {
        "selection_lock_sha256": "a" * 64,
        "barrier_sha256": "b" * 64,
        "inference_manifest": str(path),
        "inference_manifest_sha256": sha256_file(path),
        "inference_rows": len(records),
    }
    request = FeatureRequest(
        path, tmp_path / "cache", "dinov2_base", "full_frame", batch_size=2, workers=0, chunk_rows=2
    )
    return request, access


def run_tiny(request, access, *, model=None, reference=None):
    frame = read_locked_inference_manifest(request.manifest, access)
    reference = reference or {"checkpoint": {"fixture": "tiny"}, "preprocess": {"policy": "tiny"}}
    return _cache_locked_with_runtime(
        request,
        frame,
        model or TinyFeatureModel(),
        transforms.ToTensor(),
        {"fixture": "tiny"},
        {"policy": "tiny"},
        torch.device("cpu"),
        access,
        reference,
    )


def test_label_free_contract_and_immutable_development_boundary(inference):
    request, access = inference
    frame = read_locked_inference_manifest(request.manifest, access)
    assert set(frame) == INFERENCE_COLUMNS
    assert set(frame.split) == {"test"}
    with pytest.raises(ValueError, match="test is sealed"):
        read_development_manifest(request.manifest, view="full_frame")


def test_gate_hash_is_checked_before_csv_header(inference, monkeypatch):
    request, access = inference
    monkeypatch.setattr(
        pd, "read_csv", lambda *args, **kwargs: pytest.fail("CSV read before verified gate hash")
    )
    with pytest.raises(RuntimeError, match="verified access gate"):
        read_locked_inference_manifest(
            request.manifest, {**access, "inference_manifest_sha256": "f" * 64}
        )


@pytest.mark.parametrize("column", ["label", "label_index", "target", "source_label", "label_4"])
def test_inference_reader_rejects_labels_before_reading_rows(inference, column, monkeypatch):
    request, access = inference
    frame = pd.read_csv(request.manifest)
    frame[column] = "forbidden"
    frame.to_csv(request.manifest, index=False)
    access["inference_manifest_sha256"] = sha256_file(request.manifest)
    original = pd.read_csv
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    with pytest.raises(ValueError, match="label-free"):
        read_locked_inference_manifest(request.manifest, access)
    assert calls == [{"nrows": 0}]


@pytest.mark.parametrize(
    "mutation", ["split", "duplicate_id", "count", "box", "missing_image", "pixel_hash"]
)
def test_inference_invalid_cohorts_fail_closed(inference, mutation):
    request, access = inference
    frame = pd.read_csv(request.manifest)
    if mutation == "split":
        frame.loc[0, "split"] = "train"
    elif mutation == "duplicate_id":
        frame.loc[1, "image_id"] = frame.loc[0, "image_id"]
    elif mutation == "count":
        frame = frame.iloc[:-1]
    elif mutation == "box":
        frame.loc[0, "bbox_xmax"] = -1
    elif mutation == "missing_image":
        frame.loc[0, "image_path"] = "missing.png"
    else:
        frame.loc[0, "image_sha256"] = "not-a-hash"
    frame.to_csv(request.manifest, index=False)
    access["inference_manifest_sha256"] = sha256_file(request.manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        read_locked_inference_manifest(request.manifest, access)


def test_production_extraction_requires_barrier_before_manifest_or_cuda(inference, monkeypatch):
    request, access = inference

    def barrier(*args):
        raise RuntimeError("all fits have not completed")

    module = SimpleNamespace(verify_test_access=barrier)
    monkeypatch.setitem(sys.modules, "hac.polar_locked_evaluation", module)
    monkeypatch.setattr(
        "hac.polar_locked_features.read_locked_inference_manifest",
        lambda *args: pytest.fail("manifest opened before barrier"),
    )
    monkeypatch.setattr(
        "hac.polar_locked_features.require_cuda", lambda: pytest.fail("CUDA used before barrier")
    )
    with pytest.raises(RuntimeError, match="not completed"):
        cache_locked_test_features(request.manifest.parent / "lock.json", request.manifest.parent)


def test_locked_cache_has_truthful_scope_and_no_labels(inference):
    request, access = inference
    summary = run_tiny(request, access)
    assert summary["status"] == LOCKED_CACHE_STATUS
    assert summary["scope"] == "locked_test_inference"
    assert summary["test_rows_read"] == 5 and summary["test_labels_read"] is False
    assert summary["selection_lock_sha256"] == access["selection_lock_sha256"]
    assert set(pd.read_csv(request.output_dir / "rows.csv")) == {
        "row",
        "image_id",
        "split",
        "image_sha256",
    }
    assert run_tiny(request, access) == summary


def test_cache_resumes_verified_chunks_and_preserves_alignment(inference):
    request, access = inference

    class InterruptedModel(TinyFeatureModel):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, pixels):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("simulated interruption")
            return super().forward(pixels)

    with pytest.raises(RuntimeError, match="interruption"):
        run_tiny(request, access, model=InterruptedModel())
    summary = run_tiny(request, access)
    assert summary["resumed_rows"] == 2
    fresh = replace(request, output_dir=request.output_dir.parent / "fresh")
    run_tiny(fresh, access)
    np.testing.assert_array_equal(
        np.load(request.output_dir / "features.npy"), np.load(fresh.output_dir / "features.npy")
    )


def test_source_pixel_change_invalidates_cached_features(inference):
    request, access = inference
    run_tiny(request, access)
    frame = pd.read_csv(request.manifest)
    Image.new("RGB", (12, 10), "red").save(frame.loc[0, "image_path"])
    with pytest.raises(RuntimeError, match="changed"):
        run_tiny(request, access)


def test_development_preprocessing_cannot_change_at_test(inference):
    request, access = inference
    with pytest.raises(RuntimeError, match="development model/preprocessing"):
        run_tiny(
            request,
            access,
            reference={"checkpoint": {"fixture": "other"}, "preprocess": {"policy": "tiny"}},
        )


@pytest.fixture
def head_fixture(tmp_path, monkeypatch):
    records = []
    names = ["sitting", "standing", "walking", "running"]
    for index in range(100):
        records.append(
            {
                "image_id": f"d{index:03d}",
                "image_path": f"unused/{index}.jpg",
                "split": "train" if index < 80 else "val",
                "label": names[index % 4],
                "label_index": index % 4,
                "source_group": f"group-{index}",
            }
        )
    manifest = tmp_path / "development.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    output = tmp_path / "polar4" / "heads" / "dinov2_base"
    spec = {
        "C": 10.0,
        "gamma": "1/d",
        "class_weight": None,
        "views": list(HEAD_VIEWS),
        "calibration_folds": 5,
        "seed": 42,
        "output_dir": "polar4/heads/dinov2_base",
    }
    lock = {
        "output_root": str(tmp_path),
        "development_feature_root": str(tmp_path / "features"),
        "tasks": {
            "polar4": {
                "classes": 4,
                "class_names": names,
                "development_manifest": str(manifest),
                "development_manifest_sha256": sha256_file(manifest),
                "test_manifest": "DO_NOT_OPEN_THIS_TEST.csv",
                "head_fits": {"dinov2_base": spec},
            }
        },
    }
    selection = tmp_path / "final_selection_lock.json"
    selection.write_text(json.dumps(lock), encoding="utf-8")
    monkeypatch.setattr(
        "hac.polar_locked_features._read_selection_lock", lambda path: json.loads(path.read_text())
    )
    monkeypatch.setattr("hac.polar_locked_features.require_cuda", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        "hac.polar_locked_features.environment_evidence",
        lambda: {"fixture": "CPU numerical contract test"},
    )
    values = np.random.default_rng(10).normal(size=(100, 4)).astype(np.float32)
    monkeypatch.setattr(
        "hac.polar_locked_features.aligned_features",
        lambda path, frame: (values.copy(), {"fixture_view": path.name}),
    )
    monkeypatch.setattr(
        "hac.polar_benchmark_heads.cuda_rbf_kernel",
        lambda left, right, gamma: rbf_kernel(left, right, gamma=gamma),
    )
    return selection, output, lock, np.concatenate([values, values], axis=1)


@pytest.mark.parametrize(
    "field,value",
    [
        ("C", 1.0),
        ("class_weight", "balanced"),
        ("gamma", "scale"),
        ("calibration_folds", 3),
        ("seed", 99),
        ("views", ["full_frame"]),
    ],
)
def test_head_lock_rejects_any_unselected_setting(head_fixture, field, value):
    selection, output, lock, _ = head_fixture
    lock["tasks"]["polar4"]["head_fits"]["dinov2_base"][field] = value
    selection.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="selected unweighted"):
        locked_head_spec(selection, "polar4", "dinov2_base", output)


def test_source_group_calibration_covers_all_development_rows(head_fixture):
    selection, _, lock, _ = head_fixture
    frame = pd.read_csv(lock["tasks"]["polar4"]["development_manifest"])
    splits, assignments, details = calibration_partition(frame)
    assert len(splits) == 5 and len(assignments) == len(frame)
    assert set(assignments.calibration_fold) == set(range(5))
    assert all(item["group_overlap"] == 0 for item in details)
    for training, calibration in splits:
        assert not set(frame.iloc[training].source_group) & set(
            frame.iloc[calibration].source_group
        )


def test_final_head_refit_serialization_replay_and_inference_use_no_test_file(head_fixture):
    selection, output, _, features = head_fixture
    summary = fit_locked_head(selection, "polar4", "dinov2_base", output)
    assert summary["status"] == "COMPLETE" and summary["training_rows"] == 100
    assert summary["test_rows_read"] == 0 and summary["test_labels_read"] is False
    assert summary["optimization"]["class_weight"] is None
    assert summary["reload_prediction_replay"]["passed"]
    assert json.loads((output / "replay_audit.json").read_text())["predictions_identical"]
    probs = predict_locked_head(output, features[:3], expected_lock_sha256=sha256_file(selection))
    assert probs.shape == (3, 4)
    np.testing.assert_allclose(probs.sum(axis=1), 1)
    assert fit_locked_head(selection, "polar4", "dinov2_base", output) == summary
    with pytest.raises(RuntimeError, match="another selection lock"):
        predict_locked_head(output, features[:3], expected_lock_sha256="wrong")


def test_head_inference_rejects_artifact_tampering(head_fixture):
    selection, output, _, features = head_fixture
    fit_locked_head(selection, "polar4", "dinov2_base", output)
    with (output / "calibration_assignments.csv").open("a") as handle:
        handle.write("tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        predict_locked_head(output, features[:3], expected_lock_sha256=sha256_file(selection))


def test_source_lock_lists_every_new_entrypoint(tmp_path):
    paths = {path.relative_to(tmp_path).as_posix() for path in required_source_paths(tmp_path)}
    assert "src/hac/polar_locked_evaluation.py" in paths
    assert "experiments/fit_polar_locked_head.py" in paths
    assert "experiments/cache_polar_locked_test_features.py" in paths
    assert len(HEAD_MODELS) == 4


def test_alignment_requires_production_cuda_cache(inference):
    request, access = inference
    request = replace(request, output_dir=request.output_dir / "dinov2_base" / "full_frame")
    run_tiny(request, access)
    with pytest.raises(RuntimeError, match="CUDA provenance"):
        load_aligned_test_features(
            request.output_dir.parent.parent,
            "dinov2_base",
            ["i0"],
            expected_lock_sha256=access["selection_lock_sha256"],
        )


@pytest.fixture
def aligned_cache(tmp_path):
    """Synthetic files only; metadata exercises the production alignment validator."""
    root = tmp_path / "aligned"
    for offset, view in enumerate(HEAD_VIEWS):
        directory = root / "dinov3_base" / view
        directory.mkdir(parents=True)
        rows = pd.DataFrame(
            {
                "row": range(5),
                "image_id": [f"i{index}" for index in range(5)],
                "split": "test",
                "image_sha256": [f"{index:064x}" for index in range(5)],
            }
        )
        rows.to_csv(directory / "rows.csv", index=False)
        features = np.repeat((np.arange(5, dtype=np.float32) + offset * 10)[:, None], 768, axis=1)
        np.save(directory / "features.npy", features)
        contract = {
            "scope": "locked_test_inference",
            "selection_lock_sha256": "a" * 64,
            "barrier_sha256": "b" * 64,
            "manifest_sha256": "c" * 64,
            "rows": 5,
            "test_rows_read": 5,
            "test_labels_read": False,
            "device_type": "cuda",
            "model_kind": "dinov3_base",
            "model": MODEL_SPECS["dinov3_base"],
            "view": view,
        }
        provenance = {
            "status": LOCKED_CACHE_STATUS,
            "contract_sha256": canonical_hash(contract),
            "feature_shape": [5, 768],
            "artifact_sha256": {
                name: sha256_file(directory / name) for name in ("features.npy", "rows.csv")
            },
        }
        (directory / "contract.json").write_text(json.dumps(contract))
        (directory / "provenance.json").write_text(json.dumps(provenance))
    return root


def test_aligned_features_preserve_requested_subset_order_and_view_order(aligned_cache):
    result = load_aligned_test_features(
        aligned_cache, "dinov3_base", ["i4", "i1"], expected_lock_sha256="a" * 64
    )
    assert result.shape == (2, 1536)
    np.testing.assert_array_equal(result[:, 0], [4, 1])
    np.testing.assert_array_equal(result[:, 768], [14, 11])


@pytest.mark.parametrize("mutation", ["labels", "row_order", "different_barrier", "no_hashes"])
def test_alignment_rejects_wrong_semantics_even_if_file_hashes_are_consistent(
    aligned_cache, mutation
):
    directory = aligned_cache / "dinov3_base" / "person_context_10"
    contract = json.loads((directory / "contract.json").read_text())
    provenance = json.loads((directory / "provenance.json").read_text())
    if mutation in ("labels", "row_order"):
        rows = pd.read_csv(directory / "rows.csv", dtype={"image_sha256": str})
        if mutation == "labels":
            rows["label"] = "forbidden"
        else:
            rows["row"] = rows["row"][::-1].to_numpy()
        rows.to_csv(directory / "rows.csv", index=False)
        provenance["artifact_sha256"]["rows.csv"] = sha256_file(directory / "rows.csv")
    elif mutation == "different_barrier":
        contract["barrier_sha256"] = "d" * 64
    else:
        provenance["artifact_sha256"] = {}
    provenance["contract_sha256"] = canonical_hash(contract)
    (directory / "contract.json").write_text(json.dumps(contract))
    (directory / "provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(RuntimeError):
        load_aligned_test_features(
            aligned_cache, "dinov3_base", ["i4", "i1"], expected_lock_sha256="a" * 64
        )


def test_partial_test_cohort_or_online_weights_are_forbidden_even_in_internal_engine(inference):
    request, access = inference
    with pytest.raises(ValueError, match="full label-free"):
        run_tiny(replace(request, max_images=2), access)
    with pytest.raises(ValueError, match="offline weights"):
        run_tiny(replace(request, allow_download=True), access)


@pytest.mark.skipif(
    os.environ.get("POLAR_RUN_LOCKED_CUDA_SMOKE") != "1",
    reason="Synthetic CUDA check requires explicit scheduling",
)
def test_synthetic_cuda_kernel_head_checkpoint_smoke(tmp_path):
    assert torch.cuda.is_available(), "CUDA smoke must never silently fall back to CPU"
    started = time.perf_counter()
    values = np.random.default_rng(901).normal(size=(120, 8)).astype(np.float32)
    labels = np.arange(120) % 4
    groups = np.asarray([f"synthetic-{index}" for index in range(120)])
    actual = cuda_rbf_kernel(values[:12], values, 1 / 8)
    reference = rbf_kernel(values[:12].astype(np.float64), values.astype(np.float64), gamma=1 / 8)
    difference = float(np.max(np.abs(actual - reference)))
    assert difference < 2e-6
    model = fit_group_calibrated_rbf(values, labels, groups, folds=5, seed=42)
    probabilities = model.predict_proba(values[:12])
    path = tmp_path / "synthetic_head.joblib"
    joblib.dump(model, path)
    replayed = joblib.load(path).predict_proba(values[:12])
    replay_difference = float(np.max(np.abs(probabilities - replayed)))
    assert replay_difference <= 1e-7
    assert np.array_equal(probabilities.argmax(axis=1), replayed.argmax(axis=1))
    assert all(
        fit.estimator.named_steps["cudakernelsvc"].estimator_.fit_status_ == 0
        for fit in model.calibrated_classifiers_
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-7)
    print(
        json.dumps(
            {
                "status": "PASS",
                "scope": "synthetic only; zero POLAR images or labels",
                "device": torch.cuda.get_device_name(0),
                "rows": len(values),
                "folds": 5,
                "maximum_kernel_error": difference,
                "reload_probability_error": replay_difference,
                "runtime_seconds": time.perf_counter() - started,
            }
        ),
        flush=True,
    )
