"""The recovery fixes JSON sequence types, never scientific values or models."""

from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/polar_metadata_recovery.py"
with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(SCRIPT.parent))
    SPEC = importlib.util.spec_from_file_location("metadata_recovery_fixture", SCRIPT)
    repair = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(repair)


def metadata():
    return {
        "policy": "official_processor",
        "processor": {
            "image_mean": (0.5, 0.5, 0.5),
            "image_std": (0.5, 0.5, 0.5),
            "size": {"height": 224, "width": 224},
            "rescale_factor": 1 / 255,
        },
    }


def test_reproduces_incident_and_fixes_only_representation():
    raw = metadata()
    saved = json.loads(json.dumps(raw))
    assert raw != saved
    result, fields = repair.normalize_metadata(raw)
    assert result == saved
    assert fields == ["processor.image_mean", "processor.image_std"]
    assert isinstance(raw["processor"]["image_mean"], tuple)
    assert repair.normalize_metadata(result) == (result, [])
    assert json.dumps(result, sort_keys=True) == json.dumps(raw, sort_keys=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("image_mean", (0.4, 0.5, 0.5)),
        ("image_std", (0.6, 0.5, 0.5)),
        ("size", {"height": 256, "width": 224}),
        ("rescale_factor", 1 / 256),
        ("new_field", "unexpected"),
    ],
)
def test_real_preprocessing_changes_remain_mismatches(field, value):
    raw = metadata()
    saved = json.loads(json.dumps(raw))
    raw["processor"][field] = value
    result, _ = repair.normalize_metadata(raw)
    assert result != saved


@pytest.mark.parametrize(
    "values",
    [
        (0.5, 0.5),
        (float("nan"), 0.5, 0.5),
        (float("inf"), 0.5, 0.5),
        (True, 0.5, 0.5),
        ("0.5", 0.5, 0.5),
    ],
)
def test_invalid_normalization_metadata_fails_closed(values):
    raw = metadata()
    raw["processor"]["image_mean"] = values
    with pytest.raises(ValueError, match="three finite"):
        repair.normalize_metadata(raw)


def test_nonwhitelisted_sequence_and_policy_are_not_normalized():
    raw = metadata()
    raw["processor"]["unknown"] = (1, 2, 3)
    result, _ = repair.normalize_metadata(raw)
    assert isinstance(result["processor"]["unknown"], tuple)
    raw["policy"] = "historical_polar_eval"
    assert repair.normalize_metadata(raw) == (raw, [])


def test_wrapper_preserves_model_transform_checkpoint_objects_and_request():
    model, transform, checkpoint = object(), object(), {"weights": "unchanged"}
    raw, seen, records = metadata(), [], []
    request = SimpleNamespace(model_kind="siglip2_base", view="full_frame")

    def original(actual):
        seen.append(actual)
        return model, transform, checkpoint, raw

    actual = repair.prepare_wrapper(original, records.append)(request)
    assert seen == [request]
    assert actual[0] is model and actual[1] is transform and actual[2] is checkpoint
    assert actual[3] == json.loads(json.dumps(raw))
    assert records[0]["serialized_metadata_unchanged"]


def test_adapter_restores_original_after_exception(monkeypatch):
    import hac.polar_locked_features as features

    original = features.prepare_model_and_transform
    with pytest.raises(RuntimeError, match="synthetic"):
        with repair.metadata_adapter(lambda _: None):
            assert features.prepare_model_and_transform is not original
            raise RuntimeError("synthetic")
    assert features.prepare_model_and_transform is original


@pytest.mark.parametrize("mutation", [None, "checkpoint", "normalization"])
def test_original_extraction_guard_and_resume_remain_in_force(tmp_path, mutation):
    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image
    from torchvision.transforms import ToTensor

    from hac.polar import sha256_file
    from hac.polar_benchmark_features import FeatureRequest
    from hac.polar_locked_features import _cache_locked_with_runtime

    image = tmp_path / "synthetic.png"
    Image.new("RGB", (12, 10), (20, 40, 120)).save(image)
    frame = pd.DataFrame(
        [
            {
                "image_id": "synthetic",
                "image_path": str(image),
                "split": "test",
                "image_sha256": sha256_file(image),
                "bbox_xmin": 1,
                "bbox_ymin": 1,
                "bbox_xmax": 10,
                "bbox_ymax": 9,
            }
        ]
    )
    request = FeatureRequest(
        tmp_path / "unused.csv",
        tmp_path / "cache",
        "siglip2_base",
        "full_frame",
        batch_size=1,
        workers=0,
        chunk_rows=1,
    )
    access = {
        "inference_rows": 1,
        "selection_lock_sha256": "a" * 64,
        "barrier_sha256": "b" * 64,
        "inference_manifest_sha256": "c" * 64,
    }
    checkpoint = {"weights": "unchanged"}
    raw = metadata()
    reference = {"checkpoint": dict(checkpoint), "preprocess": json.loads(json.dumps(raw))}
    model = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten())

    def extract(preprocess):
        return _cache_locked_with_runtime(
            request,
            frame,
            model,
            ToTensor(),
            checkpoint,
            preprocess,
            torch.device("cpu"),
            access,
            reference,
        )

    # The original, unmodified guard reproduces the real tuple/list incident.
    with pytest.raises(RuntimeError, match="development model/preprocessing"):
        extract(raw)
    if mutation == "checkpoint":
        checkpoint["weights"] = "changed"
    elif mutation == "normalization":
        raw["processor"]["image_mean"] = (0.4, 0.5, 0.5)
    normalized, _ = repair.normalize_metadata(raw)
    if mutation is not None:
        with pytest.raises(RuntimeError, match="development model/preprocessing"):
            extract(normalized)
        assert not request.output_dir.exists()
    else:
        first = extract(normalized)
        features = np.load(request.output_dir / "features.npy")
        assert extract(normalized) == first
        np.testing.assert_array_equal(features, np.load(request.output_dir / "features.npy"))
        assert first["test_labels_read"] is False


def test_child_wrapper_forbids_training_and_preserves_evaluation_arguments(tmp_path, monkeypatch):
    calls, records = [], []

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return "sentinel"

    monkeypatch.setattr(subprocess, "Popen", capture)
    command = ["python", "evaluate.py", "--selection-lock", "unchanged.json"]
    training = ["python", "train.py", "--seed", "42"]
    plan = {
        "jobs": [
            {"id": "predict_locked_candidates", "command": command},
            {"id": "training", "command": training},
        ]
    }
    before = copy.deepcopy(plan)
    with repair.wrapped_children(plan, tmp_path / "manifest.json", records.append):
        assert subprocess.Popen(command, cwd="original") == "sentinel"
        with pytest.raises(RuntimeError, match="cannot launch"):
            subprocess.Popen(training)
        subprocess.Popen(["taskkill", "/PID", "123"])
    assert calls[0][0][-2:] == ["--job-id", "predict_locked_candidates"]
    assert calls[0][1] == {"cwd": "original"}
    assert calls[1][0] == ["taskkill", "/PID", "123"]
    assert records[0]["original_command"] == command and plan == before
    assert subprocess.Popen is capture


def test_prepare_accepts_relative_parent_paths_and_preserves_evidence(tmp_path, monkeypatch):
    import hac.polar_locked_evaluation as evaluation

    monkeypatch.chdir(tmp_path)
    run, root = tmp_path / "run", tmp_path / "repository"
    run.mkdir()
    parent_dir = run / "recovery/previous"
    parent_dir.mkdir(parents=True)
    parent_manifest = parent_dir / "manifest.json"
    parent_manifest.write_text("{}")
    for name in ("supervisor.stdout.log", "supervisor.stderr.log"):
        (parent_dir / name).write_text("original diagnostic")
    for name in (
        "final_selection_lock.json",
        "final_fits_verified.json",
        "test_access_gate.json",
        "queue_plan.json",
        "protected_historical_files.json",
    ):
        (run / name).write_text("{}")
    (run / "queue_status.json").write_text(json.dumps({"status": "FAILED_STOPPED"}))
    (run / "logs").mkdir()
    (run / "logs/extract_locked_test_features.log").write_text("original failure")
    for view in ("full_frame", "person_context_10"):
        directory = run / "test_features/dinov2_base" / view
        directory.mkdir(parents=True)
        for name in ("contract.json", "provenance.json", "progress.json"):
            (directory / name).write_text("{}")
    receipt = {
        "job": "verify_all_fits_and_open_gate",
        "status": "COMPLETE",
        "marker": str(run / "test_access_gate.json"),
        "marker_sha256": repair.status_io.digest(run / "test_access_gate.json"),
    }
    (run / "queue_receipts.json").write_text(json.dumps({"jobs": [receipt]}))
    plan = {"jobs": [{"id": name} for name in (receipt["job"], *repair.REMAINING_JOBS)]}
    monkeypatch.setattr(
        repair.status_io,
        "validate_manifest",
        lambda _: ({"run_directory": str(run), "repository_root": str(root)}, plan),
    )
    monkeypatch.setattr(evaluation, "verify_test_access", lambda *args: None)
    for name in repair.FILES:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic fixture source")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "status": "PASS",
                "selection_lock_sha256": repair.status_io.digest(run / "final_selection_lock.json"),
                "adapter_files": {
                    name: repair.status_io.digest(root / name) for name in repair.FILES
                },
                "checks": [
                    {"model_kind": name}
                    for name in ("dinov2_base", "dinov3_base", "siglip2_base", "convnextv2_base")
                ],
            }
        )
    )
    relative_parent = parent_manifest.relative_to(tmp_path)
    recovery = run / "recovery/new"
    result = repair.prepare(Path("run"), recovery, relative_parent, Path("preflight.json"))
    assert result["parent_manifest"] == str(parent_manifest)
    assert (recovery / "manifest.json").is_file()
    assert (
        recovery / "before_resume/recovery/previous/supervisor.stderr.log"
    ).read_text() == "original diagnostic"
    assert (run / "queue_status.json").read_text() == json.dumps({"status": "FAILED_STOPPED"})
