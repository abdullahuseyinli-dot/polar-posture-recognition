"""CPU synthetic safety tests: no real held-out images or labels are opened."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hac.polar import sha256_file
from hac.polar_benchmark import canonical_hash, lock_json
from hac.polar_locked_evaluation import (
    LOCK_STATUS,
    expected_fits,
    fusion_candidates,
    inside,
    normalized,
    open_test_gate,
    validate_test_rows,
    verify_fit,
    verify_selection_lock,
)


def test_no_fit_can_open_test_without_summary(tmp_path, monkeypatch):
    import hac.polar_locked_evaluation as module

    lock_path = tmp_path / "final_selection_lock.json"
    lock_path.write_text("{}")
    fake = {
        "tasks": {
            "polar9": {
                "neural_fits": {
                    "x": {"seeds": [42], "output_dir_pattern": "polar9/neural/x/seed{seed}"}
                },
                "head_fits": {},
            }
        }
    }
    monkeypatch.setattr(module, "verify_selection_lock", lambda _: fake)
    monkeypatch.setattr(module, "fit_expectation", lambda *_: {})
    opened = []
    monkeypatch.setattr(pd, "read_csv", lambda *args, **kwargs: opened.append(args))
    with pytest.raises(FileNotFoundError):
        open_test_gate(lock_path)
    assert opened == []
    assert not (tmp_path / "test_access_gate.json").exists()


def test_selection_lock_never_parses_or_requires_test_file(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("source fixture")
    dev = tmp_path / "development.csv"
    dev.write_text("development fixture")
    value = {
        "status": LOCK_STATUS,
        "output_root": str(tmp_path),
        "repository_root": str(tmp_path),
        "implementation": {"source.py": sha256_file(source)},
        "tasks": {},
    }
    for classes in (4, 9):
        value["tasks"][f"polar{classes}"] = {
            "classes": classes,
            "development_manifest": str(dev),
            "development_manifest_sha256": sha256_file(dev),
            "test_manifest": str(tmp_path / "DO_NOT_OPEN.csv"),
            "nominated_candidate": "conservative_fusion",
        }
    path = tmp_path / "lock.json"
    lock_json(path, value)
    assert verify_selection_lock(path) == value
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="changed"):
        verify_selection_lock(path)


def complete_fit_fixture(directory, *, smoke=False):
    request = {
        "selection_lock_sha256": "abc",
        "smoke": smoke,
        "epochs": 2,
        "test_rows_read": 0,
        "test_labels_read": False,
        "task": "polar9",
    }
    lock_json(directory / "request.json", request)
    (directory / "final.pt").write_bytes(b"synthetic checkpoint; not a torch pickle")
    lock_json(directory / "replay_audit.json", {"status": "PASS", "predictions_identical": True})
    summary = {
        "status": "COMPLETE",
        "role": "final_train_plus_validation_fit",
        "request_sha256": canonical_hash(request),
        "epochs_completed": 2,
        "test_rows_read": 0,
        "test_labels_read": False,
        "artifacts": {
            name: sha256_file(directory / name) for name in ("final.pt", "replay_audit.json")
        },
    }
    lock_json(directory / "summary.json", summary)
    return request, summary


def test_fit_integrity_and_component_identity(tmp_path):
    complete_fit_fixture(tmp_path)
    assert verify_fit(tmp_path, "abc", "neural", {"task": "polar9"})["kind"] == "neural"
    with pytest.raises(RuntimeError, match="identity"):
        verify_fit(tmp_path, "abc", "neural", {"task": "polar4"})
    with pytest.raises(RuntimeError, match="bound"):
        verify_fit(tmp_path, "different", "neural")
    (tmp_path / "final.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="changed"):
        verify_fit(tmp_path, "abc", "neural")


def test_engineering_smoke_cannot_unlock_test(tmp_path):
    complete_fit_fixture(tmp_path, smoke=True)
    with pytest.raises(RuntimeError, match="smoke"):
        verify_fit(tmp_path, "abc", "neural")


def test_path_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        inside(tmp_path, "../somewhere")


def test_fixed_fusion_preserves_anchor_without_label_argument():
    components = {
        "adapted_dinov2": np.array([[0.9, 0.1], [0.1, 0.9]]),
        "frozen_siglip2_base": np.array([[0.5, 0.5], [0.8, 0.2]]),
        "adapted_siglip2": np.array([[0.7, 0.3], [0.2, 0.8]]),
    }
    results = fusion_candidates(components, classes=9)
    np.testing.assert_allclose(results["conservative_fusion"], [[0.75, 0.25], [0.3, 0.7]])
    np.testing.assert_allclose(results["prior_incumbent"], [[0.7, 0.3], [0.45, 0.55]])


@pytest.mark.parametrize("bad", [[[0.0, 0.0]], [[float("nan"), 1.0]], [[-1.0, 2.0]], [[2.0, 3.0]]])
def test_invalid_probabilities_fail_closed(bad):
    with pytest.raises(ValueError):
        normalized(np.array(bad))


def test_test_cohort_source_leakage_rejected():
    test = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "image_path": ["p", "q"],
            "split": ["test"] * 2,
            "label": ["x", "y"],
            "label_index": [0, 1],
            "source_group": ["g", "h"],
            "image_sha256": ["1", "2"],
            "bbox_xmin": [0, 0],
            "bbox_ymin": [0, 0],
            "bbox_xmax": [1, 1],
            "bbox_ymax": [1, 1],
        }
    )
    dev = pd.DataFrame({"image_id": ["z"], "source_group": ["g"], "image_sha256": ["3"]})
    with pytest.raises(ValueError, match="source"):
        validate_test_rows(test, {"test_rows": 2, "classes": 2, "class_names": ["x", "y"]}, dev)
    dev.source_group = ["different"]
    assert (
        len(
            validate_test_rows(test, {"test_rows": 2, "classes": 2, "class_names": ["x", "y"]}, dev)
        )
        == 2
    )


def test_queue_order_and_budget_are_finite(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "experiments" / "run_polar_locked_queue.py"
    module_spec = importlib.util.spec_from_file_location("locked_queue_fixture", script)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    tasks = {}
    for classes in (9, 4):
        task = f"polar{classes}"
        models = (
            ["dinov2_base", "siglip2_base", "convnextv2_base"]
            if classes == 9
            else ["siglip2_base", "convnextv2_base"]
        )
        tasks[task] = {
            "neural_fits": {
                m: {"seeds": [42, 52, 62], "output_dir_pattern": f"{task}/neural/{m}/seed{{seed}}"}
                for m in models
            },
            "head_fits": {
                m: {"output_dir": f"{task}/heads/{m}"}
                for m in ["dinov2_base", "siglip2_base", "dinov3_base", "convnextv2_base"]
            },
        }
    fake = {
        "repository_root": str(tmp_path),
        "tasks": tasks,
        "resource_policy": {"job_timeout_seconds": 43200},
    }
    path = tmp_path / "lock.json"
    path.write_text("fixture")
    monkeypatch.setattr(module, "verify_selection_lock", lambda _: fake)
    plan = module.build_plan(path)
    assert len(list(expected_fits(fake))) == 23
    assert len(plan["jobs"]) == 32
    assert sum(job["stage"] == "engineering_smoke" for job in plan["jobs"]) == 5
    assert sum(job["stage"] == "final_fit" for job in plan["jobs"]) == 23
    assert plan["jobs"][28]["stage"] == "test_gate"
    assert all(job["stage"] not in {"final_fit", "engineering_smoke"} for job in plan["jobs"][28:])


def test_full_23_fit_barrier_and_test_gate_are_bound(tmp_path, monkeypatch):
    import hac.polar_locked_evaluation as module

    lock = {"output_root": str(tmp_path), "tasks": {}, "historical_exposure": {"fixture": True}}
    for classes in (9, 4):
        task = f"polar{classes}"
        development = pd.DataFrame(
            {
                "image_id": [f"dev{n}" for n in range(classes)],
                "source_group": [f"devgroup{n}" for n in range(classes)],
                "image_sha256": [f"devhash{n}" for n in range(classes)],
            }
        )
        test = pd.DataFrame(
            {
                "image_id": [f"test{n}" for n in range(classes)],
                "image_path": [f"pixel{n}" for n in range(classes)],
                "split": ["test"] * classes,
                "source_group": [f"testgroup{n}" for n in range(classes)],
                "image_sha256": [f"testhash{n}" for n in range(classes)],
                "label_index": list(range(classes)),
                "label": [f"class{n}" for n in range(classes)],
                "bbox_xmin": [0] * classes,
                "bbox_ymin": [0] * classes,
                "bbox_xmax": [1] * classes,
                "bbox_ymax": [1] * classes,
            }
        )
        dev_path, test_path = tmp_path / f"{task}_dev.csv", tmp_path / f"{task}_test.csv"
        development.to_csv(dev_path, index=False)
        test.to_csv(test_path, index=False)
        models = (
            ["dinov2_base", "siglip2_base", "convnextv2_base"]
            if classes == 9
            else ["siglip2_base", "convnextv2_base"]
        )
        lock["tasks"][task] = {
            "classes": classes,
            "class_names": test.label.tolist(),
            "development_rows": len(development),
            "development_manifest": str(dev_path),
            "development_manifest_sha256": sha256_file(dev_path),
            "test_manifest": str(test_path),
            "test_manifest_sha256": sha256_file(test_path),
            "test_rows": len(test),
            "neural_fits": {
                m: {
                    "seeds": [42, 52, 62],
                    "epochs": 2,
                    "output_dir_pattern": f"{task}/neural/{m}/seed{{seed}}",
                }
                for m in models
            },
            "head_fits": {m: {"output_dir": f"{task}/heads/{m}"} for m in module.MODELS},
        }
    path = tmp_path / "final_selection_lock.json"
    lock_json(path, lock)
    digest = sha256_file(path)
    monkeypatch.setattr(module, "verify_selection_lock", lambda _: lock)
    for _, relative, kind in expected_fits(lock):
        directory = tmp_path / relative
        request = {
            **module.fit_expectation(lock, relative, kind),
            "selection_lock_sha256": digest,
            "smoke": False,
            "test_rows_read": 0,
            "test_labels_read": False,
        }
        lock_json(directory / "request.json", request)
        artifact = "final.pt" if kind == "neural" else "head.joblib"
        (directory / artifact).write_bytes(b"fixture")
        lock_json(
            directory / "replay_audit.json", {"status": "PASS", "predictions_identical": True}
        )
        lock_json(
            directory / "summary.json",
            {
                "status": "COMPLETE",
                "role": "final_train_plus_validation_fit",
                "request_sha256": canonical_hash(request),
                "test_rows_read": 0,
                "test_labels_read": False,
                "epochs_completed": request.get("epochs"),
                "artifacts": {
                    n: sha256_file(directory / n) for n in (artifact, "replay_audit.json")
                },
            },
        )
    gate = open_test_gate(path)
    assert gate["official_open_count_this_locked_phase"] == 1
    assert set(pd.read_csv(gate["inference_manifest"]).columns) == set(module.INFERENCE_COLUMNS)
    assert module.verify_test_access(path, tmp_path) == gate
    assert open_test_gate(path) == gate
    _, relative, _ = next(expected_fits(lock))
    (tmp_path / relative / "final.pt").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        module.verify_test_access(path, tmp_path)
