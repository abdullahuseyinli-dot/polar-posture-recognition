import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from run_polar_bounded_queue import build_jobs, build_plan
from summarize_polar_bounded import summarize_task

from hac.polar import sha256_file
from hac.polar_benchmark import atomic_json, canonical_hash, probability_metrics
from hac.polar_bounded import (
    candidate_family,
    development_rows,
    implementation_hashes,
    load_probabilities,
    paired_bootstrap,
    promotion_gate,
    transitions,
    verify_bindings,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = json.loads((ROOT / "experiments/polar_bounded_protocol_20260921.json").read_text())


def fixture_frame():
    return pd.DataFrame(
        [
            {
                "image_id": f"{split}{index}",
                "image_path": f"{split}{index}.jpg",
                "split": split,
                "label": f"c{index % 4}",
                "label_index": index % 4,
                "source_group": f"{split}{index}",
            }
            for split in ("train", "val")
            for index in range(8)
        ]
    )


def save_predictions(path, frame, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        image_ids=frame.image_id.to_numpy(dtype=str),
        labels=frame.label_index.to_numpy(),
        probabilities=values,
    )


def test_queue_has_exact_fixed_budget_without_test_job(tmp_path):
    jobs = build_jobs(ROOT, tmp_path / "new", tmp_path / "parent", Path("python.exe"))
    assert len(jobs) == 17
    assert len({job["id"] for job in jobs}) == 17
    assert sum(job["id"].startswith("smoke_") for job in jobs) == 4
    fits = [job for job in jobs if job["id"].startswith("train_")]
    assert len(fits) == 12
    assert all(job["required_model"] is None for job in jobs)
    assert all(0 < job["timeout_seconds"] <= 10800 for job in jobs)
    assert all("--smoke" in job["command"] for job in jobs[:4])
    assert not any(
        "test_manifest" in arg or "evaluate_polar_final" in arg
        for job in jobs
        for arg in job["command"]
    )
    for classes in (4, 9):
        for model in ("siglip2_base", "convnextv2_base"):
            assert all(
                any(job["id"] == f"train_polar{classes}_{model}_seed{seed}" for job in fits)
                for seed in (42, 52, 62)
            )


def test_plan_cannot_write_inside_parent_or_broad_workspace(tmp_path):
    parent = ROOT / ".runs" / "fixture_parent"
    with pytest.raises(ValueError, match="separate"):
        build_plan(ROOT, parent / "child", parent, Path("python.exe"), {})
    with pytest.raises(ValueError, match="within"):
        build_plan(ROOT, tmp_path, parent, Path("python.exe"), {})


def test_source_hashes_cover_new_scripts_and_protocol():
    hashes = implementation_hashes(ROOT)
    for name in (
        "run_polar_bounded_queue.py",
        "summarize_polar_bounded.py",
        "polar_bounded_protocol_20260921.json",
    ):
        assert f"experiments/{name}" in hashes
    assert "src/hac/polar_bounded.py" in hashes


def test_development_rejects_test_before_label_read(tmp_path, monkeypatch):
    path = tmp_path / "bad.csv"
    path.write_text("split,label\ntest,sensitive\n", encoding="utf-8")
    import hac.polar_bounded as bounded

    monkeypatch.setattr(
        bounded,
        "load_development",
        lambda *args, **kwargs: pytest.fail("label parser must not run"),
    )
    with pytest.raises(ValueError, match="Test is forbidden"):
        development_rows(path, 4)


def test_prediction_alignment_and_valid_probability_guard(tmp_path):
    frame = fixture_frame().query("split == 'val'")
    values = np.full((len(frame), 4), 0.25)
    path = tmp_path / "prediction.npz"
    save_predictions(path, frame, values)
    np.testing.assert_array_equal(load_probabilities(path, frame, 4), values)
    with pytest.raises(ValueError, match="row/label"):
        load_probabilities(path, frame.iloc[::-1], 4)
    values[0, 0] = -0.1
    save_predictions(path, frame, values)
    with pytest.raises(ValueError, match="Invalid normalized"):
        load_probabilities(path, frame, 4)


def test_parent_bindings_reject_drift_and_path_escape(tmp_path):
    parent = tmp_path / "parent"
    parent.mkdir()
    path = parent / "prior.json"
    path.write_text("{}", encoding="utf-8")
    evidence = {"artifacts": {"prior.json": sha256_file(path)}}
    verify_bindings(parent, evidence)
    path.write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        verify_bindings(parent, evidence)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed"):
        verify_bindings(parent, {"artifacts": {"../outside.json": sha256_file(outside)}})


def test_finite_fusions_preserve_old_component_and_exclude_conv():
    anchor = np.array([[0.8, 0.2]])
    frozen = np.array([[0.2, 0.8]])
    incumbent = (anchor + frozen) / 2
    adapted = np.array([[0.6, 0.4]])
    convolution = np.array([[0.0, 1.0]])
    candidates = candidate_family(incumbent, anchor, adapted, convolution)
    assert len(candidates) == 5
    np.testing.assert_allclose(
        candidates["conservative_fusion"], 0.5 * anchor + 0.25 * frozen + 0.25 * adapted
    )
    np.testing.assert_array_equal(candidates["retain_incumbent"], incumbent)
    different = candidate_family(incumbent, anchor, adapted, convolution[:, ::-1])
    for key in ("conservative_fusion", "replacement_fusion", "retain_incumbent"):
        np.testing.assert_array_equal(candidates[key], different[key])


def test_gate_requires_each_seed_and_calibration_and_class_safety():
    labels = np.tile(np.arange(4), 4)
    good = np.eye(4)[labels] * 0.8 + 0.05
    prior = good.copy()
    prior[0] = [0.05, 0.85, 0.05, 0.05]
    names = [f"c{i}" for i in range(4)]
    metrics = probability_metrics(labels, good, names)
    reference = probability_metrics(labels, prior, names)
    change = transitions(labels, prior, good)
    seeds = [{"macro_f1_gain": 0.01, "net_corrections": 1}] * 3
    assert promotion_gate(
        metrics, reference, change, seeds, PROTOCOL["development_promotion_gate"]
    )["eligible"]
    for bad_seeds in (
        seeds[:2],
        seeds[:2] + [{"macro_f1_gain": -0.01, "net_corrections": 1}],
        seeds[:2] + [{"macro_f1_gain": 0.01, "net_corrections": 0}],
    ):
        assert not promotion_gate(
            metrics, reference, change, bad_seeds, PROTOCOL["development_promotion_gate"]
        )["eligible"]
    unsafe = dict(metrics, log_loss=reference["log_loss"] + 0.03)
    assert not promotion_gate(
        unsafe, reference, change, seeds, PROTOCOL["development_promotion_gate"]
    )["eligible"]


def test_bootstrap_is_paired_reproducible_and_descriptive():
    labels = np.tile(np.arange(4), 10)
    prior = np.eye(4)[labels] * 0.8 + 0.05
    candidate = prior.copy()
    prior[:4] = np.roll(prior[:4], 1, axis=1)
    groups = np.arange(len(labels)).astype(str)
    first = paired_bootstrap(labels, prior, candidate, groups, draws=20)
    assert first == paired_bootstrap(labels, prior, candidate, groups, draws=20)
    assert "not_confirmatory" in first["role"]
    assert first["row_stratified_95_interval"][0] >= 0
    identical = paired_bootstrap(labels, candidate, candidate, groups, draws=20)
    assert identical["source_group_95_interval"] == [0.0, 0.0]


def test_summary_end_to_end_and_completed_resume(tmp_path, monkeypatch):
    import summarize_polar_bounded as summarizer

    parent, run = tmp_path / "parent", tmp_path / "new"
    data = parent / "data"
    data.mkdir(parents=True)
    frame = fixture_frame().sort_values("image_id", ignore_index=True)
    manifest = data / "polar4_legacy_development_manifest.csv"
    frame.to_csv(manifest, index=False)
    validation = frame.loc[frame.split.eq("val")]
    labels = validation.label_index.to_numpy()
    good = np.eye(4)[labels] * 0.8 + 0.05
    incumbent = good.copy()
    incumbent[0] = [0.05, 0.85, 0.05, 0.05]
    analysis = parent / "polar4" / "development_analysis"
    save_predictions(analysis / "uniform_top_two_validation_validation.npz", validation, incumbent)
    save_predictions(analysis / "dinov2_base_validation.npz", validation, incumbent)
    for model in ("siglip2_base", "convnextv2_base"):
        for seed in (42, 52, 62):
            directory = run / "polar4" / model / f"seed{seed}"
            directory.mkdir(parents=True)
            request = {
                "manifest_sha256": sha256_file(manifest),
                "seed": seed,
                "model_kind": model,
                "classes": 4,
                "role": "development_adaptation",
                "initialization": "original_foundation_only",
                "test_rows_read": 0,
                "parameters": {"trainable_parameters": 16, "frozen_parameters": 32},
            }
            atomic_json(directory / "request.json", request)
            (directory / "best.pt").write_bytes(b"fixture_checkpoint")
            save_predictions(directory / "validation_predictions.npz", validation, good)
            atomic_json(
                directory / "replay_audit.json",
                {
                    "status": "PASS",
                    "labels_identical": True,
                    "predictions_identical": True,
                    "maximum_probability_difference": 0.0,
                    "checkpoint_sha256": sha256_file(directory / "best.pt"),
                },
            )
            atomic_json(
                directory / "summary.json",
                {
                    "status": "COMPLETE",
                    "request_sha256": canonical_hash(request),
                    "best_validation_macro_f1": 1.0,
                    "best_epoch": 1,
                    "epochs_completed": 1,
                    "artifacts": {
                        name: sha256_file(directory / name)
                        for name in ("best.pt", "validation_predictions.npz", "replay_audit.json")
                    },
                },
            )
    monkeypatch.setattr(summarizer, "paired_bootstrap", lambda *args, **kwargs: {"role": "fixture"})
    report = summarize_task(run, parent, 4, PROTOCOL)
    assert report["status"] == "COMPLETE"
    assert len(report["candidates"]) == 5
    assert report["selected_development_candidate"] in {
        "siglip2_adapted_3seeds",
        "convnextv2_adapted_3seeds",
    }
    assert report["test_rows_read"] == 0
    assert summarize_task(run, parent, 4, PROTOCOL) == report
    (run / "polar4" / "siglip2_base" / "seed62" / "summary.json").unlink()
    with pytest.raises(RuntimeError, match="Every declared seed"):
        summarize_task(run, parent, 4, PROTOCOL)
