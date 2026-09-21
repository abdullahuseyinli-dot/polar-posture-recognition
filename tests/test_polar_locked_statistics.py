import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score

from hac.polar_locked_statistics import (
    contained_path,
    error_transitions,
    file_sha256,
    holm_adjust,
    load_prediction,
    locked_promotion_gate,
    macro_f1_from_confusion,
    metrics_summary,
    paired_bootstrap,
    paired_randomization,
    paired_seed_summary,
    validate_arrays,
)


def fixture():
    labels = np.tile(np.arange(3), 12)
    candidate = np.eye(3)[labels] * 0.8 + 0.2 / 3
    reference = candidate.copy()
    reference[:6] = np.roll(reference[:6], 1, axis=1)
    return labels, reference, candidate, np.repeat(np.arange(12), 3).astype(str)


def test_metrics_match_sklearn_and_report_fixed_classes():
    labels, reference, _, _ = fixture()
    report = metrics_summary(labels, reference, ["a", "b", "c"])
    assert report["macro_f1"] == f1_score(labels, reference.argmax(1), average="macro")
    assert report["accuracy"] == accuracy_score(labels, reference.argmax(1))
    assert macro_f1_from_confusion(np.asarray(report["confusion_matrix"])) == report["macro_f1"]
    assert report["errors"] == 6
    assert set(report["per_class"]) == {"a", "b", "c"}
    assert report["selective_risk"][-1]["risk"] == 6 / 36


@pytest.mark.parametrize(
    "change", ["negative", "nan", "infinite", "sum", "fractional", "class", "boolean"]
)
def test_invalid_prediction_arrays_fail(change):
    labels, _, candidate, _ = fixture()
    if change == "negative":
        candidate[0, 0] = -0.1
    elif change == "nan":
        candidate[0, 0] = np.nan
    elif change == "infinite":
        candidate[0, 0] = np.inf
    elif change == "sum":
        candidate[0] *= 0.5
    elif change == "fractional":
        labels = labels.astype(float)
        labels[0] = 0.5
    elif change == "class":
        labels[labels == 2] = 1
    else:
        labels = labels.astype(bool)
    with pytest.raises(ValueError):
        validate_arrays(labels, candidate)


def test_load_prediction_checks_hash_rows_labels_and_containment(tmp_path):
    labels, _, candidate, _ = fixture()
    ids = np.asarray([f"row{i}" for i in range(len(labels))])
    path = tmp_path / "candidate.npz"
    np.savez_compressed(path, image_ids=ids, labels=labels, probabilities=candidate)
    record = {"path": path.name, "sha256": file_sha256(path)}
    np.testing.assert_array_equal(
        load_prediction(tmp_path, record, image_ids=ids, labels=labels, classes=3), candidate
    )
    with pytest.raises(ValueError, match="IDs"):
        load_prediction(tmp_path, record, image_ids=ids[::-1], labels=labels, classes=3)
    with pytest.raises(ValueError, match="labels"):
        load_prediction(tmp_path, record, image_ids=ids, labels=np.roll(labels, 1), classes=3)
    with pytest.raises(ValueError, match="bytes"):
        load_prediction(
            tmp_path, {**record, "sha256": "0" * 64}, image_ids=ids, labels=labels, classes=3
        )
    child = tmp_path / "child"
    child.mkdir()
    with pytest.raises(ValueError, match="escapes"):
        contained_path(child, path)


def test_error_transitions_counts_and_zero_error_case():
    labels, reference, candidate, _ = fixture()
    result = error_transitions(labels, reference, candidate)
    assert result["rescued"] == result["net_corrections"] == 6
    assert result["harmed"] == result["both_wrong"] == 0
    perfect = error_transitions(labels, candidate, candidate)
    assert perfect["error_jaccard"] == 1.0
    assert perfect["candidate_error_given_reference_error"] is None


def test_paired_bootstrap_identical_predictions_and_reproducibility():
    labels, reference, _, groups = fixture()
    result = paired_bootstrap(labels, reference, reference, groups, draws=50)
    assert result == paired_bootstrap(labels, reference, reference, groups, draws=50)
    for name in ("macro_f1", "accuracy"):
        assert result[name]["delta"]["row_stratified_ci95"] == [0.0, 0.0]
        assert result[name]["delta"]["source_group_ci95"] == [0.0, 0.0]
    assert result["source_groups"] == 12


def test_paired_bootstrap_records_conditioning_and_exact_delta():
    labels, reference, candidate, groups = fixture()
    result = paired_bootstrap(labels, reference, candidate, groups, draws=100)
    expected = f1_score(labels, candidate.argmax(1), average="macro") - f1_score(
        labels, reference.argmax(1), average="macro"
    )
    assert result["macro_f1"]["delta"]["point_estimate"] == expected
    assert result["source_group_conditioning"] == "all_declared_classes_present"
    with pytest.raises(ValueError):
        paired_bootstrap(labels, reference, candidate, groups[:-1], draws=5)


def test_group_randomization_not_independent_row_swap():
    labels = np.array([0, 0, 1, 1])
    candidate = np.eye(2)[labels]
    reference = candidate[:, ::-1]
    groups = np.array(["a", "a", "b", "b"])
    group = paired_randomization(labels, reference, candidate, groups, draws=3000)
    rows = paired_randomization(labels, reference, candidate, np.arange(4), draws=3000)
    assert 0.45 < group["p_value"] < 0.55
    assert 0.09 < rows["p_value"] < 0.16
    assert group["affected_groups"] == 2
    reverse = paired_randomization(labels, candidate, reference, groups, draws=3000)
    assert reverse["p_value"] == group["p_value"]
    assert reverse["observed_delta"] == -group["observed_delta"]


def test_randomization_identity_and_plus_one_never_zero():
    labels, reference, candidate, groups = fixture()
    assert paired_randomization(labels, reference, reference, groups, draws=50)["p_value"] == 1
    result = paired_randomization(labels, reference, candidate, groups, draws=100)
    assert result["p_value"] == (result["extreme_draws"] + 1) / 101
    assert result["p_value"] > 0


def test_holm_exact_order_ties_and_bad_values():
    assert holm_adjust([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm_adjust([0.5, 0.5, 0.5]) == [1.0, 1.0, 1.0]
    for values in ([], [-0.1], [np.nan], [1.1]):
        with pytest.raises(ValueError):
            holm_adjust(values)


def test_seed_sensitivity_requires_every_seed_and_does_not_select_best():
    labels, reference, candidate, _ = fixture()
    predictions = {str(seed): candidate for seed in (42, 52, 62)}
    predictions["62"] = reference
    report = paired_seed_summary(labels, reference, predictions)
    assert len(report["seeds"]) == 3
    assert report["all_seeds_positive_macro_f1_gain"] is False
    assert report["all_seeds_positive_net_corrections"] is False
    with pytest.raises(ValueError):
        paired_seed_summary(labels, reference, {"42": candidate})


@pytest.mark.parametrize("bad_groups", [["a", None], ["", "b"], ["a", np.nan]])
def test_missing_source_groups_rejected(bad_groups):
    y = np.array([0, 1])
    p = np.eye(2)
    with pytest.raises(ValueError):
        paired_randomization(y, p, p, np.asarray(bad_groups, dtype=object), draws=2)


def test_seed_sensitivity_uses_matching_reference_not_favorable_ensemble():
    labels, reference, candidate, _ = fixture()
    old = {str(seed): reference for seed in (42, 52, 62)}
    new = {str(seed): candidate for seed in (42, 52, 62)}
    old["62"] = candidate
    report = paired_seed_summary(labels, old, new)
    assert report["seeds"][-1]["macro_f1_gain"] == 0
    assert report["all_seeds_positive_macro_f1_gain"] is False
    assert report["role"].startswith("matching_reference")
    with pytest.raises(ValueError, match="matching"):
        paired_seed_summary(labels, {"42": reference}, new)


def test_positive_f1_does_not_override_safety_gates(analysis_module):
    labels, reference, candidate, _ = fixture()
    old = metrics_summary(labels, reference, ["a", "b", "c"])
    new = metrics_summary(labels, candidate, ["a", "b", "c"])
    comparison = {
        "transitions": error_transitions(labels, reference, candidate),
        "bootstrap": {"macro_f1": {"delta": {"source_group_ci95": [0.05, 0.3]}}},
        "holm_p_value": 0.001,
    }
    seeds = {"all_seeds_positive_macro_f1_gain": True, "all_seeds_positive_net_corrections": True}
    report = locked_promotion_gate(new, old, comparison, seeds, analysis_module.GATES)
    assert report["all_checks_passed"] is True
    new["log_loss"] = old["log_loss"] + 0.021
    report = locked_promotion_gate(new, old, comparison, seeds, analysis_module.GATES)
    assert report["all_checks_passed"] is False
    assert "nll_increase_within_limit" in report["failed_checks"]
    assert report["checks"]["positive_macro_f1"] is True


@pytest.fixture
def analysis_module():
    path = Path(__file__).resolve().parents[1] / "experiments" / "analyze_polar_locked.py"
    spec = importlib.util.spec_from_file_location("polar_locked_analysis_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def locked_fixture(tmp_path, monkeypatch, analysis_module):
    module = analysis_module
    run = tmp_path / "synthetic_locked_run"
    evaluation = run / "evaluation"
    evaluation.mkdir(parents=True)
    gate_path = run / "test_access_gate.json"
    gate_path.write_text("{}\n", encoding="utf-8")
    lock = {
        "output_root": str(run),
        "tasks": {},
        "statistics": {
            "bootstrap_draws": 5000,
            "randomization_draws": 10000,
            "seed": 20260921,
            "alpha": 0.05,
            "multiplicity": "one_Holm_family_all_18_comparisons",
            "randomization_unit": "detected_source_group",
            "alternative": "two_sided",
            "gates": dict(module.GATES),
            "comparisons": [],
        },
        "test_selection_forbidden": True,
        "automatic_search_after_test": False,
        "historical_exposure": {"four_class": "exposed", "nine_class": "partly_exposed"},
    }
    manifest = {"tasks": {}, "gate_sha256": file_sha256(gate_path)}
    for task, classes in (("polar9", 9), ("polar4", 4)):
        labels = np.tile(np.arange(classes), 6)
        ids = np.asarray([f"{task}_image{i:03}" for i in range(len(labels))])
        groups = [f"{task}_group{i:03}" for i in range(len(labels))]
        names = [f"class{i}" for i in range(classes)]
        frame = pd.DataFrame(
            {
                "image_id": ids,
                "image_path": [f"image{i}.jpg" for i in range(len(labels))],
                "split": "test",
                "label": [names[index] for index in labels],
                "label_index": labels,
                "source_group": groups,
                "bbox_xmin": 0,
                "bbox_ymin": 0,
                "bbox_xmax": 10,
                "bbox_ymax": 10,
                "image_sha256": [f"test_hash_{i}" for i in range(len(labels))],
            }
        )
        test_manifest = run / f"{task}_test.csv"
        development_manifest = run / f"{task}_development.csv"
        frame.to_csv(test_manifest, index=False)
        development = frame.iloc[:classes].copy()
        for column in ("image_id", "source_group", "image_sha256"):
            development[column] = "development_" + development[column]
        development.split = "train"
        development.to_csv(development_manifest, index=False)
        spec = {
            "test_manifest": str(test_manifest),
            "test_manifest_sha256": file_sha256(test_manifest),
            "test_rows": len(labels),
            "development_manifest": str(development_manifest),
            "development_manifest_sha256": file_sha256(development_manifest),
            "classes": classes,
            "class_names": names,
            "candidates": sorted(module.expected_candidates(task)),
            "nominated_candidate": module.NOMINEE,
        }
        lock["tasks"][task] = spec
        strongest = "adapted_dinov2" if classes == 9 else "frozen_siglip2_base"
        for name in spec["candidates"]:
            if name != module.NOMINEE:
                lock["statistics"]["comparisons"].append(
                    {
                        "id": f"{task}__vs__{name}",
                        "task": task,
                        "candidate": module.NOMINEE,
                        "reference": name,
                        "primary": name in {"prior_incumbent", strongest, "historical_ensemble"},
                    }
                )
        directory = evaluation / task
        directory.mkdir()
        groups_path = directory / "source_groups.json"
        groups_path.write_text(json.dumps(groups), encoding="utf-8")
        record = {
            "rows": len(labels),
            "manifest_sha256": spec["test_manifest_sha256"],
            "source_groups_path": str(groups_path),
            "source_groups_sha256": file_sha256(groups_path),
            "class_names": names,
            "candidates": {},
        }
        probabilities = np.eye(classes)[labels] * 0.8 + 0.2 / classes
        for name in sorted(module.expected_candidates(task) | module.SEED_CANDIDATES):
            path = directory / f"{name}.npz"
            np.savez_compressed(path, image_ids=ids, labels=labels, probabilities=probabilities)
            record["candidates"][name] = {
                "path": str(path),
                "sha256": file_sha256(path),
                "kind": "seed_diagnostic"
                if name in module.SEED_CANDIDATES
                else "locked_comparison",
            }
        manifest["tasks"][task] = record
    lock_path = run / "selection_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    manifest["selection_lock_sha256"] = file_sha256(lock_path)
    manifest_path = evaluation / "prediction_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(module, "verify_selection_lock", lambda _: lock)
    monkeypatch.setattr(module, "verify_test_access", lambda *_: {"status": "SYNTHETIC_GATE"})
    return module, lock, lock_path, manifest, manifest_path


def test_full_comparison_contract_cannot_be_reduced_or_reselected(locked_fixture):
    module, lock, *_ = locked_fixture
    assert len(module.validate_statistical_contract(lock)) == 18
    assert sum(item["primary"] for item in lock["statistics"]["comparisons"]) == 5
    lock["statistics"]["comparisons"].pop()
    with pytest.raises(ValueError, match="18-comparison"):
        module.validate_statistical_contract(lock)


def test_locked_input_schema_and_test_gate_precede_prediction_reads(locked_fixture, monkeypatch):
    module, _, lock_path, _, manifest_path = locked_fixture
    inputs = module.load_locked_inputs(lock_path, manifest_path)
    assert len(inputs["tasks"]["polar9"]["probabilities"]) == 16

    def fail_gate(*_):
        raise RuntimeError("Gate not yet open")

    def forbidden_read(*_, **__):
        pytest.fail("No test CSV/NPZ may be opened before the gate")

    monkeypatch.setattr(module, "verify_test_access", fail_gate)
    monkeypatch.setattr(module.pd, "read_csv", forbidden_read)
    monkeypatch.setattr(module, "load_prediction", forbidden_read)
    with pytest.raises(RuntimeError, match="Gate"):
        module.load_locked_inputs(lock_path, manifest_path)


@pytest.mark.parametrize("tampering", ["sha", "groups", "seed", "role", "lock", "gate", "formula"])
def test_locked_prediction_manifest_fails_closed(locked_fixture, tampering):
    module, _, lock_path, manifest, manifest_path = locked_fixture
    record = manifest["tasks"]["polar9"]
    if tampering == "sha":
        record["candidates"]["prior_incumbent"]["sha256"] = "0" * 64
    elif tampering == "groups":
        path = Path(record["source_groups_path"])
        groups = json.loads(path.read_text())
        groups.reverse()
        path.write_text(json.dumps(groups), encoding="utf-8")
        record["source_groups_sha256"] = file_sha256(path)
    elif tampering == "seed":
        record["candidates"].pop("prior_seed62")
    elif tampering == "role":
        record["candidates"]["prior_incumbent"]["kind"] = "seed_diagnostic"
    elif tampering == "lock":
        manifest["selection_lock_sha256"] = "0" * 64
    elif tampering == "gate":
        manifest["gate_sha256"] = "0" * 64
    else:
        item = record["candidates"]["replacement_fusion"]
        path = Path(item["path"])
        with np.load(path) as saved:
            ids, labels, probabilities = [
                saved[name].copy() for name in ("image_ids", "labels", "probabilities")
            ]
        probabilities[0] = np.roll(probabilities[0], 1)
        np.savez_compressed(path, image_ids=ids, labels=labels, probabilities=probabilities)
        item["sha256"] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        module.load_locked_inputs(lock_path, manifest_path)


def test_synthetic_final_analysis_preserves_nominee_and_all_comparisons(
    locked_fixture, monkeypatch
):
    module, _, lock_path, _, manifest_path = locked_fixture

    # Exercise production budgets as arguments; synthetic resampling is deliberately short.
    def bootstrap(*args, draws, seed):
        assert draws == 5000 and seed == 20260921
        return paired_bootstrap(*args, draws=5, seed=seed)

    def randomization(*args, draws, seed):
        assert draws == 10000 and seed == 20260921
        return paired_randomization(*args, draws=5, seed=seed)

    monkeypatch.setattr(module, "paired_bootstrap", bootstrap)
    monkeypatch.setattr(module, "paired_randomization", randomization)
    output = lock_path.parent / "analysis"
    summary = module.run_analysis(lock_path, manifest_path, output)
    assert summary["status"] == module.STATUS
    assert len(summary["comparisons"]) == 18
    assert all(row["holm_p_value"] == 1 for row in summary["comparisons"])
    assert summary["automatic_model_promotion"] is False
    for result in summary["tasks"].values():
        assert result["nominated_candidate"] == "conservative_fusion"
        assert result["promotion_gate"]["all_checks_passed"] is False
        assert result["nominee_unchanged_after_test"] is True
        assert len(result["metrics"]) == 10
    for artifact in (
        "report.md",
        "metrics.csv",
        "comparisons.csv",
        "nominee_confusion.png",
        "comparison_f1.png",
    ):
        assert (output / artifact).stat().st_size > 0
    assert module.run_analysis(lock_path, manifest_path, output) == summary
    (output / "metrics.csv").write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact changed"):
        module.run_analysis(lock_path, manifest_path, output)
