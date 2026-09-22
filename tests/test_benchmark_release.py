import hashlib
import json
import shutil

import numpy as np
import pytest

from tools.check_project import ROOT, benchmark_evidence, release_artifacts
from tools.verify_benchmark_predictions import compare, verify


def test_current_benchmark_integrity():
    assert benchmark_evidence(ROOT) == {
        "confusion_based_systems_recomputed": 20,
        "prediction_sets_hashed": 32,
        "audited_rows": 35007,
        "quarantined_rows": 317,
        "retention": "prior_incumbents_unchanged",
    }
    assert release_artifacts(ROOT) == 14


def test_prediction_level_replay():
    result = verify()
    assert result["models_recomputed"] == 20
    assert result["paired_comparisons"] == 18
    assert result["seed_pairs_checked"] == 6
    assert not result["training_or_inference_performed"]


@pytest.mark.parametrize("actual,expected", [(0.2, 0.21), ([1, 2], [1]), ({"x": 1}, {"x": 2})])
def test_metric_changes_are_rejected(actual, expected):
    with pytest.raises(ValueError):
        compare(actual, expected, "tampered")


def copy_public_evidence(tmp_path):
    for name in ("results/polar_20260921", "docs/research/20260921_final_evaluation/results"):
        shutil.copytree(ROOT / name, tmp_path / name)
    for name in ("pyproject.toml", "CITATION.cff", ".zenodo.json"):
        shutil.copy2(ROOT / name, tmp_path / name)
    return tmp_path / "results/polar_20260921"


def test_changed_prediction_bytes_fail_before_decoding(tmp_path):
    folder = copy_public_evidence(tmp_path)
    path = folder / "polar9_predictions.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Public evidence bytes differ"):
        verify(tmp_path)


@pytest.mark.parametrize("field", ["summary_path", "artifacts"])
def test_evidence_path_escape_rejected(tmp_path, field):
    folder = copy_public_evidence(tmp_path)
    path = folder / "manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if field == "summary_path":
        record[field] = "../outside.json"
    else:
        record[field] = {"../../outside.npz": {"sha256": "0" * 64}}
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes"):
        verify(tmp_path)


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "cohort.csv",
        "data_audit.json",
        "polar4_predictions.npz",
        "polar9_predictions.npz",
        "quarantine.csv",
    ],
)
def test_missing_artifact_hashes_cannot_pass(tmp_path, missing):
    folder = copy_public_evidence(tmp_path)
    path = folder / "manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    if missing is None:
        record["artifacts"] = {}
    else:
        record["artifacts"].pop(missing)
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact inventory"):
        verify(tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "task",
        "candidate",
        "candidate_name",
        "classes",
        "rows",
        "prediction_manifest",
        "selection_lock",
    ],
)
def test_public_manifest_inventory_and_bindings_are_closed(tmp_path, change):
    folder = copy_public_evidence(tmp_path)
    path = folder / "manifest.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    task = record["tasks"]["polar4"]
    if change == "task":
        record["tasks"].pop("polar9")
    elif change == "candidate":
        task["candidates"].pop("prior_seed42")
    elif change == "candidate_name":
        task["candidates"]["unexpected"] = task["candidates"].pop("prior_seed42")
    elif change == "classes":
        task["class_names"].reverse()
    elif change == "rows":
        task["rows"] -= 1
    elif change == "prediction_manifest":
        record["source_prediction_manifest_sha256"] = "0" * 64
    else:
        record["selection_lock_sha256"] = "0" * 64
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        verify(tmp_path)


@pytest.mark.parametrize("change", ["task", "metric", "seed", "comparison"])
def test_rehashed_partial_summary_is_not_the_locked_result(tmp_path, change):
    folder = copy_public_evidence(tmp_path)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary_path = tmp_path / manifest["summary_path"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if change == "task":
        summary["tasks"].pop("polar9")
    elif change == "metric":
        summary["tasks"]["polar4"]["metrics"].pop("historical_ensemble")
    elif change == "seed":
        summary["tasks"]["polar4"]["seed_diagnostic"]["seeds"] = []
    else:
        summary["comparisons"].pop()
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    manifest["summary_sha256"] = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Locked final summary hash differs"):
        verify(tmp_path)


@pytest.mark.parametrize("change", ["missing_seed", "extra_array"])
def test_rehashed_prediction_archive_requires_exact_arrays(tmp_path, change):
    folder = copy_public_evidence(tmp_path)
    path = folder / "polar4_predictions.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    if change == "missing_seed":
        arrays.pop("prior_seed42")
    else:
        arrays["unexpected"] = np.array([1])
    np.savez_compressed(path, **arrays)
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"][path.name] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Prediction archive array inventory differs"):
        verify(tmp_path)
