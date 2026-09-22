import json
import shutil

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
    assert release_artifacts(ROOT) == 10


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
    return tmp_path / "results/polar_20260921"


def test_changed_prediction_bytes_fail_before_decoding(tmp_path):
    folder = copy_public_evidence(tmp_path)
    path = folder / "polar9_predictions.npz"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="Artifact hash mismatch"):
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
