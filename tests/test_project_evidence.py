import json
import shutil

import pytest

from tools.check_project import (
    ROOT,
    confusion_metrics,
    evidence,
    figures,
    metadata,
    navigation,
    preserved_files,
    within,
)


def test_public_evidence_and_identity():
    assert metadata(ROOT) == "1.1.0"
    assert evidence(ROOT)["confusion_based_systems_recomputed"] == 8
    assert preserved_files(ROOT) == 330


def test_current_navigation_and_figures():
    assert navigation(ROOT) > 70
    assert figures(ROOT) == 6


def test_confusion_metric_arithmetic():
    scores = confusion_metrics([[8, 1], [2, 9]])
    assert scores["rows"] == 20
    assert scores["errors"] == 3
    assert scores["accuracy"] == 0.85
    assert scores["macro_f1"] == pytest.approx((16 / 19 + 18 / 21) / 2)


@pytest.mark.parametrize("matrix", [[], [[1, 0]], [[0, 0], [0, 0]], [[-1, 2], [1, 2]]])
def test_invalid_matrices_fail(matrix):
    with pytest.raises(ValueError):
        confusion_metrics(matrix)


def test_path_escape_fails(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        within(tmp_path, "../outside.json")


def test_changed_confusion_is_not_silently_accepted(tmp_path):
    shutil.copytree(ROOT / "results", tmp_path / "results")
    path = tmp_path / "results/polar_test_confusions.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["locked_ensemble"]["counts"][0][0] += 1
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="population"):
        evidence(tmp_path)
