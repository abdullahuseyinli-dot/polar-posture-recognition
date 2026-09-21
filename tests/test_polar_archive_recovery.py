"""Legacy compatibility must not enable arbitrary object deserialization."""

from __future__ import annotations

import importlib.util
import io
import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "tools/polar_archive_recovery.py"
with pytest.MonkeyPatch.context() as patch:
    patch.syspath_prepend(str(SCRIPT.parent))
    SPEC = importlib.util.spec_from_file_location("archive_recovery_fixture", SCRIPT)
    repair = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(repair)


def npy(values):
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(values, dtype=object))
    return buffer.getvalue()


@pytest.mark.parametrize("rows", [2, 17, 1001, 3329])
def test_literal_decoder_preserves_all_ids_and_order_without_unpickling(rows, monkeypatch):
    ids = [f"p1_{index:05d}" for index in range(rows)]
    encoded = npy(ids)
    monkeypatch.setattr(pickle, "load", lambda *a, **k: pytest.fail("unpickling forbidden"))
    monkeypatch.setattr(pickle, "loads", lambda *a, **k: pytest.fail("unpickling forbidden"))
    actual = repair.decode_string_vector(encoded, rows)
    assert actual.dtype.kind == "U"
    np.testing.assert_array_equal(actual, ids)


@pytest.mark.parametrize(
    "values,rows",
    [
        (["a", "b"], 3),
        (["a", "a"], 2),
        (["a", 7], 2),
        (["a", None], 2),
        (["a", ""], 2),
        (["a", "x" * 257], 2),
        (["a", "x\x00y"], 2),
        ([["a", "b"]], 2),
    ],
)
def test_malformed_vectors_are_rejected(values, rows):
    with pytest.raises(ValueError):
        repair.decode_string_vector(npy(values), rows)


def test_arbitrary_reducer_is_never_executed(tmp_path):
    marker = tmp_path / "must-not-exist"

    class Hostile:
        def __reduce__(self):
            return eval, (f"open({str(marker)!r}, 'w').write('unsafe')",)

    with pytest.raises(ValueError, match="nonliteral|envelope"):
        repair.decode_string_vector(npy(["safe", Hostile()]), 2)
    assert not marker.exists()


def test_trailing_pickle_and_size_limits_fail():
    encoded = npy(["a", "b"])
    with pytest.raises(ValueError, match="Trailing"):
        repair.decode_string_vector(encoded + b"ignored", 2)
    with pytest.raises(ValueError, match="bounded"):
        repair.decode_string_vector(encoded, 10001)


@pytest.fixture
def archive(tmp_path):
    path = tmp_path / "historical.npz"
    ids = np.array(["p1_00001", "p1_00002"], dtype=object)
    values = np.array([[0.9, 0.1], [0.1, 0.9]])
    np.savez_compressed(
        path,
        image_ids=ids,
        labels_4=[0, 1],
        class_names_4=["one", "two"],
        probabilities_locked_ensemble=values,
    )
    return {
        "path": str(path),
        "sha256": repair.status_io.digest(path),
        "expected_rows": 2,
        "label_key": "labels_4",
        "probability_key": "probabilities_locked_ensemble",
        "expected_macro_f1": 1.0,
    }, values


def test_scoped_adapter_keeps_numeric_arrays_and_other_archives_unchanged(archive, tmp_path):
    spec, values = archive
    other = tmp_path / "other.npz"
    np.savez(other, image_ids=np.array(["other"], dtype=object))
    original = np.load
    events = []
    with repair.archive_adapter(spec, events.append):
        with np.load(spec["path"], allow_pickle=False) as data:
            np.testing.assert_array_equal(data["image_ids"], ["p1_00001", "p1_00002"])
            np.testing.assert_array_equal(data["probabilities_locked_ensemble"], values)
        with np.load(other, allow_pickle=False) as data:
            with pytest.raises(ValueError, match="Object arrays"):
                data["image_ids"]
        with pytest.raises(ValueError, match="allow_pickle=False"):
            np.load(spec["path"], allow_pickle=True)
    assert np.load is original and events[0]["pickle_execution"] is False


def test_checksum_and_original_comparison_gates_remain_mandatory(archive):
    import runpy

    spec, values = archive
    evaluator = runpy.run_path(str(SCRIPT.parents[1] / "experiments/evaluate_polar_locked.py"))
    with repair.archive_adapter(spec, lambda _: None):
        actual = evaluator["historical_probabilities"](
            spec, np.array(["p1_00002", "p1_00001"]), np.array([1, 0]), ["one", "two"]
        )
        np.testing.assert_array_equal(actual, values[::-1])
        for ids, labels, names, updated in [
            (["wrong", "p1_00002"], [0, 1], ["one", "two"], {}),
            (["p1_00001", "p1_00002"], [1, 0], ["one", "two"], {}),
            (["p1_00001", "p1_00002"], [0, 1], ["two", "one"], {}),
            (["p1_00001", "p1_00002"], [0, 1], ["one", "two"], {"expected_macro_f1": 0.5}),
        ]:
            with pytest.raises(RuntimeError):
                evaluator["historical_probabilities"](
                    {**spec, **updated}, np.array(ids), np.array(labels), names
                )
    with pytest.raises(RuntimeError, match="checksum"):
        repair.verified_archive({**spec, "sha256": "a" * 64})


def test_adapter_restores_numpy_after_exception(archive):
    spec, _ = archive
    original = np.load
    with pytest.raises(RuntimeError):
        with repair.archive_adapter(spec, lambda _: None):
            raise RuntimeError("synthetic")
    assert np.load is original


def test_wrapper_only_launches_original_remaining_jobs(tmp_path, monkeypatch):
    calls, events = [], []

    def capture(command, **kwargs):
        calls.append((command, kwargs))
        return "sentinel"

    monkeypatch.setattr(subprocess, "Popen", capture)
    command = ["python", "predict.py", "--selection-lock", "original.json"]
    forbidden = ["python", "train.py"]
    plan = {
        "jobs": [
            {"id": "predict_locked_candidates", "command": command},
            {"id": "train", "command": forbidden},
        ]
    }
    before = json.dumps(plan, sort_keys=True)
    with repair.wrapped_children(plan, tmp_path / "manifest.json", events.append):
        assert subprocess.Popen(command, cwd="unchanged") == "sentinel"
        with pytest.raises(RuntimeError, match="cannot launch"):
            subprocess.Popen(forbidden)
    assert events[0]["original_command"] == command
    assert calls[0][1] == {"cwd": "unchanged"}
    assert json.dumps(plan, sort_keys=True) == before and subprocess.Popen is capture
