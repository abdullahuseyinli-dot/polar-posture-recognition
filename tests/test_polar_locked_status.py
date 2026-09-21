"""Actual Windows sharing tests on synthetic files, never live run evidence."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from hac.polar_benchmark import atomic_json

SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "polar_locked_status.py"
SPEC = importlib.util.spec_from_file_location("locked_status_observer", SCRIPT)
observer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(observer)


@pytest.mark.skipif(os.name != "nt", reason="Windows deny-delete sharing regression")
def test_ordinary_reader_reproduces_atomic_replace_failure(tmp_path):
    path = tmp_path / "progress.json"
    atomic_json(path, {"epoch": 13})
    with path.open("rb"):
        with pytest.raises(PermissionError):
            atomic_json(path, {"epoch": 14})
    assert observer.read_shared_json(path) == {"epoch": 13}
    atomic_json(path, {"epoch": 14})
    assert observer.read_shared_json(path) == {"epoch": 14}


def test_shared_reader_snapshot_and_windows_replacement_limitation(tmp_path):
    path = tmp_path / "progress.json"
    atomic_json(path, {"epoch": 13})
    with observer.open_shared_binary(path) as handle:
        # Empirically, this Windows MoveFileEx overwrite path still refuses an
        # open destination even with delete sharing. The recovery writer must
        # wait until our short-lived read handle closes, not assume POSIX rules.
        if os.name == "nt":
            with pytest.raises(PermissionError):
                atomic_json(path, {"epoch": 14})
        else:
            atomic_json(path, {"epoch": 14})
        assert json.loads(handle.read()) == {"epoch": 13}
    atomic_json(path, {"epoch": 14})
    assert observer.read_shared_json(path) == {"epoch": 14}


def test_shared_reader_is_read_only(tmp_path):
    path = tmp_path / "progress.json"
    atomic_json(path, {"epoch": 13})
    with observer.open_shared_binary(path) as handle:
        assert not handle.writable()
        with pytest.raises(OSError):
            handle.write(b"must not write")
    assert observer.read_shared_json(path) == {"epoch": 13}


def test_repeated_closed_snapshots_release_the_file(tmp_path):
    path = tmp_path / "progress.json"
    atomic_json(path, {"generation": 0})
    for number in range(100):
        with observer.open_shared_binary(path) as handle:
            assert json.loads(handle.read())["generation"] == number
        atomic_json(path, {"generation": number + 1})
    assert observer.read_shared_json(path) == {"generation": 100}


def test_optional_missing_file_does_not_hide_corrupt_json(tmp_path):
    path = tmp_path / "not_present.json"
    assert observer.read_shared_json(path, optional=True) is None
    with pytest.raises(FileNotFoundError):
        observer.read_shared_json(path)
    path.write_text("corrupt", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        observer.read_shared_json(path, optional=True)


def test_snapshot_is_read_only_and_has_precise_fit_counts(tmp_path):
    active = "polar9_neural_dinov2_base_seed52"
    atomic_json(tmp_path / "queue_status.json", {"status": "RUNNING", "job": active})
    atomic_json(
        tmp_path / "queue_plan.json",
        {"jobs": [{"id": active, "marker": "polar9/neural/dinov2_base/seed52/summary.json"}]},
    )
    atomic_json(
        tmp_path / "queue_receipts.json",
        {
            "jobs": [
                {"job": "smoke_polar4_dino", "status": "COMPLETE"},
                {"job": "polar4_head_dino", "status": "COMPLETE"},
                {"job": "polar4_neural_dino_seed42", "status": "COMPLETE"},
                {"job": "polar9_neural_dino_seed42", "status": "COMPLETE"},
                {"job": "polar9_neural_dino_seed52", "status": "FAILED"},
            ]
        },
    )
    progress_path = tmp_path / "polar9/neural/dinov2_base/seed52/progress.json"
    atomic_json(progress_path, {"epoch": 14, "epochs": 16})
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    result = observer.snapshot(tmp_path)
    assert result["completed"] == {
        "startup_checks": 1,
        "frozen_heads": 1,
        "polar4_neural": 1,
        "polar9_neural": 1,
    }
    assert result["active_progress"]["epoch"] == 14
    assert not result["test_gate_open"] and result["completion"] is None
    assert before == {
        p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()
    }


def test_snapshot_rejects_an_escaping_marker(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        observer.contained(tmp_path, "../outside/summary.json")
