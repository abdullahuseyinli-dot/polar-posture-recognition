"""Prove the declared logging-only repair works without changing fit behavior."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest

from hac.polar_benchmark import atomic_json

SCRIPT = Path(__file__).resolve().parents[1] / "tools/polar_status_io_recovery.py"
SPEC = importlib.util.spec_from_file_location("status_io_recovery_fixture", SCRIPT)
repair = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(repair)


def sharing_error(code=5):
    error = PermissionError("synthetic Windows sharing denial")
    error.winerror = code
    return error


def test_transient_status_failure_retries_identical_bytes(tmp_path, monkeypatch):
    original = Path.replace
    calls, records = [], []

    def flaky(source, destination):
        calls.append(source.read_bytes())
        if len(calls) < 3:
            raise sharing_error()
        return original(source, destination)

    monkeypatch.setattr(Path, "replace", flaky)
    with repair.status_retry(tmp_path, record=records.append):
        atomic_json(tmp_path / "progress.json", {"epoch": 14})
    assert len(calls) == 3 and len(set(calls)) == 1
    assert records[-1]["event"] == "STATUS_REPLACE_RECOVERED"
    assert json.loads((tmp_path / "progress.json").read_text()) == {"epoch": 14}
    assert Path.replace is flaky


@pytest.mark.skipif(os.name != "nt", reason="Actual Windows reader/writer race")
def test_actual_windows_live_reader_no_longer_kills_status_writer(tmp_path):
    path = tmp_path / "progress.json"
    atomic_json(path, {"epoch": 13})
    opened, release = threading.Event(), threading.Event()
    records = []

    def reader():
        with path.open("rb") as handle:
            opened.set()
            release.wait(timeout=2)
            assert json.loads(handle.read()) == {"epoch": 13}

    thread = threading.Thread(target=reader)
    thread.start()
    assert opened.wait(timeout=2)
    timer = threading.Timer(0.15, release.set)
    timer.start()
    try:
        with repair.status_retry(tmp_path, record=records.append):
            atomic_json(path, {"epoch": 14})
    finally:
        release.set()
        timer.cancel()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert any(item["event"] == "STATUS_REPLACE_RETRY" for item in records)
    assert records[-1]["event"] == "STATUS_REPLACE_RECOVERED"
    assert json.loads(path.read_text()) == {"epoch": 14}


@pytest.mark.parametrize(
    "name", ["summary.json", "request.json", "last_checkpoint.json", "final.pt"]
)
def test_model_and_evidence_writes_are_not_retried(tmp_path, monkeypatch, name):
    attempts = []

    def denied(source, target):
        attempts.append(str(target))
        raise sharing_error()

    monkeypatch.setattr(Path, "replace", denied)
    with repair.status_retry(tmp_path):
        with pytest.raises(PermissionError):
            atomic_json(tmp_path / name, {"not": "telemetry"})
    assert len(attempts) == 1


def test_status_outside_run_is_not_retried(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    attempts = []

    def denied(source, target):
        attempts.append(target)
        raise sharing_error()

    monkeypatch.setattr(Path, "replace", denied)
    with repair.status_retry(run):
        with pytest.raises(PermissionError):
            atomic_json(tmp_path / "progress.json", {})
    assert len(attempts) == 1


def test_other_permission_errors_are_not_hidden(tmp_path, monkeypatch):
    attempts = []

    def denied(source, target):
        attempts.append(target)
        raise sharing_error(999)

    monkeypatch.setattr(Path, "replace", denied)
    with repair.status_retry(tmp_path):
        with pytest.raises(PermissionError):
            atomic_json(tmp_path / "progress.json", {})
    assert len(attempts) == 1


def test_persistent_status_failure_has_a_finite_deadline(tmp_path, monkeypatch):
    def denied(source, target):
        raise sharing_error()

    monkeypatch.setattr(Path, "replace", denied)
    start = time.monotonic()
    with repair.status_retry(tmp_path, policy={**repair.RETRY_POLICY, "deadline_seconds": 0.04}):
        with pytest.raises(PermissionError):
            atomic_json(tmp_path / "progress.json", {})
    assert 0.03 <= time.monotonic() - start < 1


def test_wrapper_preserves_original_job_arguments_and_restores_popen(tmp_path, monkeypatch):
    calls, records = [], []

    def capture(command, *args, **kwargs):
        calls.append((command, kwargs))
        return "sentinel"

    monkeypatch.setattr(subprocess, "Popen", capture)
    command = ["python", "original.py", "--seed", "52"]
    plan = {"jobs": [{"id": "original_job", "command": command}]}
    manifest = tmp_path / "manifest.json"
    with repair.wrapped_children(plan, manifest, records.append):
        assert subprocess.Popen(command, cwd="unchanged") == "sentinel"
        subprocess.Popen(["taskkill", "/PID", "123"])
    assert calls[0][0][-2:] == ["--job-id", "original_job"]
    assert calls[0][1] == {"cwd": "unchanged"}
    assert calls[1][0] == ["taskkill", "/PID", "123"]
    assert records[0]["original_command"] == command
    assert plan["jobs"][0]["command"] == command
    assert subprocess.Popen is capture
