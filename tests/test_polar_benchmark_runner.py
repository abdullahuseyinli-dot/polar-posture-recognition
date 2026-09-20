import json
from pathlib import Path

import pytest
from run_polar_benchmark_queue import build_plan, verify_data_audit

from hac.polar import sha256_file
from hac.polar_benchmark import atomic_json


def test_queue_is_finite_and_never_evaluates_test(tmp_path):
    root = Path(__file__).resolve().parents[1]
    plan = build_plan(root, tmp_path, Path("python.exe"))
    assert plan["test_access"] == "FORBIDDEN"
    assert len(plan["jobs"]) == 25
    assert len({job["id"] for job in plan["jobs"]}) == len(plan["jobs"])
    for job in plan["jobs"]:
        assert 0 < job["timeout_seconds"] <= 86400
        command = job["command"]
        assert not any("test_manifest" in arg or "evaluate_polar_final" in arg for arg in command)
    assert plan["jobs"][0]["id"] == "nine_class_cuda_smoke"
    training_jobs = [job for job in plan["jobs"] if job["id"].startswith("train_polar9_")]
    assert len(training_jobs) == 2
    assert all("--seed" in job["command"] and "42" in job["command"] for job in training_jobs)


def test_queue_rejects_changed_data_after_audit(tmp_path):
    artifact = tmp_path / "development.csv"
    artifact.write_text("image_id,split\nx,train\n", encoding="utf-8")
    audit = tmp_path / "audit.json"
    atomic_json(
        audit,
        {
            "status": "LOCKED_BEFORE_NEW_BENCHMARK_FITTING",
            "artifacts": {artifact.name: {"sha256": sha256_file(artifact)}},
        },
    )
    assert (
        verify_data_audit({"data_audit": str(audit)})["status"]
        == "LOCKED_BEFORE_NEW_BENCHMARK_FITTING"
    )
    artifact.write_text("image_id,split\nx,test\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differs from audit"):
        verify_data_audit({"data_audit": str(audit)})


def test_queue_rejects_audit_path_escape(tmp_path):
    audit = tmp_path / "audit.json"
    atomic_json(
        audit,
        {
            "status": "LOCKED_BEFORE_NEW_BENCHMARK_FITTING",
            "artifacts": {"../outside": {"sha256": "fake"}},
        },
    )
    with pytest.raises(RuntimeError):
        verify_data_audit({"data_audit": str(audit)})


def test_training_has_no_final_checkpoint_initialization_switch():
    # An explicit contract check alongside real CUDA smoke and checkpoint replay.
    import train_polar_benchmark

    source = Path(train_polar_benchmark.__file__).read_text(encoding="utf-8")
    assert "original_foundation_only" in source
    assert 'parser.add_argument("--initial-checkpoint"' not in source
    assert '"replay_audit.json"' in source


def test_foundation_snapshot_requires_only_used_safe_files(tmp_path, monkeypatch):
    import train_polar_benchmark

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fixture_not_a_real_model")
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(train_polar_benchmark, "snapshot_download", snapshot)
    evidence = train_polar_benchmark.foundation_evidence()
    assert evidence["files"]["config.json"] == sha256_file(tmp_path / "config.json")
    assert calls[0]["allow_patterns"] == ["config.json", "model.safetensors"]
    assert calls[0]["local_files_only"] is True


def test_protocol_has_separate_phase_and_confirmation_gates():
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(
        (root / "experiments" / "polar_benchmark_protocol_20260920.json").read_text(
            encoding="utf-8"
        )
    )
    assert protocol["test_access_during_current_queue"] is False
    assert protocol["historical_four_class_result"]["macro_f1"] == pytest.approx(0.9398833285904803)
    assert protocol["nine_class_task"]["test_wholly_unseen"] is False
