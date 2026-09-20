"""Safe local checkpoint restoration without downloads or model training."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools/restore_local_polar_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("restore_local_polar_checkpoints", MODULE_PATH)
RESTORE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RESTORE)


def test_copy_verifies_source_and_preserves_original(tmp_path):
    source = tmp_path / "original.safetensors"
    source.write_bytes(b"a tiny checkpoint")
    expected = {
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    destination = tmp_path / "cache/model.safetensors"
    RESTORE.copy_verified_file(source, destination, expected)
    assert destination.read_bytes() == source.read_bytes()
    RESTORE.copy_verified_file(source, destination, expected)
    destination.write_bytes(b"bad")
    assert source.read_bytes() == b"a tiny checkpoint"
    with pytest.raises(RuntimeError, match="destination differs"):
        RESTORE.copy_verified_file(source, destination, expected)


def test_copy_rejects_wrong_source_before_destination(tmp_path):
    source = tmp_path / "bad.safetensors"
    source.write_bytes(b"bad")
    destination = tmp_path / "cache/model.safetensors"
    with pytest.raises(RuntimeError, match="source hash or length mismatch"):
        RESTORE.copy_verified_file(source, destination, {"bytes": 3, "sha256": "0" * 64})
    assert not destination.parent.exists()


def test_new_artifact_never_overwrites_conflicting_evidence(tmp_path):
    path = tmp_path / "receipt.json"
    RESTORE.write_new_or_verify(path, b"original")
    RESTORE.write_new_or_verify(path, b"original")
    with pytest.raises(RuntimeError, match="not overwritten"):
        RESTORE.write_new_or_verify(path, b"replacement")
    assert path.read_bytes() == b"original"


def test_declared_processor_discloses_nonrecovered_bytes():
    import transformers

    if transformers.__version__ != "5.5.3":
        pytest.skip("Recovery declaration intentionally requires the exact benchmark environment")
    encoded, recovery = RESTORE.declared_processor_configuration()
    configuration = json.loads(encoded)
    assert configuration["size"] == {"height": 224, "width": 224}
    assert configuration["resample"] == 2
    assert (
        configuration["benchmark_processor_origin"]["original_hf_processor_bytes_recovered"]
        is False
    )
    assert recovery["selection_used_benchmark_labels"] is False


def test_preparation_update_refuses_active_worker_and_preserves_history(tmp_path):
    report_path, receipt_path = tmp_path / "preparation.json", tmp_path / "receipt.json"
    report_path.write_text(
        json.dumps({"status": "CHECKPOINT_PREPARATION", "models": []}), encoding="utf-8"
    )
    receipt_path.write_text("{}", encoding="utf-8")
    restored = {
        "checkpoint": {"verified": True},
        "processor_recovery": {"original_hf_processor_bytes_recovered": False},
    }
    with pytest.raises(RuntimeError, match="still active"):
        RESTORE.merge_completed_preparation_report(report_path, restored, receipt_path)
    previous = {"model_kind": "dinov3_base", "status": "CHECKPOINT_UNAVAILABLE"}
    report_path.write_text(
        json.dumps({"status": "CHECKPOINT_PREPARATION_COMPLETE", "models": [previous]}),
        encoding="utf-8",
    )
    RESTORE.merge_completed_preparation_report(report_path, restored, receipt_path)
    output = json.loads(report_path.read_text(encoding="utf-8"))
    assert output["availability_history"] == [previous]
    assert output["models"][0]["status"] == "PINNED_CHECKPOINT_READY"
    RESTORE.merge_completed_preparation_report(report_path, restored, receipt_path)
    assert len(json.loads(report_path.read_text(encoding="utf-8"))["availability_history"]) == 1
