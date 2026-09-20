"""No-download/no-GPU tests for the development extraction and resume contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torchvision import transforms

from hac.polar_benchmark_features import (
    CACHE_STATUS,
    MODEL_SPECS,
    BenchmarkFeatureDataset,
    FeatureRequest,
    _cache_lock,
    _cache_with_runtime,
    _validate_range_response,
    cache_features,
    checkpoint_evidence,
    read_development_manifest,
    require_cuda,
    restore_public_checkpoint,
)


class TinyModel(nn.Module):
    def forward(self, values):
        return values.mean(dim=(2, 3))


@pytest.fixture
def cohort(tmp_path):
    records = []
    for index in range(5):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (13, 9), (index * 30, 20, 100)).save(path)
        records.append(
            {
                "image_id": f"image-{index:03d}",
                "image_path": str(path),
                "split": "train" if index < 3 else "val",
                "bbox_xmin": 2,
                "bbox_ymin": 1,
                "bbox_xmax": 10,
                "bbox_ymax": 8,
                "label": "standing",
                "label_index": 0,
            }
        )
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest, index=False)
    request = FeatureRequest(
        manifest,
        tmp_path / "cache",
        "dinov2_base",
        "full_frame",
        batch_size=2,
        workers=0,
        chunk_rows=2,
    )
    return request


def run_tiny(request, model=None):
    request.validate()
    frame = read_development_manifest(request.manifest, view=request.view)
    return _cache_with_runtime(
        request,
        frame,
        model or TinyModel(),
        transforms.ToTensor(),
        {"fixture": "tiny"},
        {"policy": "tiny-test-only"},
        torch.device("cpu"),
    )


def test_specs_are_immutable_and_large_is_optional():
    assert len(MODEL_SPECS) == 5
    assert all(len(item["revision"]) == 40 for item in MODEL_SPECS.values())
    assert MODEL_SPECS["dinov2_base"]["representation"] == "last4_cls_mean_patch"
    assert MODEL_SPECS["dinov3_large"]["optional_capacity_check"] is True


def test_development_manifest_rejects_test_before_labels(cohort, monkeypatch):
    frame = pd.read_csv(cohort.manifest)
    frame.loc[0, "split"] = "test"
    frame.to_csv(cohort.manifest, index=False)
    original = pd.read_csv
    calls = []

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    with pytest.raises(ValueError, match="test is sealed"):
        read_development_manifest(cohort.manifest, view="full_frame")
    assert calls == [{"nrows": 0}, {"usecols": ["split"], "dtype": str, "keep_default_na": False}]


@pytest.mark.parametrize(
    "mutation", ["duplicate_id", "duplicate_path", "invalid_box", "missing_image"]
)
def test_manifest_rejects_invalid_rows(cohort, mutation):
    frame = pd.read_csv(cohort.manifest)
    if mutation == "duplicate_id":
        frame.loc[1, "image_id"] = frame.loc[0, "image_id"]
    elif mutation == "duplicate_path":
        frame.loc[1, "image_path"] = frame.loc[0, "image_path"]
    elif mutation == "invalid_box":
        frame.loc[0, "bbox_xmax"] = -1
    else:
        frame.loc[0, "image_path"] = "missing.png"
    frame.to_csv(cohort.manifest, index=False)
    with pytest.raises((ValueError, FileNotFoundError)):
        read_development_manifest(cohort.manifest, view="person_context_10")


def test_labels_are_optional_and_rows_remain_ordered(cohort):
    frame = pd.read_csv(cohort.manifest).drop(columns=["label", "label_index"])
    frame = frame.iloc[::-1]
    frame.to_csv(cohort.manifest, index=False)
    result = read_development_manifest(cohort.manifest, view="full_frame")
    assert result.image_id.tolist() == frame.image_id.tolist()
    assert "label" not in result


def test_source_hash_is_checked_at_image_read(cohort):
    frame = read_development_manifest(cohort.manifest, view="person_context_10")
    frame["image_sha256"] = "0" * 64
    dataset = BenchmarkFeatureDataset(
        frame, view="person_context_10", transform=transforms.ToTensor()
    )
    with pytest.raises(RuntimeError, match="changed"):
        dataset[0]


def test_audit_sha256_alias_preserved_and_checked(cohort):
    frame = pd.read_csv(cohort.manifest)
    frame["sha256"] = "0" * 64
    frame.to_csv(cohort.manifest, index=False)
    loaded = read_development_manifest(cohort.manifest, view="full_frame")
    assert loaded["image_sha256"].eq("0" * 64).all()
    assert "sha256" not in loaded
    with pytest.raises(RuntimeError, match="changed"):
        BenchmarkFeatureDataset(loaded, view="full_frame", transform=transforms.ToTensor())[0]
    frame["image_sha256"] = "1" * 64
    frame.to_csv(cohort.manifest, index=False)
    with pytest.raises(ValueError, match="evidence disagree"):
        read_development_manifest(cohort.manifest, view="full_frame")


def test_cuda_fails_closed(cohort, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CPU fallback is prohibited"):
        require_cuda()
    with pytest.raises(RuntimeError, match="CPU fallback is prohibited"):
        cache_features(cohort)
    assert not cohort.output_dir.exists()


def test_cache_and_verified_reuse(cohort):
    result = run_tiny(cohort)
    assert result["status"] == CACHE_STATUS
    assert result["feature_shape"] == [5, 3]
    assert result["test_rows_read"] == 0
    assert result["test_labels_read"] is False
    values = np.load(cohort.output_dir / "features.npy", allow_pickle=False)
    assert values.dtype == np.float32
    assert np.allclose(values[:, 0], np.arange(5) * 30 / 255)
    rows = pd.read_csv(cohort.output_dir / "rows.csv")
    assert rows.row.tolist() == list(range(5))
    assert rows.image_id.tolist() == [f"image-{index:03d}" for index in range(5)]

    class MustNotExecute(nn.Module):
        def forward(self, values):
            raise AssertionError("Verified cache must not execute the model")

    assert run_tiny(cohort, MustNotExecute()) == result


def test_changed_contract_rejected(cohort):
    run_tiny(cohort)
    with pytest.raises(RuntimeError, match="contract differs"):
        run_tiny(replace(cohort, batch_size=1))


def test_final_artifact_tamper_rejected(cohort):
    run_tiny(cohort)
    with (cohort.output_dir / "features.npy").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        run_tiny(cohort)


def test_source_image_tamper_rejected_on_reuse(cohort):
    run_tiny(cohort)
    Image.new("RGB", (13, 9), (255, 255, 255)).save(cohort.manifest.parent / "0.png")
    with pytest.raises(RuntimeError, match="source image changed"):
        run_tiny(cohort)


def test_resume_after_interruption_skips_verified_chunks(cohort):
    class InterruptedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, values):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("simulated interruption")
            return values.mean(dim=(2, 3))

    with pytest.raises(RuntimeError, match="simulated"):
        run_tiny(cohort, InterruptedModel())
    assert not (cohort.output_dir / "provenance.json").exists()
    assert len(list((cohort.output_dir / "chunks").glob("*.json"))) == 1
    result = run_tiny(cohort)
    assert result["resumed_rows"] == 2
    assert result["feature_shape"] == [5, 3]


def test_resume_rejects_tampered_chunk(cohort):
    run_tiny(cohort)
    (cohort.output_dir / "provenance.json").unlink()
    chunk = cohort.output_dir / "chunks" / "00000000.npy"
    with chunk.open("ab") as handle:
        handle.write(b"invalid")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        run_tiny(cohort)


def test_nonfinite_features_rejected(cohort):
    class InvalidModel(nn.Module):
        def forward(self, values):
            return torch.full((len(values), 3), float("nan"))

    with pytest.raises(RuntimeError, match="nonfinite"):
        run_tiny(cohort, InvalidModel())
    assert not (cohort.output_dir / "provenance.json").exists()


def test_checkpoint_evidence_requires_complete_safetensors(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no safetensors"):
        checkpoint_evidence(tmp_path, "dinov2_base")
    (tmp_path / "model.safetensors").write_bytes(b"tiny fixture checkpoint")
    result = checkpoint_evidence(tmp_path, "dinov2_base")
    assert len(result["files"]) == 2
    assert (
        next(row for row in result["files"] if row["path"] == "model.safetensors")["sha256"]
        == hashlib.sha256(b"tiny fixture checkpoint").hexdigest()
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "missing.safetensors"}}), encoding="utf-8"
    )
    with pytest.raises(FileNotFoundError, match="missing weight shards"):
        checkpoint_evidence(tmp_path, "dinov2_base")


def test_active_writer_lock_and_release(tmp_path):
    with _cache_lock(tmp_path):
        with pytest.raises(RuntimeError, match="active writer"), _cache_lock(tmp_path):
            pass
    with _cache_lock(tmp_path):
        pass


def test_public_range_requires_exact_status_range_and_length():
    response = SimpleNamespace(
        status_code=206, headers={"Content-Range": "bytes 0-3/9"}, content=b"abcd"
    )
    assert _validate_range_response(response, start=0, end=3, total=9) == b"abcd"
    response.status_code = 200
    with pytest.raises(RuntimeError, match="exact requested"):
        _validate_range_response(response, start=0, end=3, total=9)
    response.status_code = 206
    response.headers["Content-Range"] = "bytes 4-7/9"
    with pytest.raises(RuntimeError, match="exact requested"):
        _validate_range_response(response, start=0, end=3, total=9)
    response.headers["Content-Range"] = "bytes 0-3/9"
    response.content = b"abc"
    with pytest.raises(RuntimeError, match="wrong length"):
        _validate_range_response(response, start=0, end=3, total=9)


def test_public_range_cannot_be_used_for_gated_or_unrequested_download(cohort):
    with pytest.raises(ValueError, match="authorized public"):
        restore_public_checkpoint(cohort, cohort.output_dir)
    with pytest.raises(ValueError, match="authorized public"):
        restore_public_checkpoint(
            replace(cohort, model_kind="dinov3_base", allow_download=True), cohort.output_dir
        )


def test_public_range_publishes_only_verified_blob(cohort, monkeypatch):
    import httpx
    import huggingface_hub

    payload = b"tiny safe weights"
    digest = hashlib.sha256(payload).hexdigest()
    request = replace(
        cohort,
        model_kind="convnextv2_base",
        allow_download=True,
        cache_dir=cohort.manifest.parent / "hf-cache",
    )
    specification = MODEL_SPECS[request.model_kind]
    metadata = SimpleNamespace(
        etag=digest,
        size=len(payload),
        commit_hash=specification["revision"],
        location="https://invalid.example/no-real-network",
    )
    monkeypatch.setattr(huggingface_hub, "get_hf_file_metadata", lambda *args, **kwargs: metadata)
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda **kwargs: "fixture-configuration"
    )

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def get(self, url, headers):
            assert headers["Range"] == f"bytes=0-{len(payload) - 1}"
            return SimpleNamespace(
                status_code=206,
                headers={"Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}"},
                content=payload,
            )

    monkeypatch.setattr(httpx, "Client", FakeClient)
    snapshot = restore_public_checkpoint(request, cohort.output_dir)
    assert (snapshot / "model.safetensors").read_bytes() == payload
    evidence = json.loads((cohort.output_dir / "progress.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "PUBLIC_CHECKPOINT_VERIFIED"
    assert evidence["sha256"] == digest
    assert restore_public_checkpoint(request, cohort.output_dir) == snapshot
    (snapshot / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="canonical checkpoint blob failed"):
        restore_public_checkpoint(request, cohort.output_dir)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"chunk_rows": 3},
        {"workers": -1},
        {"max_images": 0},
        {"model_kind": "siglip2_base", "preprocess": "historical_polar_eval"},
    ],
)
def test_invalid_requests(cohort, kwargs):
    with pytest.raises(ValueError):
        replace(cohort, **kwargs).validate()
