"""Pinned, resumable development-only feature extraction for POLAR benchmarks.

This module intentionally has no held-out extraction switch. A separately reviewed
evaluation lock must be implemented before the historical or nine-class test is used.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from hac.augmentations import build_aspect_preserving_eval_transform, build_eval_transform
from hac.polar import image_view, sha256_file
from hac.polar_features import official_multilayer_features
from hac.polar_models import DINO_MODEL_SPECS
from hac.vcoco_v3_representations import DINO_V3_MODEL_SPEC, SIGLIP2_MODEL_SPEC

MODEL_SPECS = {
    "dinov2_base": {
        **DINO_MODEL_SPECS["dinov2_base"],
        "representation": "last4_cls_mean_patch",
        "dimensions": 3840,
    },
    "dinov3_base": {**DINO_V3_MODEL_SPEC, "dimensions": 768},
    "siglip2_base": {
        "model_id": SIGLIP2_MODEL_SPEC["model_id"],
        "revision": SIGLIP2_MODEL_SPEC["revision"],
        "representation": "vision_pooler_output",
        "dimensions": 768,
    },
    "convnextv2_base": {
        "model_id": "facebook/convnextv2-base-22k-224",
        "revision": "758ff0922dc09136abb55774e7f8b1e1bd0dc344",
        "representation": "pooler_output",
        "dimensions": 1024,
        "pretraining": "FCMAE followed by ImageNet-22K supervised fine-tuning",
    },
    "dinov3_large": {
        "model_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
        "representation": "pooler_output",
        "dimensions": 1024,
        "access": "manually_gated",
        "optional_capacity_check": True,
    },
}
VIEWS = ("full_frame", "person_context_10")
PREPROCESS_POLICIES = ("official_processor", "historical_polar_eval", "aspect_preserving_pad")
BBOX_COLUMNS = ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax")
CACHE_STATUS = "POLAR_BENCHMARK_DEVELOPMENT_FEATURE_CACHE_COMPLETE"


def _json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class FeatureRequest:
    manifest: Path
    output_dir: Path
    model_kind: str
    view: str
    preprocess: str = "official_processor"
    batch_size: int = 32
    workers: int = 4
    chunk_rows: int = 1024
    cache_dir: Path | None = None
    allow_download: bool = False
    max_images: int | None = None
    autocast_dtype: str = "float16"

    def validate(self) -> None:
        if self.model_kind not in MODEL_SPECS or self.view not in VIEWS:
            raise ValueError("Unknown model kind or image view")
        if self.preprocess not in PREPROCESS_POLICIES:
            raise ValueError("Unknown preprocessing policy")
        if self.preprocess != "official_processor" and self.model_kind != "dinov2_base":
            raise ValueError("Non-official preprocessing is an explicit DINOv2 control only")
        if self.batch_size < 1 or self.workers < 0:
            raise ValueError("Batch size must be positive and workers cannot be negative")
        if self.chunk_rows < self.batch_size or self.chunk_rows % self.batch_size:
            raise ValueError("chunk_rows must be a positive multiple of batch_size")
        if self.max_images is not None and self.max_images < 1:
            raise ValueError("max_images must be positive")
        if self.autocast_dtype not in {"float16", "bfloat16"}:
            raise ValueError("Unknown CUDA autocast dtype")


def read_development_manifest(path: Path, *, view: str) -> pd.DataFrame:
    """Reject held-out rows before parsing any label column; never silently filter."""
    path = path.resolve(strict=True)
    header = pd.read_csv(path, nrows=0).columns.tolist()
    required = {"image_id", "image_path", "split"}
    if view == "person_context_10":
        required.update(BBOX_COLUMNS)
    if view not in VIEWS or not required.issubset(header):
        raise ValueError(f"Manifest is missing required columns: {sorted(required - set(header))}")
    split = pd.read_csv(path, usecols=["split"], dtype=str, keep_default_na=False)["split"]
    if not len(split) or set(split) - {"train", "val"}:
        raise ValueError(
            "Only nonempty train/val development manifests are accepted; test is sealed"
        )
    retained = required | ({"label", "label_index", "image_sha256", "sha256"} & set(header))
    frame = pd.read_csv(
        path,
        usecols=lambda column: column in retained,
        dtype={
            "image_id": str,
            "image_path": str,
            "split": str,
            "image_sha256": str,
            "sha256": str,
        },
        keep_default_na=False,
    )
    identifiers = frame["image_id"].astype(str)
    if identifiers.str.strip().eq("").any() or identifiers.duplicated().any():
        raise ValueError("Manifest image identifiers must be nonempty and globally unique")
    frame["image_path"] = frame["image_path"].map(
        lambda item: str((path.parent / str(item)).resolve())
    )
    if frame["image_path"].duplicated().any():
        raise ValueError("An image path cannot occur more than once in a development manifest")
    if not frame["image_path"].map(lambda item: Path(item).is_file()).all():
        raise FileNotFoundError("One or more development images are missing")
    if "sha256" in frame:
        if "image_sha256" in frame and not frame["sha256"].eq(frame["image_sha256"]).all():
            raise ValueError("Manifest sha256 and image_sha256 evidence disagree")
        frame["image_sha256"] = frame.pop("sha256")
    if view == "person_context_10":
        coordinates = frame[list(BBOX_COLUMNS)].apply(pd.to_numeric, errors="raise")
        if not np.isfinite(coordinates.to_numpy(dtype=float)).all():
            raise ValueError("Bounding boxes must be finite")
        if (
            (coordinates.bbox_xmax <= coordinates.bbox_xmin)
            | (coordinates.bbox_ymax <= coordinates.bbox_ymin)
        ).any():
            raise ValueError("Bounding boxes must have positive area")
        frame[list(BBOX_COLUMNS)] = coordinates
    if (
        "image_sha256" in frame
        and not frame["image_sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()
    ):
        raise ValueError("Manifest image_sha256 values must be lowercase SHA256 digests")
    return frame.reset_index(drop=True)


def resolve_checkpoint(request: FeatureRequest) -> Path:
    from huggingface_hub import snapshot_download

    specification = MODEL_SPECS[request.model_kind]
    if re.fullmatch(r"[0-9a-f]{40}", specification["revision"]) is None:
        raise ValueError("An immutable full model revision is required")
    # Download only data, never repository Python or pickle-format checkpoint files.
    return Path(
        snapshot_download(
            repo_id=specification["model_id"],
            revision=specification["revision"],
            cache_dir=request.cache_dir,
            local_files_only=not request.allow_download,
            allow_patterns=["*.json", "*.safetensors"],
        )
    )


def checkpoint_evidence(snapshot: Path, model_kind: str) -> dict:
    specification = MODEL_SPECS[model_kind]
    if not (snapshot / "config.json").is_file():
        raise FileNotFoundError("The pinned snapshot has no model configuration")
    weight_paths = sorted(snapshot.glob("*.safetensors"))
    if not weight_paths:
        raise FileNotFoundError(
            "The pinned snapshot has no safetensors weights; no substitute used"
        )
    index_path = snapshot / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        expected = set(index["weight_map"].values())
        if expected - {path.name for path in weight_paths}:
            raise FileNotFoundError("The pinned checkpoint is missing weight shards")
    elif not (snapshot / "model.safetensors").is_file():
        raise FileNotFoundError("No complete safetensors checkpoint or shard index exists")
    paths = sorted(set(weight_paths) | set(snapshot.glob("*.json")))
    return {
        "model_id": specification["model_id"],
        "revision": specification["revision"],
        "files": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in paths
        ],
    }


def probe_model_availability(cache_dir: Path | None = None) -> list[dict]:
    """No download, CUDA initialization, or token output; optional models stay optional."""
    output = []
    for model_kind, specification in MODEL_SPECS.items():
        request = FeatureRequest(Path("unused"), Path("unused"), model_kind, "full_frame")
        request = FeatureRequest(**{**request.__dict__, "cache_dir": cache_dir})
        try:
            snapshot = resolve_checkpoint(request)
            # Read only directory metadata during a cheap availability probe.
            complete = (snapshot / "config.json").is_file() and (
                (snapshot / "model.safetensors").is_file()
                or (snapshot / "model.safetensors.index.json").is_file()
            )
            status = "CACHED_SNAPSHOT_PRESENT" if complete else "INCOMPLETE_LOCAL_SNAPSHOT"
        except (OSError, ValueError):
            status = "NOT_CACHED_ACCESS_OR_DOWNLOAD_REQUIRED"
        output.append({"model_kind": model_kind, **specification, "status": status})
    return output


def download_model_checkpoints(
    model_kinds: list[str],
    *,
    cache_dir: Path | None,
    report_path: Path,
    public_range_download: bool = False,
) -> list[dict]:
    """Explicit data-only download; gated access failures never trigger substitutes."""
    results = []
    for model_kind in model_kinds:
        if model_kind not in MODEL_SPECS:
            raise ValueError("Unknown requested model")
        request = FeatureRequest(
            Path("unused"),
            Path("unused"),
            model_kind,
            "full_frame",
            cache_dir=cache_dir,
            allow_download=True,
        )
        started = time.perf_counter()
        try:
            if public_range_download and MODEL_SPECS[model_kind].get("access") != "manually_gated":
                restore_public_checkpoint(
                    request, report_path.parent / "public_checkpoint_parts" / model_kind
                )
            snapshot = resolve_checkpoint(request)
            evidence = checkpoint_evidence(snapshot, model_kind)
            result = {
                "model_kind": model_kind,
                "status": "PINNED_CHECKPOINT_READY",
                "checkpoint": evidence,
            }
        except Exception as error:
            # Do not persist provider exceptions that could contain credential-bearing URLs.
            result = {
                "model_kind": model_kind,
                "status": "CHECKPOINT_UNAVAILABLE",
                "error_type": type(error).__name__,
                "substitute_used": False,
            }
        result["runtime_seconds"] = time.perf_counter() - started
        results.append(result)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(report_path, {"status": "CHECKPOINT_PREPARATION", "models": results})
        print(
            json.dumps({key: value for key, value in result.items() if key != "checkpoint"}),
            flush=True,
        )
    _write_json(report_path, {"status": "CHECKPOINT_PREPARATION_COMPLETE", "models": results})
    return results


def _validate_range_response(response: Any, *, start: int, end: int, total: int) -> bytes:
    expected_range = f"bytes {start}-{end}/{total}"
    if response.status_code != 206 or response.headers.get("Content-Range") != expected_range:
        raise RuntimeError("Public checkpoint server did not return the exact requested byte range")
    data = response.content
    if len(data) != end - start + 1:
        raise RuntimeError("Public checkpoint range response has the wrong length")
    return data


def restore_public_checkpoint(request: FeatureRequest, work_dir: Path, *, workers: int = 8) -> Path:
    """Restore public, immutable safetensors by bounded ranges and an exact SHA256.

    Gated models are categorically excluded. Only the fully verified blob is added
    to the ordinary HF cache; incomplete ranges never become model snapshots.
    """
    import httpx
    from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url
    from huggingface_hub.constants import HF_HUB_CACHE

    specification = MODEL_SPECS[request.model_kind]
    if not request.allow_download or specification.get("access") == "manually_gated":
        raise ValueError("Range restore requires an explicitly authorized public checkpoint")
    if workers < 1 or workers > 8:
        raise ValueError("Public range downloads are bounded to one through eight workers")
    revision, model_id = specification["revision"], specification["model_id"]
    url = hf_hub_url(model_id, "model.safetensors", revision=revision)
    metadata = get_hf_file_metadata(url, token=False)
    digest = str(metadata.etag).strip('"')
    size = int(metadata.size or 0)
    if not re.fullmatch(r"[0-9a-f]{64}", digest) or size < 1 or metadata.commit_hash != revision:
        raise RuntimeError("Public checkpoint metadata does not establish an immutable SHA256 blob")
    if request.model_kind == "dinov2_base" and (
        digest != "d73036b56966966d07975d696bde331762f37297e2f095de8cea0040c3aa0841"
        or size != 346345912
    ):
        raise RuntimeError("DINOv2 weights differ from the historical checkpoint evidence")
    hub_root = Path(request.cache_dir or HF_HUB_CACHE)
    repository = hub_root / ("models--" + model_id.replace("/", "--"))
    blob = repository / "blobs" / digest
    snapshot = repository / "snapshots" / revision
    if blob.exists() and (blob.stat().st_size != size or sha256_file(blob) != digest):
        raise RuntimeError(
            "An existing canonical checkpoint blob failed verification; not overwritten"
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    download_contract = {
        "model_id": model_id,
        "revision": revision,
        "sha256": digest,
        "bytes": size,
        "chunk_bytes": 1024 * 1024,
    }
    contract_path = work_dir / "download_contract.json"
    if (
        contract_path.exists()
        and json.loads(contract_path.read_text(encoding="utf-8")) != download_contract
    ):
        raise RuntimeError("Public download resume contract has changed")
    _write_json(contract_path, download_contract)
    chunk_bytes = download_contract["chunk_bytes"]
    starts = list(range(0, size, chunk_bytes))
    started = time.perf_counter()
    if not blob.exists():
        with httpx.Client(
            follow_redirects=True, timeout=60.0, limits=httpx.Limits(max_connections=workers)
        ) as client:

            def restore_range(start: int) -> Path:
                end = min(start + chunk_bytes, size) - 1
                path = work_dir / f"{start:012d}.part"
                evidence_path = path.with_suffix(".json")
                if path.exists() and evidence_path.exists():
                    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                    if path.stat().st_size == end - start + 1 and evidence == {
                        "start": start,
                        "end": end,
                        "sha256": sha256_file(path),
                    }:
                        return path
                    raise RuntimeError("An existing public checkpoint range failed verification")
                for attempt in range(3):
                    try:
                        response = client.get(
                            metadata.location,
                            headers={
                                "Range": f"bytes={start}-{end}",
                                "Accept-Encoding": "identity",
                            },
                        )
                        data = _validate_range_response(response, start=start, end=end, total=size)
                        temporary = path.with_suffix(".part.tmp")
                        temporary.write_bytes(data)
                        temporary.replace(path)
                        _write_json(
                            evidence_path,
                            {
                                "start": start,
                                "end": end,
                                "sha256": hashlib.sha256(data).hexdigest(),
                            },
                        )
                        return path
                    except (httpx.HTTPError, RuntimeError):
                        if attempt == 2:
                            raise
                raise AssertionError("Range retry loop ended without a result")

            completed = 0
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(restore_range, start) for start in starts]
                for future in as_completed(futures):
                    completed += future.result().stat().st_size
                    if completed % (16 * chunk_bytes) == 0 or completed == size:
                        progress = {
                            "model_kind": request.model_kind,
                            "status": "PUBLIC_CHECKPOINT_DOWNLOADING",
                            "bytes_complete": completed,
                            "bytes_total": size,
                            "runtime_seconds": time.perf_counter() - started,
                        }
                        _write_json(work_dir / "progress.json", progress)
                        print(json.dumps(progress), flush=True)
        assembled = work_dir / "verified_model.safetensors"
        with assembled.open("wb") as handle:
            for start in starts:
                handle.write((work_dir / f"{start:012d}.part").read_bytes())
        if assembled.stat().st_size != size or sha256_file(assembled) != digest:
            raise RuntimeError("Assembled public checkpoint failed its mandatory full-file SHA256")
        blob.parent.mkdir(parents=True, exist_ok=True)
        # A canonical blob is published only after complete independent verification.
        if blob.exists():
            if blob.stat().st_size != size or sha256_file(blob) != digest:
                raise RuntimeError("A competing cache writer published a different blob")
        else:
            os.link(assembled, blob)
    snapshot.mkdir(parents=True, exist_ok=True)
    snapshot_weight = snapshot / "model.safetensors"
    if snapshot_weight.exists():
        if sha256_file(snapshot_weight) != digest:
            raise RuntimeError("The existing snapshot weight differs from the verified checkpoint")
    else:
        os.link(blob, snapshot_weight)
    for filename in ("config.json", "preprocessor_config.json"):
        hf_hub_download(
            repo_id=model_id,
            revision=revision,
            filename=filename,
            cache_dir=request.cache_dir,
            token=False,
        )
    _write_json(
        work_dir / "progress.json",
        {
            "model_kind": request.model_kind,
            "status": "PUBLIC_CHECKPOINT_VERIFIED",
            "bytes": size,
            "sha256": digest,
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    return snapshot


class BenchmarkFeatureModel(nn.Module):
    def __init__(self, backbone: nn.Module, model_kind: str) -> None:
        super().__init__()
        self.backbone = backbone
        self.model_kind = model_kind

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        if self.model_kind == "dinov2_base":
            output = self.backbone(pixel_values=pixels, output_hidden_states=True)
            return official_multilayer_features(output.hidden_states, self.backbone.layernorm)
        output = self.backbone(pixel_values=pixels)
        if getattr(output, "pooler_output", None) is None:
            raise RuntimeError("The pinned representation has no pooler_output; no fallback used")
        return output.pooler_output


class ProcessorTransform:
    """Pickleable wrapper so Windows DataLoader workers can use an official processor."""

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, image: Image.Image) -> torch.Tensor:
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


def prepare_model_and_transform(request: FeatureRequest) -> tuple[nn.Module, Any, dict, dict]:
    from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

    snapshot = resolve_checkpoint(request)
    checkpoint = checkpoint_evidence(snapshot, request.model_kind)
    model_class = SiglipVisionModel if request.model_kind == "siglip2_base" else AutoModel
    backbone, loading = model_class.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(
            "The pinned backbone did not load completely; no random weights accepted"
        )
    if request.preprocess == "official_processor":
        processor = AutoImageProcessor.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=False, use_fast=False
        )
        transform = ProcessorTransform(processor)
        preprocess = {"policy": request.preprocess, "processor": processor.to_dict()}
    elif request.preprocess == "historical_polar_eval":
        transform = build_eval_transform(224)
        preprocess = {
            "policy": request.preprocess,
            "resize_short_edge": 256,
            "center_crop": 224,
            "interpolation": "bicubic",
            "normalization": "ImageNet",
        }
    else:
        transform = build_aspect_preserving_eval_transform(224)
        preprocess = {
            "policy": request.preprocess,
            "square_pad_fill": [124, 116, 104],
            "resize": [224, 224],
            "interpolation": "bicubic",
            "normalization": "ImageNet",
        }
    checkpoint["unexpected_keys"] = sorted(loading.get("unexpected_keys", []))
    return BenchmarkFeatureModel(backbone, request.model_kind), transform, checkpoint, preprocess


class BenchmarkFeatureDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, *, view: str, transform: Any) -> None:
        self.records = frame.to_dict(orient="records")
        self.view = view
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        row = self.records[index]
        encoded = Path(row["image_path"]).read_bytes()
        digest = hashlib.sha256(encoded).hexdigest()
        if row.get("image_sha256") and digest != row["image_sha256"]:
            raise RuntimeError("Source image changed since the manifest was sealed")
        with Image.open(io.BytesIO(encoded)) as image:
            pixels = self.transform(image_view(image.convert("RGB"), row, self.view))
        return {"pixels": pixels, "image_sha256": digest}


@contextlib.contextmanager
def _cache_lock(output_dir: Path):
    """OS lock is released on process exit, including crashes; no stale-PID guessing."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".cache.lock").open("a+b") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(
                "This feature-cache directory already has an active writer"
            ) from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _verify_artifacts(directory: Path, artifact_hashes: dict) -> None:
    for name, digest in artifact_hashes.items():
        if Path(name).name != name or not (directory / name).is_file():
            raise RuntimeError("Feature-cache artifact is missing or escapes its directory")
        if sha256_file(directory / name) != digest:
            raise RuntimeError(f"Feature-cache artifact hash mismatch: {name}")


def _verify_source_rows(rows: pd.DataFrame, frame: pd.DataFrame) -> None:
    if rows["image_id"].astype(str).tolist() != frame["image_id"].astype(str).tolist():
        raise RuntimeError("Cached feature row alignment differs from the manifest")
    for path, digest in zip(frame["image_path"], rows["image_sha256"], strict=True):
        if sha256_file(path) != digest:
            raise RuntimeError("A cached source image changed; refuse stale features")


def _resume_chunks(output_dir: Path, contract_hash: str, frame: pd.DataFrame) -> list[dict]:
    completed = []
    position = 0
    for path in sorted((output_dir / "chunks").glob("*.json")):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("contract_sha256") != contract_hash or metadata.get("start") != position:
            raise RuntimeError("Resumable feature chunks have drifted or are not contiguous")
        end = metadata["end"]
        if end <= position or end > len(frame):
            raise RuntimeError("Invalid feature chunk row boundaries")
        _verify_artifacts(path.parent, metadata["artifact_sha256"])
        rows = pd.read_csv(path.parent / metadata["rows_file"], dtype={"image_id": str})
        _verify_source_rows(rows, frame.iloc[position:end])
        features = np.load(
            path.parent / metadata["features_file"], mmap_mode="r", allow_pickle=False
        )
        if list(features.shape) != metadata["shape"] or features.dtype != np.float32:
            raise RuntimeError("Resumable feature chunk shape or dtype mismatch")
        completed.append(metadata)
        position = end
    return completed


def _save_chunk(
    output_dir: Path,
    contract_hash: str,
    frame: pd.DataFrame,
    start: int,
    features: np.ndarray,
    image_hashes: list[str],
) -> dict:
    directory = output_dir / "chunks"
    directory.mkdir(exist_ok=True)
    stem = f"{start:08d}"
    array_path, rows_path = directory / f"{stem}.npy", directory / f"{stem}.csv"
    with array_path.with_suffix(".npy.tmp").open("wb") as handle:
        np.save(handle, features, allow_pickle=False)
    array_path.with_suffix(".npy.tmp").replace(array_path)
    columns = [item for item in ("image_id", "split", "label", "label_index") if item in frame]
    rows = frame.iloc[start : start + len(features)][columns].copy()
    rows.insert(0, "row", np.arange(start, start + len(features)))
    rows["image_sha256"] = image_hashes
    rows.to_csv(rows_path.with_suffix(".csv.tmp"), index=False, lineterminator="\n")
    rows_path.with_suffix(".csv.tmp").replace(rows_path)
    metadata = {
        "contract_sha256": contract_hash,
        "start": start,
        "end": start + len(features),
        "shape": list(features.shape),
        "features_file": array_path.name,
        "rows_file": rows_path.name,
        "artifact_sha256": {
            array_path.name: sha256_file(array_path),
            rows_path.name: sha256_file(rows_path),
        },
    }
    _write_json(directory / f"{stem}.json", metadata)
    return metadata


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("POLAR benchmark extraction requires CUDA; CPU fallback is prohibited")
    return torch.device("cuda:0")


def cache_features(request: FeatureRequest) -> dict:
    """Production entry point. Every inference call requires an actual CUDA device."""
    request.validate()
    frame = read_development_manifest(request.manifest, view=request.view)
    if request.max_images is not None:
        frame = frame.head(request.max_images).copy()
    device = require_cuda()
    if request.autocast_dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("The requested CUDA device does not support bfloat16")
    torch.set_float32_matmul_precision("high")
    with _cache_lock(request.output_dir):
        model, transform, checkpoint, preprocess = prepare_model_and_transform(request)
        return _cache_with_runtime(request, frame, model, transform, checkpoint, preprocess, device)


@torch.inference_mode()
def _cache_with_runtime(
    request: FeatureRequest,
    frame: pd.DataFrame,
    model: nn.Module,
    transform: Any,
    checkpoint: dict,
    preprocess: dict,
    device: torch.device,
) -> dict:
    """The internal CPU-capable core permits tiny deterministic tests, not a CLI fallback."""
    started = time.perf_counter()
    output_dir = request.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    import transformers

    contract = {
        "schema_version": 1,
        "model_kind": request.model_kind,
        "model": MODEL_SPECS[request.model_kind],
        "view": request.view,
        "preprocess": preprocess,
        "checkpoint": checkpoint,
        "manifest_sha256": sha256_file(request.manifest),
        "rows": len(frame),
        "max_images": request.max_images,
        "scope": "development_smoke" if request.max_images is not None else "development",
        "batch_size": request.batch_size,
        "chunk_rows": request.chunk_rows,
        "autocast_dtype": request.autocast_dtype,
        "device_type": device.type,
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_runtime": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "test_rows_read": 0,
        "test_labels_read": False,
        "source_sha256": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "polar_benchmark_features.py",
                "polar_features.py",
                "polar_models.py",
                "vcoco_v3_representations.py",
                "polar.py",
                "augmentations.py",
            )
        },
    }
    contract_hash = _json_hash(contract)
    provenance_path = output_dir / "provenance.json"
    if provenance_path.is_file():
        previous = json.loads(provenance_path.read_text(encoding="utf-8"))
        if (
            previous.get("contract_sha256") != contract_hash
            or previous.get("status") != CACHE_STATUS
        ):
            raise RuntimeError(
                "Existing feature-cache contract differs; use a new output directory"
            )
        _verify_artifacts(output_dir, previous["artifact_sha256"])
        rows = pd.read_csv(output_dir / "rows.csv", dtype={"image_id": str})
        _verify_source_rows(rows, frame)
        return previous
    contract_path = output_dir / "contract.json"
    if contract_path.is_file():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if _json_hash(existing) != contract_hash:
            raise RuntimeError("Incomplete feature-cache contract differs; refuse cross-run resume")
    else:
        _write_json(contract_path, contract)
    chunks = _resume_chunks(output_dir, contract_hash, frame)
    position = chunks[-1]["end"] if chunks else 0
    resumed_rows = position
    loader = DataLoader(
        BenchmarkFeatureDataset(frame.iloc[position:], view=request.view, transform=transform),
        batch_size=request.batch_size,
        shuffle=False,
        num_workers=request.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=request.workers > 0,
    )
    model = model.to(device).eval().requires_grad_(False)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    pending: list[np.ndarray] = []
    hashes: list[str] = []
    feature_dimensions = chunks[0]["shape"][1] if chunks else None
    extraction_started = time.perf_counter()
    for batch in loader:
        pixels = batch["pixels"].to(device, non_blocking=device.type == "cuda")
        with torch.autocast(
            device_type=device.type,
            dtype=getattr(torch, request.autocast_dtype),
            enabled=device.type == "cuda",
        ):
            features = model(pixels).float().cpu().numpy()
        if features.ndim != 2 or len(features) != len(pixels) or not np.isfinite(features).all():
            raise RuntimeError("Feature model returned invalid dimensions or nonfinite values")
        if feature_dimensions is not None and features.shape[1] != feature_dimensions:
            raise RuntimeError("Feature dimensionality changed within a cache")
        feature_dimensions = features.shape[1]
        if (
            device.type == "cuda"
            and feature_dimensions != MODEL_SPECS[request.model_kind]["dimensions"]
        ):
            raise RuntimeError("The pinned model returned an unexpected representation dimension")
        pending.append(features)
        hashes.extend(batch["image_sha256"])
        if len(hashes) >= request.chunk_rows or position + len(hashes) == len(frame):
            chunk = _save_chunk(
                output_dir, contract_hash, frame, position, np.concatenate(pending), hashes
            )
            chunks.append(chunk)
            position = chunk["end"]
            pending, hashes = [], []
            elapsed = time.perf_counter() - extraction_started
            rate = (position - resumed_rows) / max(elapsed, 1e-9)
            progress = {
                "status": "EXTRACTING",
                "rows_complete": position,
                "rows_total": len(frame),
                "resumed_rows": resumed_rows,
                "images_per_second": rate,
                "estimated_seconds_remaining": (len(frame) - position) / max(rate, 1e-9),
                "runtime_seconds": time.perf_counter() - started,
            }
            _write_json(output_dir / "progress.json", progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
    if position != len(frame) or not chunks:
        raise RuntimeError("Feature extraction did not produce the complete requested cohort")
    dimensions = chunks[0]["shape"][1]
    features_path = output_dir / "features.npy"
    temporary = output_dir / "features.npy.tmp"
    joined = np.lib.format.open_memmap(
        temporary, mode="w+", dtype=np.float32, shape=(len(frame), dimensions)
    )
    row_parts = []
    for chunk in chunks:
        values = np.load(
            output_dir / "chunks" / chunk["features_file"], mmap_mode="r", allow_pickle=False
        )
        joined[chunk["start"] : chunk["end"]] = values
        row_parts.append(
            pd.read_csv(output_dir / "chunks" / chunk["rows_file"], dtype={"image_id": str})
        )
    joined.flush()
    del joined
    temporary.replace(features_path)
    rows_path = output_dir / "rows.csv"
    pd.concat(row_parts, ignore_index=True).to_csv(rows_path, index=False, lineterminator="\n")
    elapsed = time.perf_counter() - started
    provenance = {
        **contract,
        "status": CACHE_STATUS,
        "contract_sha256": contract_hash,
        "feature_shape": [len(frame), dimensions],
        "feature_dtype": "float32",
        "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        "runtime_seconds": elapsed,
        "resumed_rows": resumed_rows,
        "new_rows_per_second": (len(frame) - resumed_rows) / max(elapsed, 1e-9),
        "chunks": len(chunks),
        "artifact_sha256": {
            rows_path.name: sha256_file(rows_path),
            features_path.name: sha256_file(features_path),
        },
        "rows_sha256": sha256_file(rows_path),
        "features_sha256": sha256_file(features_path),
    }
    _write_json(provenance_path, provenance)
    _write_json(
        output_dir / "progress.json",
        {"status": CACHE_STATUS, "rows_complete": len(frame), "runtime_seconds": elapsed},
    )
    return provenance
