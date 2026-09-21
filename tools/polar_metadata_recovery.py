"""Audited metadata-only continuation of the sealed POLAR final evaluation.

Normalize only processor.image_mean/image_std tuples to JSON lists. The actual
model, transform, checkpoint evidence, original parity guard and locked sources
are unchanged. Reuse the separately audited bounded Windows status-I/O retry.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import functools
import gc
import json
import math
import os
import runpy
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import polar_status_io_recovery as status_io

STATUS = "USER_AUTHORIZED_PROCESSOR_METADATA_RECOVERY"
FIELDS = ("image_mean", "image_std")
REMAINING_JOBS = (
    "extract_locked_test_features",
    "predict_locked_candidates",
    "compare_locked_candidates",
)
FILES = ("tools/polar_metadata_recovery.py", "tests/test_polar_metadata_recovery.py")
POLICY = {
    "fields": [f"processor.{name}" for name in FIELDS],
    "operation": "tuple_to_list_only_preserving_exact_JSON_and_numeric_values",
    "model_transform_checkpoint_changed": False,
    "parity_guard_bypassed": False,
    "test_based_selection": False,
    "training_allowed": False,
}


def normalize_metadata(preprocess):
    """Copy metadata, never the processor; preserve its exact JSON representation."""
    result = copy.deepcopy(preprocess)
    changed = []
    if result.get("policy") == "official_processor":
        processor = result.get("processor", {})
        for field in FIELDS:
            values = processor.get(field)
            if isinstance(values, tuple):
                if len(values) != 3 or any(
                    type(value) not in (int, float) or not math.isfinite(value) for value in values
                ):
                    raise ValueError("Normalization metadata requires three finite numbers")
                processor[field] = list(values)
                changed.append(f"processor.{field}")
    before = json.dumps(preprocess, sort_keys=True, allow_nan=False)
    after = json.dumps(result, sort_keys=True, allow_nan=False)
    if before != after:
        raise RuntimeError("Metadata normalization changed serialized scientific values")
    return result, changed


def prepare_wrapper(original, record):
    @functools.wraps(original)
    def prepare(request):
        model, transform, checkpoint, preprocess = original(request)
        normalized, changed = normalize_metadata(preprocess)
        record(
            {
                "event": "PROCESSOR_METADATA_CHECKED",
                "model_kind": request.model_kind,
                "view": request.view,
                "converted_fields": changed,
                "serialized_metadata_unchanged": True,
            }
        )
        return model, transform, checkpoint, normalized

    return prepare


@contextlib.contextmanager
def metadata_adapter(record):
    import hac.polar_locked_features as features

    original = features.prepare_model_and_transform
    features.prepare_model_and_transform = prepare_wrapper(original, record)
    try:
        yield
    finally:
        features.prepare_model_and_transform = original


def preflight(run, report_path):
    """Check all four backbones/eight metadata contracts on synthetic pixels only."""
    import numpy as np
    import torch
    from PIL import Image

    from hac.polar_benchmark_features import FeatureRequest, prepare_model_and_transform
    from hac.polar_locked_evaluation import verify_selection_lock
    from hac.polar_locked_features import HEAD_MODELS, HEAD_VIEWS

    lock = verify_selection_lock(run / "final_selection_lock.json")
    if not torch.cuda.is_available():
        raise RuntimeError("The preflight requires CUDA, with no silent CPU fallback")
    image = Image.fromarray(
        np.random.default_rng(20260921).integers(0, 256, (257, 301, 3), np.uint8)
    )
    results = []
    for model_kind in HEAD_MODELS:
        cache = Path(lock["development_feature_root"]) / model_kind
        references = {
            view: status_io.read_json(cache / view / "contract.json") for view in HEAD_VIEWS
        }
        reference = references[HEAD_VIEWS[0]]
        request = FeatureRequest(
            manifest=Path("unused_synthetic_preflight"),
            output_dir=Path("unused_synthetic_preflight"),
            model_kind=model_kind,
            view=HEAD_VIEWS[0],
            preprocess=reference["preprocess"]["policy"],
            cache_dir=Path(lock["hf_cache_dir"]) if lock.get("hf_cache_dir") else None,
            allow_download=False,
        )
        original = prepare_model_and_transform(request)
        events = []
        wrapped = prepare_wrapper(lambda _, prepared=original: prepared, events.append)(request)
        model, transform, checkpoint, normalized = wrapped
        if not all(wrapped[index] is original[index] for index in range(3)):
            raise RuntimeError("Model/transform/checkpoint object identity changed")
        for view, expected in references.items():
            if checkpoint != expected["checkpoint"] or normalized != expected["preprocess"]:
                raise RuntimeError(f"Development contract still differs: {model_kind}/{view}")
        before, after = original[1](image), transform(image)
        if not torch.equal(before, after):
            raise RuntimeError("Synthetic preprocessing tensors changed")
        torch.set_float32_matmul_precision(reference["matmul_precision"])
        model = model.to("cuda").eval().requires_grad_(False)
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=getattr(torch, reference["autocast_dtype"])),
        ):
            first = model(before.unsqueeze(0).to("cuda")).float().cpu()
            second = model(after.unsqueeze(0).to("cuda")).float().cpu()
        if not torch.isfinite(first).all() or not torch.equal(first, second):
            raise RuntimeError("Synthetic CUDA feature outputs changed")
        row = {
            "model_kind": model_kind,
            "views": list(HEAD_VIEWS),
            "converted_fields": events[0]["converted_fields"],
            "checkpoint_and_development_metadata_match": True,
            "model_transform_checkpoint_same_objects": True,
            "synthetic_pixel_max_difference": 0.0,
            "synthetic_feature_max_difference": 0.0,
            "feature_shape": list(first.shape),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        del model, transform, original, wrapped, first, second
        gc.collect()
        torch.cuda.empty_cache()
    report = {
        "status": "PASS",
        "scope": "synthetic_pixels_only_no_POLAR_images_labels_or_performance",
        "selection_lock_sha256": status_io.digest(run / "final_selection_lock.json"),
        "cuda_device": torch.cuda.get_device_name(),
        "checks": results,
        "adapter_files": {
            name: status_io.digest(Path(lock["repository_root"]) / name) for name in FILES
        },
        "created_utc": datetime.now(UTC).isoformat(),
    }
    status_io.create_json(report_path, report)
    return report


def prepare(run, recovery, parent_manifest, preflight_path):
    from hac.polar_locked_evaluation import verify_test_access

    run, recovery = run.resolve(strict=True), recovery.resolve()
    parent_manifest = parent_manifest.resolve(strict=True)
    parent, plan = status_io.validate_manifest(parent_manifest)
    if run != Path(parent["run_directory"]) or not recovery.is_relative_to(run) or recovery == run:
        raise ValueError("Recovery must be a new child directory of the original run")
    if (run / "queue_process.lock").exists():
        raise RuntimeError("A supervisor lock exists; do not start a duplicate")
    if status_io.read_json(run / "queue_status.json").get("status") != "FAILED_STOPPED":
        raise RuntimeError("Recovery requires the recorded stopped queue")
    verify_test_access(run / "final_selection_lock.json", run)
    receipts = status_io.read_json(run / "queue_receipts.json")["jobs"]
    expected_prefix = [job["id"] for job in plan["jobs"][:-3]]
    if [item["job"] for item in receipts] != expected_prefix:
        raise RuntimeError("All 23 fits and the original test gate must already be complete")
    if tuple(job["id"] for job in plan["jobs"][-3:]) != REMAINING_JOBS:
        raise RuntimeError("Only the three original evaluation jobs may remain")
    for receipt in receipts:
        if (
            receipt.get("status") != "COMPLETE"
            or status_io.digest(receipt["marker"]) != receipt["marker_sha256"]
        ):
            raise RuntimeError("Completed receipt changed")
    root = Path(parent["repository_root"])
    for relative, expected in status_io.read_json(run / "protected_historical_files.json").items():
        if status_io.digest(root / relative) != expected:
            raise RuntimeError(f"Protected historical artifact changed: {relative}")
    report = status_io.read_json(preflight_path)
    source_hashes = {name: status_io.digest(root / name) for name in FILES}
    if (
        report.get("status") != "PASS"
        or report.get("selection_lock_sha256")
        != status_io.digest(run / "final_selection_lock.json")
        or report.get("adapter_files") != source_hashes
        or {item["model_kind"] for item in report["checks"]}
        != {"dinov2_base", "dinov3_base", "siglip2_base", "convnextv2_base"}
    ):
        raise RuntimeError("A matching four-backbone preflight is required")
    recovery.mkdir(parents=True, exist_ok=False)
    originals = [
        run / name
        for name in (
            "queue_status.json",
            "queue_receipts.json",
            "final_fits_verified.json",
            "test_access_gate.json",
            "logs/extract_locked_test_features.log",
        )
    ]
    originals += [
        parent_manifest.parent / name for name in ("supervisor.stderr.log", "supervisor.stdout.log")
    ]
    originals += [
        run / "test_features/dinov2_base" / view / name
        for view in ("full_frame", "person_context_10")
        for name in ("contract.json", "provenance.json", "progress.json")
    ]
    preserved = {}
    for path in originals:
        relative = path.relative_to(run)
        target = recovery / "before_resume" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if status_io.digest(path) != status_io.digest(target):
            raise RuntimeError("Recovery evidence copy failed verification")
        preserved[relative.as_posix()] = status_io.digest(target)
    shutil.copy2(preflight_path, recovery / "preflight.json")
    value = {
        "status": STATUS,
        "authorization": "User: complete and make sure all is running correctly",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_directory": str(run),
        "repository_root": str(root),
        "parent_manifest": str(parent_manifest.resolve()),
        "parent_manifest_sha256": status_io.digest(parent_manifest),
        "selection_lock_sha256": status_io.digest(run / "final_selection_lock.json"),
        "queue_plan_sha256": status_io.digest(run / "queue_plan.json"),
        "final_fits_verified_sha256": status_io.digest(run / "final_fits_verified.json"),
        "test_access_gate_sha256": status_io.digest(run / "test_access_gate.json"),
        "adapter_files": source_hashes,
        "preflight_sha256": status_io.digest(recovery / "preflight.json"),
        "policy": POLICY,
        "remaining_jobs": list(REMAINING_JOBS),
        "preserved_artifacts": preserved,
        "scientific_source_changes": False,
        "test_gate_already_open": True,
        "performance_scores_inspected_before_repair": False,
    }
    status_io.create_json(recovery / "manifest.json", value)
    return value


def validate_manifest(path):
    value = status_io.read_json(path)
    run, root = Path(value["run_directory"]), Path(value["repository_root"])
    if not path.resolve().is_relative_to(run.resolve()) or value.get("status") != STATUS:
        raise ValueError("Invalid metadata recovery manifest")
    if value.get("policy") != POLICY or value.get("remaining_jobs") != list(REMAINING_JOBS):
        raise ValueError("Metadata-only recovery scope cannot drift")
    if set(value["adapter_files"]) != set(FILES):
        raise ValueError("Required recovery sources are missing")
    for relative, expected in value["adapter_files"].items():
        if status_io.digest(root / relative) != expected:
            raise RuntimeError(f"Recovery source changed: {relative}")
    for filename, field in (
        ("final_selection_lock.json", "selection_lock_sha256"),
        ("queue_plan.json", "queue_plan_sha256"),
        ("final_fits_verified.json", "final_fits_verified_sha256"),
        ("test_access_gate.json", "test_access_gate_sha256"),
    ):
        if status_io.digest(run / filename) != value[field]:
            raise RuntimeError(f"Original locked evidence changed: {filename}")
    parent = Path(value["parent_manifest"])
    if status_io.digest(parent) != value["parent_manifest_sha256"]:
        raise RuntimeError("Original status recovery manifest changed")
    if status_io.digest(path.parent / "preflight.json") != value["preflight_sha256"]:
        raise RuntimeError("Recovery preflight changed")
    _, plan = status_io.validate_manifest(parent)
    return value, plan


@contextlib.contextmanager
def wrapped_children(plan, manifest, record):
    original = subprocess.Popen
    commands = {tuple(job["command"]): job["id"] for job in plan["jobs"]}

    def launch(command, *args, **kwargs):
        key = tuple(command) if isinstance(command, (list, tuple)) else None
        if key in commands:
            job = commands[key]
            if job not in REMAINING_JOBS:
                raise RuntimeError("Metadata recovery cannot launch any training or gate job")
            wrapped = [
                command[0],
                str(Path(__file__).resolve()),
                "--manifest",
                str(manifest),
                "--job-id",
                job,
            ]
            record(
                {
                    "event": "LOCKED_CHILD_WRAPPED",
                    "job": job,
                    "original_command": list(command),
                    "executed_command": wrapped,
                }
            )
            command = wrapped
        return original(command, *args, **kwargs)

    subprocess.Popen = launch
    try:
        yield
    finally:
        subprocess.Popen = original


def execute(manifest_path, job_id=None):
    value, plan = validate_manifest(manifest_path)
    run = Path(value["run_directory"])
    event_file = manifest_path.parent / f"runtime_{os.getpid()}.jsonl"
    manifest_sha = status_io.digest(manifest_path)

    def record(event):
        with event_file.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "utc": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),
                        "manifest_sha256": manifest_sha,
                        **event,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    if job_id is None:
        command = [
            plan["jobs"][0]["command"][0],
            str(Path(value["repository_root"]) / "experiments/run_polar_locked_queue.py"),
            "--selection-lock",
            str(run / "final_selection_lock.json"),
        ]
    else:
        matches = [job["command"] for job in plan["jobs"] if job["id"] == job_id]
        if job_id not in REMAINING_JOBS or len(matches) != 1:
            raise ValueError("Only original remaining evaluation jobs may execute")
        command = matches[0]
    if Path(sys.executable).resolve() != Path(command[0]).resolve():
        raise RuntimeError("Recovery must use the original Python environment")
    record(
        {"event": "RECOVERY_STARTED", "job": job_id or "supervisor", "original_command": command}
    )
    children = (
        wrapped_children(plan, manifest_path, record)
        if job_id is None
        else contextlib.nullcontext()
    )
    adapter = (
        metadata_adapter(record)
        if job_id == "extract_locked_test_features"
        else contextlib.nullcontext()
    )
    with status_io.status_retry(run, record=record), children, adapter:
        sys.argv = command[1:]
        sys.path.insert(0, str(Path(command[1]).parent))
        runpy.run_path(command[1], run_name="__main__")
    record({"event": "RECOVERY_COMPLETED", "job": job_id or "supervisor"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight-run", type=Path)
    mode.add_argument("--prepare-run", type=Path)
    mode.add_argument("--manifest", type=Path)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--recovery-dir", type=Path)
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--job-id", choices=REMAINING_JOBS)
    args = parser.parse_args()
    if args.preflight_run:
        if not args.preflight_report:
            parser.error("Preflight requires --preflight-report")
        preflight(args.preflight_run.resolve(strict=True), args.preflight_report)
    elif args.prepare_run:
        if not all((args.recovery_dir, args.parent_manifest, args.preflight_report)):
            parser.error(
                "Preparation requires recovery directory, parent manifest and preflight report"
            )
        value = prepare(
            args.prepare_run, args.recovery_dir, args.parent_manifest, args.preflight_report
        )
        print(json.dumps({"status": value["status"], "remaining_jobs": value["remaining_jobs"]}))
    else:
        execute(args.manifest.resolve(strict=True), args.job_id)


if __name__ == "__main__":
    main()
