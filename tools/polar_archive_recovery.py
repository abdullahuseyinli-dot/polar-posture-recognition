"""Resume a sealed evaluation with a non-executing legacy string-ID decoder.

Only the hash-bound historical archive's image_ids array is adapted. NumPy's
allow_pickle=False remains mandatory; no pickle instructions are executed.
All original models, sources, selection checks and comparisons remain locked.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import pickletools
import runpy
import shutil
import subprocess
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polar_metadata_recovery as parent_recovery
import polar_status_io_recovery as status_io

STATUS = "USER_AUTHORIZED_LEGACY_ID_RECOVERY"
JOBS = ("predict_locked_candidates", "compare_locked_candidates")
FILES = ("tools/polar_archive_recovery.py", "tests/test_polar_archive_recovery.py")
POLICY = {
    "adapted_member": "image_ids.npy",
    "decoder": "passive_pickletools_literal_strings_exact_numpy_object_vector_envelope",
    "pickle_execution": False,
    "numeric_arrays_changed": False,
    "scientific_source_changes": False,
    "training_or_extraction_allowed": False,
}


def decode_string_vector(npy_bytes: bytes, rows: int) -> np.ndarray:
    """Interpret a deliberately narrow legacy envelope; never call an unpickler."""
    if not 1 <= rows <= 10000 or len(npy_bytes) > 1024 * 1024:
        raise ValueError("Legacy string vector exceeds the bounded format")
    stream = io.BytesIO(npy_bytes)
    if np.lib.format.read_magic(stream) != (1, 0):
        raise ValueError("Unsupported legacy NPY version")
    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
    if shape != (rows,) or fortran or dtype != np.dtype(object):
        raise ValueError("Legacy ID vector header differs from the declared cohort")
    payload = stream.read()
    operations = list(pickletools.genops(payload))
    if not operations or operations[0][:2] != (pickletools.code2op["\x80"], 4):
        raise ValueError("Only the observed protocol-4 string vector is supported")
    if operations[-1][0].name != "STOP" or operations[-1][2] + 1 != len(payload):
        raise ValueError("Trailing or incomplete legacy object data")

    # These are inert metadata tokens, NOT functions to import or call. The
    # reference memo index is part of the observed protocol-4 ndarray envelope.
    prefix = [
        ("STRING", "numpy._core.multiarray"),
        ("STRING", "_reconstruct"),
        ("STACK_GLOBAL", None),
        ("STRING", "numpy"),
        ("STRING", "ndarray"),
        ("STACK_GLOBAL", None),
        ("INT", 0),
        ("TUPLE1", None),
        ("SHORT_BINBYTES", b"b"),
        ("TUPLE3", None),
        ("REDUCE", None),
        ("MARK", None),
        ("INT", 1),
        ("INT", rows),
        ("TUPLE1", None),
        ("BINGET", 3),
        ("STRING", "dtype"),
        ("STACK_GLOBAL", None),
        ("STRING", "O8"),
        ("NEWFALSE", None),
        ("NEWTRUE", None),
        ("TUPLE3", None),
        ("REDUCE", None),
        ("MARK", None),
        ("INT", 3),
        ("STRING", "|"),
        ("NONE", None),
        ("NONE", None),
        ("NONE", None),
        ("INT", -1),
        ("INT", -1),
        ("INT", 63),
        ("TUPLE", None),
        ("BUILD", None),
        ("NEWFALSE", None),
        ("EMPTY_LIST", None),
    ]
    tokens = []
    for operation, argument, _ in operations[1:]:
        name = operation.name
        if name in {"FRAME", "MEMOIZE"}:
            continue
        if name in {"BININT", "BININT1", "BININT2"}:
            name = "INT"
        elif name in {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8"}:
            name = "STRING"
        tokens.append((name, argument))
    if tokens[: len(prefix)] != prefix or tokens[-3:] != [
        ("TUPLE", None),
        ("BUILD", None),
        ("STOP", None),
    ]:
        raise ValueError("Unrecognized legacy NumPy envelope; no object code executed")
    strings, in_batch, batch_size = [], False, 0
    for name, value in tokens[len(prefix) : -3]:
        if name == "MARK" and not in_batch:
            in_batch, batch_size = True, 0
        elif name == "STRING" and in_batch:
            if type(value) is not str or not value or len(value) > 256 or "\x00" in value:
                raise ValueError("Invalid legacy image-ID string")
            strings.append(value)
            batch_size += 1
        elif name == "APPENDS" and in_batch and batch_size:
            in_batch = False
        else:
            raise ValueError("Legacy payload contains nonliteral or unsupported instructions")
    if in_batch or len(strings) != rows or len(set(strings)) != rows:
        raise ValueError("Legacy ID count, uniqueness or list structure differs")
    return np.asarray(strings, dtype=str)


def verified_archive(spec):
    """Read one immutable snapshot, verify it, then decode just the ID member."""
    path = Path(spec["path"])
    if path.stat().st_size > 32 * 1024 * 1024:
        raise ValueError("Historical archive exceeds the bounded size")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != spec["sha256"]:
        raise RuntimeError("Historical archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("Historical archive contains duplicate members")
        member = archive.getinfo("image_ids.npy")
        if member.file_size > 1024 * 1024:
            raise ValueError("Historical ID member exceeds the bounded size")
        ids = decode_string_vector(archive.read(member), spec["expected_rows"])
    return content, ids


class HistoricalArchive:
    def __init__(self, archive, ids):
        self.archive, self.ids = archive, ids

    def __enter__(self):
        self.archive.__enter__()
        return self

    def __exit__(self, *args):
        return self.archive.__exit__(*args)

    def __getitem__(self, name):
        return self.ids.copy() if name == "image_ids" else self.archive[name]

    def __getattr__(self, name):
        return getattr(self.archive, name)


@contextlib.contextmanager
def archive_adapter(spec, record):
    original = np.load
    target = Path(spec["path"]).resolve(strict=True)

    def load(file, *args, **kwargs):
        if isinstance(file, (str, os.PathLike)) and Path(file).resolve() == target:
            if args or kwargs.get("allow_pickle") is not False:
                raise ValueError("Historical compatibility requires explicit allow_pickle=False")
            content, ids = verified_archive(spec)
            record(
                {
                    "event": "LEGACY_ID_VECTOR_DECODED",
                    "rows": len(ids),
                    "archive_sha256": spec["sha256"],
                    "pickle_execution": False,
                    "numeric_arrays_changed": False,
                }
            )
            return HistoricalArchive(original(io.BytesIO(content), **kwargs), ids)
        return original(file, *args, **kwargs)

    np.load = load
    try:
        yield
    finally:
        np.load = original


def preflight(lock_path):
    """Exercise the ORIGINAL historical comparator and the actual archived cohort."""
    import pandas as pd

    from hac.polar_locked_evaluation import verify_test_access

    run = lock_path.parent
    verify_test_access(lock_path, run)
    lock = status_io.read_json(lock_path)
    spec = lock["historical_reference"]
    frame = pd.read_csv(
        lock["tasks"]["polar4"]["test_manifest"], dtype={"image_id": str}
    ).sort_values("image_id")
    evaluator = runpy.run_path(
        str(Path(lock["repository_root"]) / "experiments/evaluate_polar_locked.py")
    )
    records = []
    with archive_adapter(spec, records.append):
        probabilities = evaluator["historical_probabilities"](
            spec,
            frame.image_id.to_numpy(dtype=str),
            frame.label_index.to_numpy(dtype=int),
            lock["tasks"]["polar4"]["class_names"],
        )
    analysis = runpy.run_path(
        str(Path(lock["repository_root"]) / "experiments/analyze_polar_locked.py")
    )
    comparisons = analysis["validate_statistical_contract"](lock)
    return {
        "status": "PASS",
        "historical_rows": len(probabilities),
        "original_historical_replay_gate_passed": True,
        "expected_historical_macro_f1": spec["expected_macro_f1"],
        "comparison_contracts": len(comparisons),
        "events": records,
        "new_model_performance_inspected": False,
        "selection_lock_sha256": status_io.digest(lock_path),
    }


def prepare(run, recovery, parent_manifest):
    run, recovery, parent_manifest = (
        run.resolve(strict=True),
        recovery.resolve(),
        parent_manifest.resolve(strict=True),
    )
    parent, plan = parent_recovery.validate_manifest(parent_manifest)
    if run != Path(parent["run_directory"]) or not recovery.is_relative_to(run) or recovery == run:
        raise ValueError("Recovery must be a new child of the same run")
    if (run / "queue_process.lock").exists() or status_io.read_json(run / "queue_status.json").get(
        "status"
    ) != "FAILED_STOPPED":
        raise RuntimeError("Preparation requires the stopped queue without an active supervisor")
    receipts = status_io.read_json(run / "queue_receipts.json")["jobs"]
    if [row["job"] for row in receipts] != [job["id"] for job in plan["jobs"][:-2]] or tuple(
        job["id"] for job in plan["jobs"][-2:]
    ) != JOBS:
        raise RuntimeError("Only prediction and comparison may remain")
    for row in receipts:
        if row["status"] != "COMPLETE" or status_io.digest(row["marker"]) != row["marker_sha256"]:
            raise RuntimeError("Completed queue receipt changed")
    report = preflight(run / "final_selection_lock.json")
    root = Path(parent["repository_root"])
    for name, expected in status_io.read_json(run / "protected_historical_files.json").items():
        if status_io.digest(root / name) != expected:
            raise RuntimeError(f"Protected historical file changed: {name}")
    recovery.mkdir(parents=True, exist_ok=False)
    originals = [
        run / name
        for name in (
            "queue_status.json",
            "queue_receipts.json",
            "logs/predict_locked_candidates.log",
            "evaluation/inference_progress.json",
            "test_features/summary.json",
        )
    ]
    originals += list((run / "evaluation").rglob("*.json")) + list(
        (run / "evaluation").rglob("*.npz")
    )
    originals += [
        parent_manifest.parent / name for name in ("supervisor.stdout.log", "supervisor.stderr.log")
    ]
    preserved = {}
    for path in dict.fromkeys(originals):
        relative = path.relative_to(run)
        target = recovery / "before_resume" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        digest = status_io.digest(path)
        if status_io.digest(target) != digest:
            raise RuntimeError("Evidence copy checksum mismatch")
        preserved[relative.as_posix()] = digest
    status_io.create_json(recovery / "preflight.json", report)
    manifest = {
        "status": STATUS,
        "authorization": "User: complete and make sure all is running correctly",
        "created_utc": datetime.now(UTC).isoformat(),
        "run_directory": str(run),
        "repository_root": str(root),
        "parent_manifest": str(parent_manifest),
        "parent_manifest_sha256": status_io.digest(parent_manifest),
        "policy": POLICY,
        "remaining_jobs": list(JOBS),
        "adapter_files": {name: status_io.digest(root / name) for name in FILES},
        "preflight_sha256": status_io.digest(recovery / "preflight.json"),
        "preserved_artifacts": preserved,
    }
    status_io.create_json(recovery / "manifest.json", manifest)
    return manifest


def validate_manifest(path):
    value = status_io.read_json(path)
    run, root = Path(value["run_directory"]), Path(value["repository_root"])
    if (
        not path.resolve().is_relative_to(run)
        or value.get("status") != STATUS
        or value.get("policy") != POLICY
        or value.get("remaining_jobs") != list(JOBS)
    ):
        raise ValueError("Invalid or altered legacy-ID recovery scope")
    if set(value["adapter_files"]) != set(FILES):
        raise ValueError("Required recovery sources are missing")
    for name, expected in value["adapter_files"].items():
        if status_io.digest(root / name) != expected:
            raise RuntimeError(f"Recovery source changed: {name}")
    parent = Path(value["parent_manifest"])
    if (
        status_io.digest(parent) != value["parent_manifest_sha256"]
        or status_io.digest(path.parent / "preflight.json") != value["preflight_sha256"]
    ):
        raise RuntimeError("Parent recovery or preflight changed")
    _, plan = parent_recovery.validate_manifest(parent)
    return value, plan


@contextlib.contextmanager
def wrapped_children(plan, manifest, record):
    original = subprocess.Popen
    commands = {tuple(job["command"]): job["id"] for job in plan["jobs"]}

    def launch(command, *args, **kwargs):
        key = tuple(command) if isinstance(command, (list, tuple)) else None
        if key in commands:
            job = commands[key]
            if job not in JOBS:
                raise RuntimeError(
                    "Archive recovery cannot launch training, extraction or gate jobs"
                )
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
    manifest_sha = status_io.digest(manifest_path)
    event_file = manifest_path.parent / f"runtime_{os.getpid()}.jsonl"

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
        if job_id not in JOBS or len(matches) != 1:
            raise ValueError("Only the two original remaining jobs may execute")
        command = matches[0]
    if Path(sys.executable).resolve() != Path(command[0]).resolve():
        raise RuntimeError("The original Python environment is required")
    record(
        {"event": "RECOVERY_STARTED", "job": job_id or "supervisor", "original_command": command}
    )
    lock = status_io.read_json(run / "final_selection_lock.json")
    children = (
        wrapped_children(plan, manifest_path, record)
        if job_id is None
        else contextlib.nullcontext()
    )
    adapter = (
        archive_adapter(lock["historical_reference"], record)
        if job_id == "predict_locked_candidates"
        else contextlib.nullcontext()
    )
    with status_io.status_retry(run, record=record), children, adapter:
        sys.argv = command[1:]
        sys.path.insert(0, str(Path(command[1]).parent))
        runpy.run_path(command[1], run_name="__main__")
    record({"event": "RECOVERY_COMPLETED", "job": job_id or "supervisor"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prepare-run", type=Path)
    group.add_argument("--manifest", type=Path)
    parser.add_argument("--recovery-dir", type=Path)
    parser.add_argument("--parent-manifest", type=Path)
    parser.add_argument("--job-id", choices=JOBS)
    args = parser.parse_args()
    if args.prepare_run:
        if not args.recovery_dir or not args.parent_manifest or args.job_id:
            parser.error("Preparation requires recovery directory and parent manifest only")
        value = prepare(args.prepare_run, args.recovery_dir, args.parent_manifest)
        print(json.dumps({"status": value["status"], "remaining_jobs": value["remaining_jobs"]}))
    else:
        execute(args.manifest.resolve(strict=True), args.job_id)


if __name__ == "__main__":
    main()
