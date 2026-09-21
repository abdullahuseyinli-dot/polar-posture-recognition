"""Audited, telemetry-only Windows rename retry for an already sealed POLAR run.

Original lock, plan, scientific sources and requests remain byte-identical.
This explicitly recorded runtime adapter wraps the original queue/commands;
it never retries training, changes epochs, or bypasses an integrity check.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import runpy
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

STATUS_NAMES = (
    "progress.json",
    "queue_status.json",
    "queue_receipts.json",
    "inference_progress.json",
)
RETRY_POLICY = {
    "winerrors": [5, 32, 33],
    "deadline_seconds": 5.0,
    "initial_delay_seconds": 0.01,
    "maximum_delay_seconds": 0.25,
}


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def create_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def status_swap(source: Path, target: Path, run: Path) -> bool:
    source, target, run = source.resolve(), target.resolve(), run.resolve()
    return (
        source.parent == target.parent
        and source.is_relative_to(run)
        and target.name in STATUS_NAMES
        and source.name == f"{target.name}.{os.getpid()}.tmp"
    )


@contextlib.contextmanager
def status_retry(run: Path, *, policy=None, record=None):
    """Retry only known transient Windows errors on exact telemetry temp swaps."""
    policy = dict(RETRY_POLICY if policy is None else policy)
    original = Path.replace

    def replace(source, target):
        if not status_swap(Path(source), Path(target), run):
            return original(source, target)
        deadline, delay, retries = (
            time.monotonic() + policy["deadline_seconds"],
            policy["initial_delay_seconds"],
            0,
        )
        while True:
            try:
                result = original(source, target)
                if retries and record:
                    record(
                        {
                            "event": "STATUS_REPLACE_RECOVERED",
                            "target": str(target),
                            "retries": retries,
                        }
                    )
                return result
            except PermissionError as error:
                if (
                    getattr(error, "winerror", None) not in policy["winerrors"]
                    or time.monotonic() >= deadline
                ):
                    raise
                retries += 1
                if record:
                    record(
                        {
                            "event": "STATUS_REPLACE_RETRY",
                            "target": str(target),
                            "winerror": error.winerror,
                            "attempt": retries,
                        }
                    )
                time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
                delay = min(delay * 2, policy["maximum_delay_seconds"])

    Path.replace = replace
    try:
        yield
    finally:
        Path.replace = original


def prepare(run: Path, recovery: Path):
    from hac.polar_benchmark import canonical_hash
    from hac.polar_locked_evaluation import verify_selection_lock

    run, recovery = run.resolve(strict=True), recovery.resolve()
    if not recovery.is_relative_to(run) or recovery == run:
        raise ValueError("Recovery evidence must use a new child directory of the run")
    if (run / "queue_process.lock").exists() or (run / "test_access_gate.json").exists():
        raise RuntimeError("Preparation requires a stopped, pre-test queue")
    lock_path = run / "final_selection_lock.json"
    lock = verify_selection_lock(lock_path)
    status = read_json(run / "queue_status.json")
    if status.get("status") != "FAILED_STOPPED":
        raise RuntimeError("This incident recovery is only for the recorded stopped queue")
    directory = run / "polar9/neural/dinov2_base/seed52"
    request = read_json(directory / "request.json")
    sidecar = read_json(directory / "last_checkpoint.json")
    if (
        sidecar["sha256"] != digest(directory / "last.pt")
        or sidecar["history_sha256"] != digest(directory / "history.csv")
        or sidecar["request_sha256"] != canonical_hash(request)
        or sidecar["lock_sha256"] != digest(lock_path)
    ):
        raise RuntimeError("Checkpoint/request/history does not match the frozen resume state")
    recovery.mkdir(parents=True, exist_ok=False)
    originals = [
        run / name
        for name in (
            "queue_status.json",
            "queue_receipts.json",
            "supervisor.stderr.log",
            "supervisor.stdout.log",
        )
    ]
    originals += [run / "logs/polar9_neural_dinov2_base_seed52.log"]
    originals += [
        directory / name
        for name in (
            "request.json",
            "history.csv",
            "last.pt",
            "last_checkpoint.json",
            "progress.json",
        )
    ]
    originals += list(directory.glob("progress.json.*.tmp"))
    preserved = {}
    for path in originals:
        relative = path.relative_to(run)
        target = recovery / "before_resume" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        original_hash = digest(path)
        if digest(target) != original_hash:
            raise RuntimeError("Preserved incident snapshot failed its copy checksum")
        preserved[relative.as_posix()] = original_hash
    root = Path(lock["repository_root"])
    files = (
        "tools/polar_status_io_recovery.py",
        "tools/polar_locked_status.py",
        "tests/test_polar_status_io_recovery.py",
        "tests/test_polar_locked_status.py",
    )
    manifest = {
        "status": "USER_AUTHORIZED_TELEMETRY_ONLY_RECOVERY",
        "created_utc": datetime.now(UTC).isoformat(),
        "authorization": "User: fix and run as needed; no scientific recipe or test-gate amendment",
        "run_directory": str(run),
        "repository_root": str(root),
        "selection_lock_sha256": digest(lock_path),
        "queue_plan_sha256": digest(run / "queue_plan.json"),
        "locked_implementation_sha256": canonical_hash(lock["implementation"]),
        "adapter_files": {name: digest(root / name) for name in files},
        "status_filenames": list(STATUS_NAMES),
        "retry_policy": RETRY_POLICY,
        "preserved_artifacts": preserved,
        "resume_fit": "polar9/neural/dinov2_base/seed52",
        "resume_after_epoch": 13,
        "resume_at_epoch": 14,
        "locked_final_epoch": 16,
        "checkpoint_optimizer_scheduler_scaler": "separately_verified_present",
        "scientific_source_changes": False,
        "automatic_training_retry": False,
        "test_gate_open_at_authorization": False,
        "scope": "Path.replace transient Windows permission errors on same-directory PID-scoped telemetry JSON temp files only",
    }
    create_json(recovery / "manifest.json", manifest)
    return manifest


def validate_manifest(path: Path):
    from hac.polar_locked_evaluation import verify_selection_lock

    value = read_json(path)
    run, root = Path(value["run_directory"]), Path(value["repository_root"])
    if (
        not path.resolve().is_relative_to(run.resolve())
        or value.get("status") != "USER_AUTHORIZED_TELEMETRY_ONLY_RECOVERY"
    ):
        raise ValueError("Invalid recovery authorization manifest")
    if value["retry_policy"] != RETRY_POLICY or value["status_filenames"] != list(STATUS_NAMES):
        raise ValueError("Recovery retry scope cannot drift")
    for relative, expected in value["adapter_files"].items():
        candidate = (root / relative).resolve()
        if not candidate.is_relative_to(root.resolve()) or digest(candidate) != expected:
            raise RuntimeError(f"Recovery adapter source changed: {relative}")
    if (
        digest(run / "final_selection_lock.json") != value["selection_lock_sha256"]
        or digest(run / "queue_plan.json") != value["queue_plan_sha256"]
    ):
        raise RuntimeError("Original selection lock or queue plan changed")
    verify_selection_lock(run / "final_selection_lock.json")
    return value, read_json(run / "queue_plan.json")


@contextlib.contextmanager
def wrapped_children(plan, manifest, record):
    """Wrap only exact commands from the original verified finite queue plan."""
    original = subprocess.Popen
    commands = {tuple(job["command"]): job["id"] for job in plan["jobs"]}

    def launch(command, *args, **kwargs):
        key = tuple(command) if isinstance(command, (list, tuple)) else None
        if key in commands:
            wrapped = [
                command[0],
                str(Path(__file__).resolve()),
                "--manifest",
                str(manifest),
                "--job-id",
                commands[key],
            ]
            record(
                {
                    "event": "LOCKED_CHILD_WRAPPED",
                    "job": commands[key],
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


def execute(manifest_path: Path, job_id=None):
    value, plan = validate_manifest(manifest_path)
    run = Path(value["run_directory"])
    event_file = manifest_path.parent / f"runtime_{os.getpid()}.jsonl"

    def record(event):
        with event_file.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "utc": datetime.now(UTC).isoformat(),
                        "pid": os.getpid(),
                        "manifest_sha256": digest(manifest_path),
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
        if len(matches) != 1:
            raise ValueError("Recovery may execute only a uniquely registered original job")
        command = matches[0]
    if Path(sys.executable).resolve() != Path(command[0]).resolve():
        raise RuntimeError("Recovery must use the unchanged locked Python environment")
    record(
        {
            "event": "RUNTIME_ADAPTER_STARTED",
            "job": job_id or "supervisor",
            "original_command": command,
        }
    )
    children = (
        wrapped_children(plan, manifest_path, record)
        if job_id is None
        else contextlib.nullcontext()
    )
    with status_retry(run, record=record), children:
        sys.argv = command[1:]
        sys.path.insert(0, str(Path(command[1]).parent))
        runpy.run_path(command[1], run_name="__main__")
    record({"event": "RUNTIME_ADAPTER_COMPLETED", "job": job_id or "supervisor"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--prepare-run", type=Path)
    parser.add_argument("--recovery-dir", type=Path)
    parser.add_argument("--job-id")
    args = parser.parse_args()
    if args.prepare_run:
        if not args.recovery_dir or args.manifest or args.job_id:
            parser.error("Preparation requires only --prepare-run and --recovery-dir")
        value = prepare(args.prepare_run, args.recovery_dir)
        print(json.dumps({"status": value["status"], "resume_at_epoch": value["resume_at_epoch"]}))
    elif args.manifest and not args.recovery_dir:
        execute(args.manifest.resolve(strict=True), args.job_id)
    else:
        parser.error("Provide a recovery manifest to execute")


if __name__ == "__main__":
    main()
