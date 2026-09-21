"""Finite, single-GPU final refits followed by one sealed evaluation/comparison panel."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hac.polar import sha256_file
from hac.polar_benchmark import atomic_json, lock_json, utc_now
from hac.polar_locked_evaluation import (
    expected_fits,
    open_test_gate,
    read_json,
    verify_file,
    verify_fit,
    verify_selection_lock,
)


def build_plan(selection_lock: Path, *, engineering_smokes=True) -> dict:
    lock = verify_selection_lock(selection_lock)
    run, root = selection_lock.parent, Path(lock["repository_root"])
    python = str(Path(sys.executable).resolve())
    jobs = []

    def add(identifier, script, args, marker, statuses, stage):
        jobs.append(
            {
                "id": identifier,
                "command": [python, str(root / "experiments" / script), *map(str, args)],
                "marker": str(marker),
                "allowed_statuses": statuses,
                "stage": stage,
                "timeout_seconds": lock["resource_policy"]["job_timeout_seconds"],
            }
        )

    if engineering_smokes:
        for task, spec in lock["tasks"].items():
            for model in spec["neural_fits"]:
                output = run / "smokes" / f"{task}_{model}"
                add(
                    f"smoke_{task}_{model}",
                    "train_polar_locked.py",
                    [
                        "--lock",
                        selection_lock,
                        "--task",
                        task,
                        "--model-kind",
                        model,
                        "--seed",
                        42,
                        "--output-dir",
                        output,
                        "--workers",
                        4,
                        "--smoke",
                    ],
                    output / "summary.json",
                    ["COMPLETE"],
                    "engineering_smoke",
                )
    for task, spec in lock["tasks"].items():
        # Validate all frozen-feature refits before expensive neural refits in each task.
        for model, fit in spec["head_fits"].items():
            output = run / fit["output_dir"]
            add(
                f"{task}_head_{model}",
                "fit_polar_locked_head.py",
                [
                    "--selection-lock",
                    selection_lock,
                    "--task",
                    task,
                    "--model-kind",
                    model,
                    "--output-dir",
                    output,
                ],
                output / "summary.json",
                ["COMPLETE"],
                "final_fit",
            )
        for model, fit in spec["neural_fits"].items():
            for seed in fit["seeds"]:
                output = run / fit["output_dir_pattern"].format(seed=seed)
                add(
                    f"{task}_neural_{model}_seed{seed}",
                    "train_polar_locked.py",
                    [
                        "--lock",
                        selection_lock,
                        "--task",
                        task,
                        "--model-kind",
                        model,
                        "--seed",
                        seed,
                        "--output-dir",
                        output,
                        "--workers",
                        4,
                    ],
                    output / "summary.json",
                    ["COMPLETE"],
                    "final_fit",
                )
    add(
        "verify_all_fits_and_open_gate",
        "run_polar_locked_queue.py",
        ["--selection-lock", selection_lock, "--open-gate"],
        run / "test_access_gate.json",
        ["FINAL_TEST_ACCESS_OPENED"],
        "test_gate",
    )
    add(
        "extract_locked_test_features",
        "cache_polar_locked_test_features.py",
        ["--selection-lock", selection_lock, "--run-dir", run],
        run / "test_features" / "summary.json",
        ["COMPLETE"],
        "test_inference",
    )
    add(
        "predict_locked_candidates",
        "evaluate_polar_locked.py",
        ["--selection-lock", selection_lock],
        run / "evaluation" / "summary.json",
        ["LOCKED_PREDICTIONS_COMPLETE"],
        "test_inference",
    )
    add(
        "compare_locked_candidates",
        "analyze_polar_locked.py",
        [
            "--selection-lock",
            selection_lock,
            "--prediction-manifest",
            run / "evaluation" / "prediction_manifest.json",
            "--output-dir",
            run / "comparisons",
        ],
        run / "comparisons" / "summary.json",
        ["POLAR_LOCKED_STATISTICS_COMPLETE"],
        "statistical_comparison",
    )
    return {
        "selection_lock": str(selection_lock),
        "selection_lock_sha256": sha256_file(selection_lock),
        "working_directory": str(root),
        "run_directory": str(run),
        "jobs": jobs,
    }


def stop_owned(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True
        )
    else:
        process.kill()
    process.wait(timeout=30)


def verify_protected(run: Path, root: Path):
    for relative, digest in read_json(run / "protected_historical_files.json").items():
        verify_file(root / relative, digest)


def execute(plan_path: Path):
    plan = read_json(plan_path)
    selection_lock = Path(plan["selection_lock"])
    if plan != build_plan(selection_lock):
        raise RuntimeError("Queue commands/order/paths differ from the deterministic locked plan")
    run, root = Path(plan["run_directory"]), Path(plan["working_directory"])
    process_lock = run / "queue_process.lock"
    with process_lock.open("x", encoding="utf-8") as handle:
        json.dump(
            {"pid": os.getpid(), "started_utc": utc_now(), "plan_sha256": sha256_file(plan_path)},
            handle,
        )
    active = None
    receipts = (
        read_json(run / "queue_receipts.json").get("jobs", [])
        if (run / "queue_receipts.json").exists()
        else []
    )
    completed = {item["job"]: item for item in receipts}
    started = utc_now()
    try:
        verify_file(selection_lock, plan["selection_lock_sha256"])
        lock = verify_selection_lock(selection_lock)
        verify_protected(run, root)
        fit_map = {
            identifier: (run / relative, kind) for identifier, relative, kind in expected_fits(lock)
        }
        environment = {
            **os.environ,
            "PYTHONPATH": str(root / "src"),
            "PYTHONUNBUFFERED": "1",
            "OMP_NUM_THREADS": "4",
            "MKL_NUM_THREADS": "4",
            "OPENBLAS_NUM_THREADS": "4",
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
        logs = run / "logs"
        logs.mkdir(exist_ok=True)
        for index, job in enumerate(plan["jobs"]):
            verify_file(selection_lock, plan["selection_lock_sha256"])
            verify_selection_lock(selection_lock)
            marker = Path(job["marker"])
            if job["id"] in completed:
                verify_file(marker, completed[job["id"]]["marker_sha256"])
                if job["id"] in fit_map:
                    directory, kind = fit_map[job["id"]]
                    verify_fit(directory, plan["selection_lock_sha256"], kind)
                continue
            if (run / "STOP_AFTER_CURRENT_JOB").exists():
                atomic_json(
                    run / "queue_status.json",
                    {"status": "STOP_REQUESTED", "completed_jobs": len(receipts), "utc": utc_now()},
                )
                return
            if (
                job["stage"] in {"final_fit", "engineering_smoke"}
                and (run / "test_access_gate.json").exists()
            ):
                raise RuntimeError("No missing/new training job may run after test access")
            log_path = logs / f"{job['id']}.log"
            before = time.monotonic()
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\nSupervisor launch {utc_now()}\n")
                handle.flush()
                active = subprocess.Popen(
                    job["command"],
                    cwd=root,
                    env=environment,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                while active.poll() is None:
                    elapsed = time.monotonic() - before
                    atomic_json(
                        run / "queue_status.json",
                        {
                            "status": "RUNNING",
                            "job": job["id"],
                            "stage": job["stage"],
                            "job_index": index + 1,
                            "total_jobs": len(plan["jobs"]),
                            "completed_jobs": len(receipts),
                            "child_pid": active.pid,
                            "supervisor_pid": os.getpid(),
                            "job_elapsed_seconds": elapsed,
                            "log": str(log_path),
                            "started_utc": started,
                            "utc": utc_now(),
                            "test_gate_open": (run / "test_access_gate.json").exists(),
                        },
                    )
                    if elapsed > job["timeout_seconds"]:
                        stop_owned(active)
                        raise TimeoutError(f"Finite job deadline exceeded: {job['id']}")
                    time.sleep(10)
                if active.returncode:
                    raise RuntimeError(f"{job['id']} failed ({active.returncode}); see {log_path}")
            if read_json(marker).get("status") not in job["allowed_statuses"]:
                raise RuntimeError(
                    f"Job did not produce the declared completion evidence: {job['id']}"
                )
            receipt = {
                "job": job["id"],
                "status": "COMPLETE",
                "marker": str(marker),
                "marker_sha256": sha256_file(marker),
                "seconds": time.monotonic() - before,
            }
            receipts.append(receipt)
            atomic_json(run / "queue_receipts.json", {"jobs": receipts})
            active = None
        verify_protected(run, root)
        summary = read_json(run / "comparisons" / "summary.json")
        if summary["selection_lock_sha256"] != sha256_file(selection_lock):
            raise RuntimeError("Final comparison is bound to a different selection lock")
        completion = {
            "status": "LOCKED_FINAL_EVALUATION_COMPLETE",
            "selection_lock_sha256": sha256_file(selection_lock),
            "comparison_summary_sha256": sha256_file(run / "comparisons" / "summary.json"),
            "completed_jobs": len(receipts),
            "historical_files_unchanged": True,
            "test_used_for_selection": False,
            "automatic_promotion": False,
            "completed_utc": utc_now(),
        }
        if (run / "completion.json").exists():
            previous = read_json(run / "completion.json")
            if any(
                previous.get(key) != value
                for key, value in completion.items()
                if key != "completed_utc"
            ):
                raise RuntimeError("Completed evaluation evidence has changed")
            completion = previous
        lock_json(run / "completion.json", completion)
        atomic_json(run / "queue_status.json", completion)
    except BaseException as error:
        if active is not None and active.poll() is None:
            stop_owned(active)
        atomic_json(
            run / "queue_status.json",
            {
                "status": "FAILED_STOPPED",
                "message": str(error),
                "error_type": type(error).__name__,
                "completed_jobs": len(receipts),
                "test_gate_open": (run / "test_access_gate.json").exists(),
                "utc": utc_now(),
            },
        )
        raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            process_lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--open-gate", action="store_true")
    args = parser.parse_args()
    selection_lock = args.selection_lock.resolve(strict=True)
    plan_path = selection_lock.parent / "queue_plan.json"
    if args.open_gate:
        open_test_gate(selection_lock)
    elif args.write_plan:
        plan = build_plan(selection_lock)
        lock_json(plan_path, plan)
        print(
            json.dumps(
                {"status": "FINAL_QUEUE_LOCKED", "jobs": len(plan["jobs"]), "path": str(plan_path)},
                indent=2,
            )
        )
    else:
        execute(plan_path)


if __name__ == "__main__":
    main()
