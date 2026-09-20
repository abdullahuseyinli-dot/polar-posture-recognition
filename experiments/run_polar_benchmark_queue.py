"""Single-GPU, finite, unattended development queue with durable status and logs.

Model downloads and source restoration can run concurrently on CPU/network. GPU
jobs are serialized to avoid oversubscribing the 12 GB device. No test evaluation
is implemented or invoked by this queue.
"""

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
from hac.polar_benchmark import (
    atomic_json,
    canonical_hash,
    implementation_evidence,
    lock_json,
    utc_now,
)


def build_plan(root: Path, run: Path, python: Path) -> dict:
    data = run / "data"
    manifests = {
        4: data / "polar4_legacy_development_manifest.csv",
        9: data / "polar9_development_manifest.csv",
    }
    feature_pool = data / "feature_pool_development.csv"
    jobs = []

    def job(name, script, arguments, marker, *, model=None, timeout=21600):
        jobs.append(
            {
                "id": name,
                "command": [
                    str(python),
                    str(root / "experiments" / script),
                    *[str(item) for item in arguments],
                ],
                "completion_marker": str(marker),
                "required_model": model,
                "timeout_seconds": timeout,
            }
        )

    def cache(model, view):
        output = run / "features" / model / view
        job(
            f"cache_{model}_{view}",
            "cache_polar_benchmark_features.py",
            [
                "--manifest",
                feature_pool,
                "--output-dir",
                output,
                "--model-kind",
                model,
                "--view",
                view,
                "--preprocess",
                "historical_polar_eval" if model == "dinov2_base" else "official_processor",
                "--batch-size",
                32,
                "--workers",
                4,
            ],
            output / "provenance.json",
            model=model,
        )

    def screen(classes, model):
        output = run / f"polar{classes}" / "heads" / model
        job(
            f"screen_polar{classes}_{model}",
            "screen_polar_benchmark.py",
            [
                "--manifest",
                manifests[classes],
                "--cache-root",
                run / "features",
                "--model-kind",
                model,
                "--output-dir",
                output,
                "--classes",
                classes,
                "--include-rbf",
            ],
            output / "summary.json",
            model=model,
        )

    def train(recipe, seed, *, smoke=False):
        output = (
            run
            / "polar9"
            / "adaptation"
            / ("engineering_smoke" if smoke else f"{recipe}_seed{seed}")
        )
        arguments = [
            "--manifest",
            manifests[9],
            "--output-dir",
            output,
            "--classes",
            9,
            "--recipe",
            recipe,
            "--seed",
            seed,
            "--workers",
            4,
        ]
        if smoke:
            arguments.append("--smoke")
        job(
            "nine_class_cuda_smoke" if smoke else f"train_polar9_{recipe}_seed{seed}",
            "train_polar_benchmark.py",
            arguments,
            output / "summary.json",
            model="dinov2_base",
            timeout=3600 if smoke else 43200,
        )

    train("historical_mild", 42, smoke=True)
    for view in ("full_frame", "person_context_10"):
        cache("dinov2_base", view)
    screen(9, "dinov2_base")
    screen(4, "dinov2_base")
    # Both fixed view recipes are compared before the two independent confirmation seeds.
    train("historical_mild", 42)
    train("person_preserving", 42)
    confirmation = run / "polar9" / "adaptation" / "confirmation"
    job(
        "confirm_polar9_selected_recipe",
        "confirm_polar_benchmark_recipe.py",
        [
            "--manifest",
            manifests[9],
            "--adaptation-root",
            run / "polar9" / "adaptation",
            "--output-dir",
            confirmation,
            "--classes",
            9,
        ],
        confirmation / "summary.json",
        model="dinov2_base",
        timeout=86400,
    )
    for model in ("siglip2_base", "convnextv2_base", "dinov3_base", "dinov3_large"):
        for view in ("full_frame", "person_context_10"):
            cache(model, view)
        screen(9, model)
        screen(4, model)
    summary = run / "development_summary.json"
    job(
        "summarize_development",
        "summarize_polar_benchmark.py",
        ["--run-dir", run],
        summary,
        timeout=1800,
    )
    return {
        "schema_version": 1,
        "phase": "development_only_before_final_selection_lock",
        "run_directory": str(run),
        "working_directory": str(root),
        "python": str(python),
        "data_audit": str(data / "polar9_data_audit.json"),
        "checkpoint_preparation": str(run / "checkpoint_preparation.json"),
        "jobs": jobs,
        "implementation_sha256": implementation_evidence(root),
        "data_wait_seconds": 43200,
        "model_wait_seconds": 43200,
        "test_access": "FORBIDDEN",
        "historical_results": "READ_ONLY",
        "failure_policy": "stop_on_error; checkpoint_unavailable_recorded_not_substituted",
        "stop_file": str(run / "STOP_AFTER_CURRENT_JOB"),
    }


def verify_implementation(plan):
    root = Path(plan["working_directory"])
    for relative, digest in plan["implementation_sha256"].items():
        if sha256_file(root / relative) != digest:
            raise RuntimeError(f"Implementation changed after queue lock: {relative}")


def verify_data_audit(plan):
    path = Path(plan["data_audit"])
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("status") != "LOCKED_BEFORE_NEW_BENCHMARK_FITTING" or not audit.get("artifacts"):
        raise RuntimeError("Data audit is not complete and locked")
    for relative, item in audit["artifacts"].items():
        artifact = (path.parent / relative).resolve()
        if (
            not artifact.is_relative_to(path.parent.resolve())
            or sha256_file(artifact) != item["sha256"]
        ):
            raise RuntimeError(f"Data artifact differs from audit: {relative}")
    return audit


def wait_for(path, *, timeout, status_path, phase, stop_file):
    started = time.monotonic()
    while not path.is_file():
        if stop_file.exists():
            return False
        if time.monotonic() - started > timeout:
            raise TimeoutError(f"Preparation deadline exceeded: {path.name}")
        atomic_json(
            status_path,
            {
                "status": phase,
                "required_artifact": str(path),
                "elapsed_seconds": time.monotonic() - started,
                "utc": utc_now(),
                "supervisor_pid": os.getpid(),
            },
        )
        time.sleep(20)
    return True


def model_status(plan, model, status_path, stop_file):
    if model is None:
        return True
    path = Path(plan["checkpoint_preparation"])
    started = time.monotonic()
    while True:
        if stop_file.exists():
            return False
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            matches = [item for item in report["models"] if item["model_kind"] == model]
            if matches:
                return matches[0]["status"] == "PINNED_CHECKPOINT_READY"
            if report["status"] == "CHECKPOINT_PREPARATION_COMPLETE":
                return False
        if time.monotonic() - started > plan["model_wait_seconds"]:
            raise TimeoutError(f"Checkpoint preparation deadline exceeded: {model}")
        atomic_json(
            status_path,
            {
                "status": "WAITING_FOR_PINNED_CHECKPOINT",
                "model": model,
                "elapsed_seconds": time.monotonic() - started,
                "utc": utc_now(),
                "supervisor_pid": os.getpid(),
            },
        )
        time.sleep(20)


def stop_owned_process_tree(process):
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"], check=False, capture_output=True
        )
    else:
        process.kill()
    process.wait(timeout=30)


def execute_plan(plan_path: Path):
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    run, root = Path(plan["run_directory"]), Path(plan["working_directory"])
    status_path = run / "queue_status.json"
    stop_file = Path(plan["stop_file"])
    lock_path = run / "queue_process.lock"
    run.mkdir(parents=True, exist_ok=True)
    # Intentionally fails closed after a crashed supervisor; no blind stale-lock deletion.
    with lock_path.open("x", encoding="utf-8") as handle:
        json.dump(
            {"pid": os.getpid(), "plan_sha256": sha256_file(plan_path), "started_utc": utc_now()},
            handle,
        )
    active = None
    try:
        verify_implementation(plan)
        if not wait_for(
            Path(plan["data_audit"]),
            timeout=plan["data_wait_seconds"],
            status_path=status_path,
            phase="WAITING_FOR_AUDITED_DATA",
            stop_file=stop_file,
        ):
            atomic_json(status_path, {"status": "STOP_REQUESTED", "utc": utc_now()})
            return
        audit = verify_data_audit(plan)
        if audit["legacy_four_class_additional_nine_class_quarantine_flags"] != 0:
            raise RuntimeError(
                "New source-overlap findings require a separately locked four-class sensitivity plan"
            )
        receipts, unavailable = [], set()
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(root / "src"),
                "PYTHONUNBUFFERED": "1",
                "OMP_NUM_THREADS": "4",
                "MKL_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4",
                "HF_HUB_DISABLE_XET": "1",
            }
        )
        logs = run / "logs"
        logs.mkdir(exist_ok=True)
        for index, job in enumerate(plan["jobs"]):
            if stop_file.exists():
                atomic_json(
                    status_path,
                    {"status": "STOP_REQUESTED", "completed_jobs": receipts, "utc": utc_now()},
                )
                return
            verify_implementation(plan)
            verify_data_audit(plan)
            model = job["required_model"]
            if model in unavailable or not model_status(plan, model, status_path, stop_file):
                if stop_file.exists():
                    atomic_json(
                        status_path,
                        {"status": "STOP_REQUESTED", "completed_jobs": receipts, "utc": utc_now()},
                    )
                    return
                unavailable.add(model)
                receipt = {
                    "job": job["id"],
                    "status": "NOT_RUN_CHECKPOINT_UNAVAILABLE",
                    "substitute_used": False,
                }
                receipts.append(receipt)
                atomic_json(
                    run / "queue_receipts.json",
                    {"jobs": receipts, "unavailable_models": sorted(unavailable)},
                )
                continue
            job_log = logs / f"{job['id']}.log"
            before = time.monotonic()
            with job_log.open("a", encoding="utf-8") as handle:
                handle.write(f"\n=== Supervisor launch {utc_now()} ===\n")
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
                        status_path,
                        {
                            "status": "RUNNING",
                            "job": job["id"],
                            "job_index": index + 1,
                            "total_jobs": len(plan["jobs"]),
                            "command": job["command"],
                            "child_pid": active.pid,
                            "supervisor_pid": os.getpid(),
                            "elapsed_seconds": elapsed,
                            "log": str(job_log),
                            "utc": utc_now(),
                        },
                    )
                    if elapsed > job["timeout_seconds"]:
                        stop_owned_process_tree(active)
                        raise TimeoutError(f"Bounded job exceeded time budget: {job['id']}")
                    time.sleep(10)
                if active.returncode != 0:
                    raise RuntimeError(
                        f"Job failed ({active.returncode}): {job['id']}; inspect {job_log}"
                    )
            marker = Path(job["completion_marker"])
            if not marker.is_file():
                raise RuntimeError(f"Successful process omitted completion evidence: {job['id']}")
            receipts.append(
                {
                    "job": job["id"],
                    "status": "COMPLETE",
                    "seconds": time.monotonic() - before,
                    "completion_marker": str(marker),
                    "marker_sha256": sha256_file(marker),
                }
            )
            atomic_json(
                run / "queue_receipts.json",
                {"jobs": receipts, "unavailable_models": sorted(unavailable)},
            )
            active = None
        atomic_json(
            status_path,
            {
                "status": "DEVELOPMENT_COMPLETE_REVIEW_REQUIRED",
                "completed_jobs": receipts,
                "unavailable_models": sorted(unavailable),
                "test_evaluated": False,
                "utc": utc_now(),
            },
        )
    except BaseException as error:
        if active is not None and active.poll() is None:
            stop_owned_process_tree(active)
        atomic_json(
            status_path,
            {
                "status": "FAILED_STOPPED",
                "error_type": type(error).__name__,
                "message": str(error),
                "utc": utc_now(),
                "test_evaluated": False,
            },
        )
        raise
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--write-plan", action="store_true")
    parser.add_argument("--plan-name", default="queue_plan.json")
    parser.add_argument(
        "--supersedes", help="Existing plan filename; preserves its original evidence"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    for name in (args.plan_name, args.supersedes):
        if name is not None and (Path(name).name != name or not name.endswith(".json")):
            parser.error("Plan names must be JSON filenames inside the run directory")
    plan_path = run / args.plan_name
    if args.write_plan:
        plan = build_plan(root, run, Path(sys.executable).resolve())
        if args.supersedes:
            previous = run / args.supersedes
            if previous == plan_path or not previous.is_file():
                parser.error("--supersedes must identify a different existing plan")
            plan["supersedes"] = {"filename": args.supersedes, "sha256": sha256_file(previous)}
        lock_json(plan_path, plan)
        print(
            json.dumps(
                {
                    "status": "QUEUE_LOCKED",
                    "jobs": len(plan["jobs"]),
                    "plan_sha256": canonical_hash(plan),
                    "path": str(plan_path),
                },
                indent=2,
            )
        )
    else:
        execute_plan(plan_path)


if __name__ == "__main__":
    main()
