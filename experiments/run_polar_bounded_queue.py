"""Lock and execute one finite modern-backbone phase, never final test evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from run_polar_benchmark_queue import execute_plan

from hac.polar import sha256_file
from hac.polar_benchmark import canonical_hash, lock_json
from hac.polar_bounded import (
    MODEL_KINDS,
    SEEDS,
    TASKS,
    audit_parent,
    implementation_hashes,
    manifest_path,
    read_json,
    verify_bindings,
)


def build_jobs(root: Path, run: Path, parent: Path, python: Path) -> list[dict]:
    jobs = []

    def train(classes, model, seed, smoke=False):
        name = f"{'smoke' if smoke else 'train'}_polar{classes}_{model}_seed{seed}"
        directory = run / ("smokes" if smoke else f"polar{classes}") / model / f"seed{seed}"
        if smoke:
            directory = directory / f"classes{classes}"
        command = [
            str(python),
            str(root / "experiments" / "train_polar_bounded.py"),
            "--manifest",
            str(manifest_path(parent, classes)),
            "--output-dir",
            str(directory),
            "--classes",
            str(classes),
            "--model-kind",
            model,
            "--seed",
            str(seed),
            "--workers",
            "4",
        ]
        if smoke:
            command.append("--smoke")
        jobs.append(
            {
                "id": name,
                "command": command,
                "completion_marker": str(directory / "summary.json"),
                # Both models are mandatory here, unlike the optional old DINOv3-L.
                # The trainer fails on missing/mismatched original weights; never skip a fit.
                "required_model": None,
                "timeout_seconds": 900 if smoke else 10800,
            }
        )

    for classes in TASKS:
        for model in MODEL_KINDS:
            train(classes, model, 42, smoke=True)
    for classes in TASKS:
        for model in MODEL_KINDS:
            for seed in SEEDS:
                train(classes, model, seed)
    jobs.append(
        {
            "id": "summarize_bounded_development",
            "command": [
                str(python),
                str(root / "experiments" / "summarize_polar_bounded.py"),
                "--run-dir",
                str(run),
            ],
            "completion_marker": str(run / "bounded_summary.json"),
            "required_model": None,
            "timeout_seconds": 1800,
        }
    )
    return jobs


def build_plan(root: Path, run: Path, parent: Path, python: Path, parent_evidence: dict) -> dict:
    root, run, parent = root.resolve(), run.resolve(), parent.resolve()
    if run == parent or run.is_relative_to(parent) or parent.is_relative_to(run):
        raise ValueError("New evidence directory must be separate from its parent run")
    if not run.is_relative_to(root / ".runs") or not parent.is_relative_to(root / ".runs"):
        raise ValueError("Both run directories must be within this repository's .runs")
    protocol = root / "experiments" / "polar_bounded_protocol_20260921.json"
    instructions = read_json(protocol)
    if instructions["test_access"] != "FORBIDDEN" or instructions["maximum_production_fits"] != 12:
        raise RuntimeError("Unexpected bounded development budget")
    source_hashes = implementation_hashes(root)
    # Reuse the tested supervisor, whose per-job hash check also binds immutable
    # parent data/probabilities/protocol. No source files from phase one are edited.
    for relative, digest in parent_evidence["artifacts"].items():
        source_hashes[(parent / relative).relative_to(root).as_posix()] = digest
    evidence_path = run / "parent_evidence_lock.json"
    source_hashes[evidence_path.relative_to(root).as_posix()] = sha256_file(evidence_path)
    return {
        "schema_version": 1,
        "phase": "one_bounded_adaptation_phase_no_test_access",
        "run_directory": str(run),
        "parent_directory": str(parent),
        "working_directory": str(root),
        "python": str(python),
        "protocol": str(protocol),
        "protocol_sha256": sha256_file(protocol),
        "parent_evidence_lock": str(evidence_path),
        "data_audit": str(parent / "data" / "polar9_data_audit.json"),
        "checkpoint_preparation": str(parent / "checkpoint_preparation.json"),
        "jobs": build_jobs(root, run, parent, python),
        "implementation_sha256": source_hashes,
        "production_fits": 12,
        "engineering_smokes": 4,
        "data_wait_seconds": 60,
        "model_wait_seconds": 60,
        "test_access": "FORBIDDEN",
        "historical_results": "READ_ONLY",
        "failure_policy": "stop_on_error_no_skip_substitution_or_performance_retry",
        "stop_file": str(run / "STOP_AFTER_CURRENT_JOB"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--write-plan", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = args.run_dir.resolve()
    plan_path = run / "queue_plan.json"
    if args.write_plan:
        if args.parent_run is None:
            parser.error("--parent-run is required to lock a new queue")
        parent = args.parent_run.resolve()
        if run == parent or run.is_relative_to(parent) or parent.is_relative_to(run):
            parser.error("A distinct, non-nested output run directory is required")
        if not run.is_relative_to(root / ".runs") or not parent.is_relative_to(root / ".runs"):
            parser.error("Both run directories must be within this repository's .runs")
        evidence = audit_parent(parent)
        lock_json(run / "parent_evidence_lock.json", evidence)
        plan = build_plan(root, run, parent, Path(sys.executable).resolve(), evidence)
        lock_json(plan_path, plan)
        print(
            {
                "status": "BOUNDED_PLAN_LOCKED",
                "jobs": len(plan["jobs"]),
                "production_fits": 12,
                "plan_sha256": canonical_hash(plan),
                "path": str(plan_path),
            },
            flush=True,
        )
    else:
        plan = read_json(plan_path)
        if Path(plan["run_directory"]).resolve() != run or plan.get("test_access") != "FORBIDDEN":
            raise RuntimeError("Locked queue is not the requested development run")
        if sha256_file(Path(plan["protocol"])) != plan["protocol_sha256"]:
            raise RuntimeError("Protocol changed after lock")
        parent = Path(plan["parent_directory"])
        verify_bindings(parent, read_json(Path(plan["parent_evidence_lock"])))
        execute_plan(plan_path)


if __name__ == "__main__":
    main()
