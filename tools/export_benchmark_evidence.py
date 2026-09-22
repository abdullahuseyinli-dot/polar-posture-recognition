"""Export portable numerical evidence from the completed, hash-bound POLAR run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = Path("docs/research/20260921_final_evaluation/results")
SUMMARY_SHA256 = "5beab5852a8aea3375a2b7425308f4c89974acc2f135c1411d6994a4e52b813a"
LOCK_SHA256 = "b8334696ee21ab850c9b8d364d89ace59f8065e6551dae62bf9defe5185a5bbe"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_path(root: Path, path: str, expected: str) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_relative_to(root.resolve()) or digest(resolved) != expected:
        raise ValueError(f"Source outside the run or hash mismatch: {resolved.name}")
    return resolved


def export(run: Path, data: Path, output: Path) -> dict:
    summary = read_json(ROOT / REPORT / "summary.json")
    if digest(ROOT / REPORT / "summary.json") != SUMMARY_SHA256:
        raise ValueError("Final report bytes differ from the verified result")
    if digest(run / "final_selection_lock.json") != LOCK_SHA256:
        raise ValueError("Final selection lock differs")
    completion = read_json(run / "completion.json")
    if (
        completion["status"] != "LOCKED_FINAL_EVALUATION_COMPLETE"
        or completion["comparison_summary_sha256"] != SUMMARY_SHA256
    ):
        raise ValueError("Run is not the completed locked evaluation")
    source_manifest = run / "evaluation/prediction_manifest.json"
    if digest(source_manifest) != summary["prediction_manifest_sha256"]:
        raise ValueError("Prediction manifest differs from the final report")
    manifest = read_json(source_manifest)
    # A versioned export is never replaced in place.
    output.mkdir(parents=True, exist_ok=False)
    tasks = {}
    for task, record in manifest["tasks"].items():
        group_path = checked_path(run, record["source_groups_path"], record["source_groups_sha256"])
        arrays = {
            "source_groups": np.asarray(read_json(group_path), dtype=str),
            "class_names": np.asarray(record["class_names"], dtype=str),
        }
        source_hashes = {}
        for name, candidate in record["candidates"].items():
            path = checked_path(run, candidate["path"], candidate["sha256"])
            with np.load(path, allow_pickle=False) as saved:
                for field in ("image_ids", "labels"):
                    values = saved[field]
                    if field in arrays and not np.array_equal(arrays[field], values):
                        raise ValueError("Candidate row/label alignment differs")
                    arrays[field] = values
                arrays[name] = saved["probabilities"]
            source_hashes[name] = {"sha256": candidate["sha256"], "role": candidate["kind"]}
        if (
            len(arrays["labels"]) != record["rows"]
            or len(arrays["source_groups"]) != record["rows"]
        ):
            raise ValueError("Source groups or evaluation population differ")
        np.savez_compressed(output / f"{task}_predictions.npz", **arrays)
        tasks[task] = {
            "rows": record["rows"],
            "class_names": record["class_names"],
            "candidates": source_hashes,
            "source_groups_sha256": record["source_groups_sha256"],
        }

    audit_path = data / "polar9_data_audit.json"
    audit = read_json(audit_path)
    # Numerical audit has no local image files or credentials; omit execution timing.
    exported_audit = {k: v for k, v in audit.items() if k != "runtime_seconds"}
    (output / "data_audit.json").write_text(
        json.dumps(exported_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    columns = [
        "image_id",
        "split",
        "label",
        "label_index",
        "bbox_xmin",
        "bbox_ymin",
        "bbox_xmax",
        "bbox_ymax",
        "image_sha256",
        "annotation_sha256",
        "source_group",
    ]
    for source, target in (
        ("polar9_clean_manifest.csv", "cohort.csv"),
        ("polar9_quarantine.csv", "quarantine.csv"),
    ):
        path = data / source
        if digest(path) != audit["artifacts"][source]["sha256"]:
            raise ValueError("Data audit does not bind the exported membership")
        with path.open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        with (output / target).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=columns, lineterminator="\n", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerows(records)
    artifacts = {
        p.name: {"sha256": digest(p), "bytes": p.stat().st_size}
        for p in sorted(output.iterdir())
        if p.is_file()
    }
    payload = {
        "schema_version": 1,
        "project_version": "1.1.0",
        "scope": "Unchanged prediction values and audited membership; no images, weights or local paths",
        "summary_path": (REPORT / "summary.json").as_posix(),
        "summary_sha256": SUMMARY_SHA256,
        "selection_lock_sha256": LOCK_SHA256,
        "source_prediction_manifest_sha256": digest(source_manifest),
        "source_data_audit_sha256": digest(audit_path),
        "tasks": tasks,
        "artifacts": artifacts,
    }
    (output / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "tasks": len(tasks),
        "prediction_sets": sum(len(t["candidates"]) for t in tasks.values()),
        "bytes": sum(a["bytes"] for a in artifacts.values()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "results/polar_20260921")
    args = parser.parse_args()
    print(
        json.dumps(export(args.run_dir.resolve(), args.data_dir.resolve(), args.output.resolve()))
    )
