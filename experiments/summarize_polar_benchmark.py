"""Collect development comparisons and bounded fusion; never open a test manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file
from hac.polar_benchmark import (
    atomic_json,
    canonical_hash,
    check_completed,
    label_names,
    load_development,
    lock_json,
    probability_metrics,
    utc_now,
)


def collect_task(run, *, task_name, classes, manifest):
    frame = load_development(manifest, num_classes=classes)
    validation = frame.loc[frame.split.eq("val")]
    labels = validation.label_index.to_numpy()
    identifiers = validation.image_id.to_numpy(dtype=str)
    names = label_names(frame)
    candidates, probabilities, source_hashes = [], {}, {}
    for path in sorted((run / task_name / "heads").glob("*/summary.json")):
        directory = path.parent
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        if request["manifest_sha256"] != sha256_file(manifest):
            raise RuntimeError("Head belongs to a different cohort")
        summary = check_completed(directory, request)
        if summary is None:
            raise RuntimeError("Missing completed head")
        name = directory.name
        child = directory / summary["best_candidate"]
        child_request = json.loads((child / "request.json").read_text(encoding="utf-8"))
        if check_completed(child, child_request) is None:
            raise RuntimeError("Selected candidate has not completed its integrity checks")
        prediction_path = child / "validation_predictions.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            if not np.array_equal(archive["image_ids"], identifiers) or not np.array_equal(
                archive["labels"], labels
            ):
                raise RuntimeError("Model predictions do not share evaluation rows")
            probabilities[name] = archive["probabilities"].copy()
        source_hashes[name] = {
            "summary": sha256_file(path),
            "predictions": sha256_file(prediction_path),
        }
    confirmation = run / task_name / "adaptation" / "confirmation"
    if (confirmation / "summary.json").exists():
        selection = json.loads((confirmation / "selection_lock.json").read_text(encoding="utf-8"))
        check_completed(confirmation, selection)
        with np.load(confirmation / "validation_predictions.npz", allow_pickle=False) as archive:
            if not np.array_equal(archive["image_ids"], identifiers) or not np.array_equal(
                archive["labels"], labels
            ):
                raise RuntimeError("Adapted seed ensemble row mismatch")
            probabilities["dinov2_adapted_3seeds"] = archive["probabilities"].copy()
        source_hashes["dinov2_adapted_3seeds"] = {
            "summary": sha256_file(confirmation / "summary.json"),
            "predictions": sha256_file(confirmation / "validation_predictions.npz"),
        }
    if not probabilities:
        return {"status": "NO_ADMISSIBLE_COMPLETED_MODELS", "test_evaluated": False}
    for name, values in probabilities.items():
        metrics = probability_metrics(labels, values, names)
        candidates.append({"candidate": name, "kind": "standalone_or_seed_average", **metrics})
    ranked = sorted(
        candidates, key=lambda row: (-row["macro_f1"], row["log_loss"], row["candidate"])
    )
    # Only these two uniform fusions are declared. No arbitrary learned test-time routing.
    fusion_members = {"uniform_all": sorted(probabilities)}
    if len(probabilities) >= 2:
        fusion_members["uniform_top_two_validation"] = [row["candidate"] for row in ranked[:2]]
    for name, members in fusion_members.items():
        if len(members) < 2:
            continue
        values = np.mean([probabilities[member] for member in members], axis=0)
        values /= values.sum(axis=1, keepdims=True)
        probabilities[name] = values
        candidates.append(
            {
                "candidate": name,
                "kind": "validation_selected_uniform_fusion",
                **probability_metrics(labels, values, names),
            }
        )
    destination = run / task_name / "development_analysis"
    destination.mkdir(parents=True, exist_ok=True)
    request = {
        "manifest_sha256": sha256_file(manifest),
        "inputs": source_hashes,
        "fusion_members": fusion_members,
        "test_rows_read": 0,
    }
    lock_json(destination / "analysis_request.json", request)
    transitions = []
    reference_name = "dinov2_base" if "dinov2_base" in probabilities else ranked[0]["candidate"]
    reference_errors = probabilities[reference_name].argmax(axis=1) != labels
    for name, values in probabilities.items():
        errors = values.argmax(axis=1) != labels
        rescued = int((reference_errors & ~errors).sum())
        harmed = int((~reference_errors & errors).sum())
        union = int((reference_errors | errors).sum())
        transitions.append(
            {
                "reference": reference_name,
                "candidate": name,
                "rescued": rescued,
                "harmed": harmed,
                "net_corrections": rescued - harmed,
                "error_jaccard": float((reference_errors & errors).sum() / union) if union else 1.0,
            }
        )
        np.savez_compressed(
            destination / f"{name}_validation.npz",
            image_ids=identifiers,
            labels=labels,
            probabilities=values,
        )
    candidates.sort(key=lambda row: (-row["macro_f1"], row["log_loss"], row["candidate"]))
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in candidates
        ]
    ).to_csv(destination / "development_ranking.csv", index=False)
    pd.DataFrame(transitions).to_csv(destination / "error_transitions.csv", index=False)
    report = {
        "status": "DEVELOPMENT_ANALYSIS_COMPLETE",
        "request_sha256": canonical_hash(request),
        "candidates": candidates,
        "reference": reference_name,
        "error_transitions": transitions,
        "fusion_members": fusion_members,
        "evidence_role": "selected_on_validation_not_confirmatory; no_SOTA_claim",
        "test_rows_read": 0,
        "next_gate": "audit_replays_and_bounded_adapted_challengers_before_final_selection_lock",
    }
    atomic_json(destination / "analysis.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    audit = json.loads((run / "data" / "polar9_data_audit.json").read_text(encoding="utf-8"))
    tasks = {"polar9": (9, "polar9_development_manifest.csv")}
    if audit["legacy_four_class_additional_nine_class_quarantine_flags"]:
        tasks["polar4_source_audited"] = (4, "polar4_source_audited_development_manifest.csv")
    else:
        tasks["polar4"] = (4, "polar4_legacy_development_manifest.csv")
    reports = {
        task: collect_task(run, task_name=task, classes=classes, manifest=run / "data" / manifest)
        for task, (classes, manifest) in tasks.items()
    }
    atomic_json(
        run / "development_summary.json",
        {
            "status": "DEVELOPMENT_COMPLETE_REVIEW_REQUIRED",
            "tasks": reports,
            "test_evaluated": False,
            "historical_result_overwritten": False,
            "completed_utc": utc_now(),
        },
    )


if __name__ == "__main__":
    main()
