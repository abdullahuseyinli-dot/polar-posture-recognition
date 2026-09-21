"""Compare a fixed candidate family, retaining the incumbent unless all gates pass."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file
from hac.polar_benchmark import (
    atomic_json,
    canonical_hash,
    check_completed,
    label_names,
    lock_json,
    probability_metrics,
    utc_now,
)
from hac.polar_bounded import (
    INCUMBENT,
    MODEL_KINDS,
    SEEDS,
    TASKS,
    candidate_family,
    development_rows,
    load_probabilities,
    manifest_path,
    paired_bootstrap,
    promotion_gate,
    read_json,
    transitions,
    verify_bindings,
)


def summarize_task(run: Path, parent: Path, classes: int, protocol: dict) -> dict:
    task = f"polar{classes}"
    manifest = manifest_path(parent, classes)
    frame = development_rows(manifest, classes)
    validation = frame.loc[frame.split.eq("val")]
    labels, names = validation.label_index.to_numpy(), label_names(frame)
    parent_analysis = parent / task / "development_analysis"
    incumbent = load_probabilities(
        parent_analysis / f"{INCUMBENT}_validation.npz", validation, classes
    )
    anchor_name = "dinov2_adapted_3seeds" if classes == 9 else "dinov2_base"
    anchor = load_probabilities(
        parent_analysis / f"{anchor_name}_validation.npz", validation, classes
    )
    evidence, seed_values, model_statistics = {}, {}, {}
    for model in MODEL_KINDS:
        seed_values[model] = []
        scores = []
        for seed in SEEDS:
            directory = run / task / model / f"seed{seed}"
            request = read_json(directory / "request.json")
            if (
                request.get("manifest_sha256") != sha256_file(manifest)
                or request.get("seed") != seed
                or request.get("model_kind") != model
                or request.get("classes") != classes
                or request.get("role") != "development_adaptation"
                or request.get("initialization") != "original_foundation_only"
                or request.get("test_rows_read") != 0
            ):
                raise RuntimeError("New seed run does not match the bounded cohort/model contract")
            summary = check_completed(directory, request)
            if summary is None:
                raise RuntimeError("Every declared seed must finish; no favorable-seed selection")
            replay = read_json(directory / "replay_audit.json")
            if (
                replay.get("status") != "PASS"
                or not replay.get("labels_identical")
                or not replay.get("predictions_identical")
                or replay.get("maximum_probability_difference", float("inf")) > 1e-5
                or replay.get("checkpoint_sha256") != sha256_file(directory / "best.pt")
            ):
                raise RuntimeError("Completed seed has not passed checkpoint replay")
            values = load_probabilities(
                directory / "validation_predictions.npz", validation, classes
            )
            score = probability_metrics(labels, values, names)["macro_f1"]
            if abs(score - summary["best_validation_macro_f1"]) > 1e-12:
                raise RuntimeError("Seed summary disagrees with saved predictions")
            seed_values[model].append(values)
            scores.append(score)
            evidence[f"{model}_seed{seed}"] = {
                "summary_sha256": sha256_file(directory / "summary.json"),
                "predictions_sha256": sha256_file(directory / "validation_predictions.npz"),
                "best_epoch": summary["best_epoch"],
                "epochs_completed": summary["epochs_completed"],
                "trainable_parameters": request["parameters"]["trainable_parameters"],
                "frozen_parameters": request["parameters"]["frozen_parameters"],
            }
        model_statistics[model] = {
            "seeds": list(SEEDS),
            "macro_f1": scores,
            "mean": float(np.mean(scores)),
            "sample_sd": float(np.std(scores, ddof=1)),
        }
    candidates = candidate_family(
        incumbent,
        anchor,
        np.mean(seed_values["siglip2_base"], axis=0),
        np.mean(seed_values["convnextv2_base"], axis=0),
    )
    single_seed_candidates = [
        candidate_family(
            incumbent,
            anchor,
            seed_values["siglip2_base"][index],
            seed_values["convnextv2_base"][index],
        )
        for index in range(3)
    ]
    reference_metrics = probability_metrics(labels, incumbent, names)
    results = []
    directory = run / task / "development_analysis"
    request = {
        "manifest_sha256": sha256_file(manifest),
        "protocol_sha256": canonical_hash(protocol),
        "inputs": evidence,
        "prior_incumbent_predictions_sha256": sha256_file(
            parent_analysis / f"{INCUMBENT}_validation.npz"
        ),
        "anchor_predictions_sha256": sha256_file(parent_analysis / f"{anchor_name}_validation.npz"),
        "test_rows_read": 0,
    }
    if previous := check_completed(directory, request):
        return previous
    lock_json(directory / "request.json", request)
    for name, values in candidates.items():
        values = values / values.sum(axis=1, keepdims=True)
        metrics = probability_metrics(labels, values, names)
        transition = transitions(labels, incumbent, values)
        seed_comparisons = []
        for seed, counterparts in zip(SEEDS, single_seed_candidates, strict=True):
            counterpart = counterparts[name]
            seed_comparisons.append(
                {
                    "seed": seed,
                    "macro_f1_gain": probability_metrics(labels, counterpart, names)["macro_f1"]
                    - reference_metrics["macro_f1"],
                    **transitions(labels, incumbent, counterpart),
                }
            )
        gate = promotion_gate(
            metrics,
            reference_metrics,
            transition,
            seed_comparisons,
            protocol["development_promotion_gate"],
        )
        if name == "retain_incumbent":
            gate = {"eligible": True, "role": "explicit_no_change_action"}
        uncertainty = (
            paired_bootstrap(labels, incumbent, values, validation.source_group.to_numpy())
            if name != "retain_incumbent"
            else None
        )
        np.savez_compressed(
            directory / f"{name}_validation.npz",
            image_ids=validation.image_id.to_numpy(dtype=str),
            labels=labels,
            probabilities=values,
        )
        results.append(
            {
                "candidate": name,
                **metrics,
                "macro_f1_gain": metrics["macro_f1"] - reference_metrics["macro_f1"],
                "transitions": transition,
                "seed_sensitivity": seed_comparisons,
                "promotion_gate": gate,
                "descriptive_uncertainty": uncertainty,
            }
        )
    results.sort(key=lambda item: (-item["macro_f1"], item["log_loss"], item["candidate"]))
    eligible = [row for row in results if row["promotion_gate"]["eligible"]]
    selected = eligible[0]["candidate"]
    pd.DataFrame(
        [
            {key: value for key, value in row.items() if not isinstance(value, (dict, list))}
            for row in results
        ]
    ).to_csv(directory / "ranking.csv", index=False)
    artifacts = {
        path.name: sha256_file(path)
        for path in directory.iterdir()
        if path.suffix in {".npz", ".csv"}
    }
    artifacts["request.json"] = sha256_file(directory / "request.json")
    report = {
        "status": "COMPLETE",
        "request_sha256": canonical_hash(request),
        "candidates": results,
        "seed_statistics": model_statistics,
        "selected_development_candidate": selected,
        "incumbent_retained": selected == "retain_incumbent",
        "anchor": anchor_name,
        "role": "selected_validation_only_not_confirmatory",
        "test_rows_read": 0,
        "artifacts": artifacts,
        "completed_utc": utc_now(),
    }
    atomic_json(directory / "summary.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    plan = read_json(run / "queue_plan.json")
    if plan.get("test_access") != "FORBIDDEN" or plan.get("production_fits") != 12:
        raise RuntimeError("Unexpected development plan")
    parent = Path(plan["parent_directory"])
    verify_bindings(parent, read_json(Path(plan["parent_evidence_lock"])))
    protocol_path = Path(plan["protocol"])
    if sha256_file(protocol_path) != plan["protocol_sha256"]:
        raise RuntimeError("Bounded protocol changed")
    protocol = read_json(protocol_path)
    tasks = {f"polar{classes}": summarize_task(run, parent, classes, protocol) for classes in TASKS}
    report = {
        "status": "BOUNDED_DEVELOPMENT_COMPLETE_EVALUATION_LOCK_REQUIRED",
        "tasks": tasks,
        "all_12_production_fits_verified": True,
        "test_evaluated": False,
        "historical_result_overwritten": False,
        "next_step": "review_finalists_and_lock_final_refits_and_comparisons_before_test_evaluation; no_further_automatic_search",
        "completed_utc": utc_now(),
    }
    atomic_json(run / "bounded_summary.json", report)
    print(
        {
            "status": report["status"],
            "selected": {
                name: value["selected_development_candidate"] for name, value in tasks.items()
            },
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
