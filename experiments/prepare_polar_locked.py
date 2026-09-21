"""Freeze development-selected final refits/comparisons before new test inference."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import zipfile
from pathlib import Path

from hac.polar import sha256_file
from hac.polar_benchmark import check_completed, label_names, lock_json, utc_now
from hac.polar_bounded import audit_parent, development_rows
from hac.polar_locked_evaluation import LOCK_STATUS, MODELS, SEEDS, read_json

HISTORICAL_MEMBER = "modern_checkout/.runs/polar_final/test_evaluation/test_predictions.npz"
HISTORICAL_SHA256 = "0f96ecb7411abf8b3385004380a5fa001965113f0855015db4337fb76078c5f9"


def recover_historical(transfer: Path, run: Path) -> dict:
    record = None
    with (transfer / "FILE_MANIFEST.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("member") == HISTORICAL_MEMBER:
                if record is not None:
                    raise ValueError("Duplicate historical archive record")
                record = item
    if record is None or record["sha256"] != HISTORICAL_SHA256:
        raise RuntimeError("Historical predictions are not the registered original evidence")
    archive = (transfer / record["archive"]).resolve(strict=True)
    if not archive.is_relative_to(transfer.resolve()):
        raise ValueError("Archive escapes transfer root")
    target = run / "inputs" / "historical_four_test_predictions.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        with zipfile.ZipFile(archive) as bundle:
            data = bundle.read(HISTORICAL_MEMBER)
        import hashlib

        if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != HISTORICAL_SHA256:
            raise RuntimeError("Historical archive member integrity failed")
        with target.open("xb") as handle:
            handle.write(data)
    if sha256_file(target) != HISTORICAL_SHA256:
        raise RuntimeError("Previously recovered evidence changed")
    return {
        "path": str(target),
        "sha256": HISTORICAL_SHA256,
        "archive": str(archive),
        "member": HISTORICAL_MEMBER,
        "manifest_sha256": sha256_file(transfer / "FILE_MANIFEST.jsonl"),
        "expected_macro_f1": 0.9398833285904803,
        "expected_rows": 3329,
        "probability_key": "probabilities_locked_ensemble",
        "label_key": "labels_4",
    }


def prepare(root: Path, run: Path, parent: Path, bounded: Path, transfer: Path) -> dict:
    run.mkdir(parents=True, exist_ok=True)
    lock_path = run / "final_selection_lock.json"
    if lock_path.exists():
        from hac.polar_locked_evaluation import verify_selection_lock

        return verify_selection_lock(lock_path)
    if (run / "test_access_gate.json").exists():
        raise RuntimeError("Cannot create a new selection lock after test access")
    parent_audit = audit_parent(parent)
    lock_json(run / "parent_verification.json", parent_audit)
    summary_path = bounded / "bounded_summary.json"
    bounded_summary = read_json(summary_path)
    if bounded_summary.get(
        "status"
    ) != "BOUNDED_DEVELOPMENT_COMPLETE_EVALUATION_LOCK_REQUIRED" or not bounded_summary.get(
        "all_12_production_fits_verified"
    ):
        raise RuntimeError("Bounded development has not completed")
    evidence = []

    def bind(path: Path):
        evidence.append({"path": str(path.resolve(strict=True)), "sha256": sha256_file(path)})

    bind(summary_path)
    bind(parent / "development_summary.json")
    bind(parent / "checkpoint_preparation.json")
    bind(parent / "data" / "polar9_data_audit.json")
    audit = read_json(parent / "data" / "polar9_data_audit.json")
    tasks = {}
    for classes in (9, 4):
        task = f"polar{classes}"
        selected = bounded_summary["tasks"][task]
        if selected["selected_development_candidate"] != "conservative_fusion":
            raise ValueError("Unexpected development nomination; manual review required")
        candidate = next(
            item for item in selected["candidates"] if item["candidate"] == "conservative_fusion"
        )
        if not candidate["promotion_gate"]["eligible"] or selected["test_rows_read"] != 0:
            raise ValueError("Development nomination did not pass the locked gates")
        prefix = "polar9" if classes == 9 else "polar4_legacy"
        development = parent / "data" / f"{prefix}_development_manifest.csv"
        test_manifest = parent / "data" / f"{prefix}_test_manifest.csv"
        frame = development_rows(development, classes)
        test_binding = audit["artifacts"][test_manifest.name]
        # The test bytes were audited already; no test labels/predictions parsed here.
        if sha256_file(test_manifest) != test_binding["sha256"]:
            raise ValueError("Audited test manifest bytes changed")
        neural = {}
        model_kinds = (
            ("dinov2_base", "siglip2_base", "convnextv2_base")
            if classes == 9
            else ("siglip2_base", "convnextv2_base")
        )
        for model in model_kinds:
            epochs, parents = [], []
            for seed in SEEDS:
                directory = (
                    parent / task / "adaptation" / f"person_preserving_seed{seed}"
                    if model == "dinov2_base"
                    else bounded / task / model / f"seed{seed}"
                )
                request = read_json(directory / "request.json")
                complete = check_completed(directory, request)
                if (
                    complete is None
                    or complete.get("role") != "development_adaptation"
                    or complete.get("test_rows_read") != 0
                ):
                    raise ValueError("Incomplete or test-exposed development fit")
                replay = read_json(directory / "replay_audit.json")
                if replay["status"] != "PASS":
                    raise ValueError("Development replay failed")
                epochs.append(int(complete["best_epoch"]))
                parents.append(str(directory))
                bind(directory / "summary.json")
                bind(directory / "request.json")
                bind(directory / "replay_audit.json")
            neural[model] = {
                "seeds": list(SEEDS),
                "epochs": int(statistics.median(epochs)),
                "development_best_epochs": epochs,
                "epoch_selection": "median_of_three_development_best_epochs",
                "schedule_horizon_epochs": 20,
                "development_parents": parents,
                "output_dir_pattern": f"{task}/neural/{model}/seed{{seed}}",
                "initialization": "fresh_pinned_foundation_not_development_checkpoint",
            }
        heads = {}
        for model in MODELS:
            head_parent = parent / task / "heads" / model
            head_summary = read_json(head_parent / "summary.json")
            if head_summary["best_candidate"] != "two_view_concat__cuda_kernel_rbf__c10":
                raise ValueError("Frozen head does not match the actual development winner")
            bind(head_parent / "summary.json")
            heads[model] = {
                "C": 10.0,
                "gamma": "1/d",
                "class_weight": None,
                "views": ["full_frame", "person_context_10"],
                "calibration_folds": 5,
                "seed": 42,
                "output_dir": f"{task}/heads/{model}",
                "calibration": "sigmoid_StratifiedGroupKFold_ensemble_true",
                "scaler": "inside_each_calibration_training_fold",
            }
        names = ["conservative_fusion", "prior_incumbent", "replacement_fusion"]
        names += [f"frozen_{model}" for model in MODELS]
        names += ["adapted_siglip2", "adapted_convnextv2"]
        names += ["adapted_dinov2" if classes == 9 else "historical_ensemble"]
        tasks[task] = {
            "classes": classes,
            "class_names": label_names(frame),
            "development_manifest": str(development),
            "development_manifest_sha256": sha256_file(development),
            "development_rows": len(frame),
            "test_manifest": str(test_manifest),
            "test_manifest_sha256": test_binding["sha256"],
            "test_rows": test_binding["rows"],
            "neural_fits": neural,
            "head_fits": heads,
            "nominated_candidate": "conservative_fusion",
            "candidates": names,
            "development_nomination_macro_f1": candidate["macro_f1"],
        }
    historical = recover_historical(transfer, run)
    comparisons = []
    for task, spec in tasks.items():
        strongest = "adapted_dinov2" if task == "polar9" else "frozen_siglip2_base"
        for name in spec["candidates"]:
            if name != "conservative_fusion":
                comparisons.append(
                    {
                        "id": f"{task}__vs__{name}",
                        "task": task,
                        "candidate": "conservative_fusion",
                        "reference": name,
                        "primary": name in {"prior_incumbent", strongest, "historical_ensemble"},
                    }
                )
    sources = sorted((root / "src" / "hac").glob("*.py")) + sorted(
        (root / "experiments").glob("*.py")
    )
    tracked = subprocess.check_output(
        ["git", "ls-files", "results", "assets", "docs"], cwd=root, text=True
    ).splitlines()
    protected = {
        relative: sha256_file(root / relative)
        for relative in tracked
        if (root / relative).is_file()
    }
    lock_json(run / "protected_historical_files.json", protected)
    value = {
        "schema_version": 1,
        "status": LOCK_STATUS,
        "created_utc": utc_now(),
        "repository_root": str(root),
        "output_root": str(run),
        "parent_development": str(parent),
        "bounded_development": str(bounded),
        "development_feature_root": str(parent / "features"),
        "implementation": {
            path.relative_to(root).as_posix(): sha256_file(path) for path in sources
        },
        "development_evidence": evidence,
        "tasks": tasks,
        "historical_reference": historical,
        "neural_training": {
            "batch_size": 16,
            "accumulation": 4,
            "head_lr": 0.001,
            "backbone_lr": 5e-6,
            "weight_decay": 1e-4,
            "dropout": 0.1,
            "gradient_clip": 1.0,
            "loss": "cross_entropy",
            "warmup_fraction": 0.1,
            "schedule_horizon_epochs": 20,
            "early_stopping": False,
            "workers": 4,
            "precision": "cuda_bfloat16",
            "view": "person_context_25",
            "augmentation": "person_safe_mild",
        },
        "fusion": {
            "anchor": 0.5,
            "frozen_siglip2": 0.25,
            "adapted_siglip2_three_seed_mean": 0.25,
            "nine_class_anchor": "adapted_dinov2_three_seed_mean",
            "four_class_anchor": "frozen_dinov2_base",
        },
        "statistics": {
            "bootstrap_draws": 5000,
            "randomization_draws": 10000,
            "seed": 20260921,
            "alpha": 0.05,
            "comparisons": comparisons,
            "multiplicity": "one_Holm_family_all_18_comparisons",
            "randomization_unit": "detected_source_group",
            "alternative": "two_sided",
            "gates": {
                "positive_macro_f1": True,
                "positive_net_corrections": True,
                "source_group_delta_ci_lower_gt": 0,
                "holm_p_lt": 0.05,
                "max_per_class_f1_drop": 0.01,
                "max_nll_increase": 0.02,
                "max_brier_increase": 0.01,
                "all_seed_counterparts_positive": True,
            },
        },
        "historical_exposure": {
            "four_class": "previously_inspected_test_same_membership; not_a_fresh_blind_holdout",
            "nine_class": "includes_historically_inspected_four_class_subset; not_wholly_unseen",
            "new_phase_test_prediction_selection": False,
        },
        "limitations": audit["limitations"],
        "test_selection_forbidden": True,
        "automatic_search_after_test": False,
        "resource_policy": {
            "single_gpu_serialized": True,
            "job_timeout_seconds": 43200,
            "stop_file": "STOP_AFTER_CURRENT_JOB",
            "retry_failed_job_automatically": False,
        },
    }
    lock_json(lock_path, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--parent-run", required=True, type=Path)
    parser.add_argument("--bounded-run", required=True, type=Path)
    parser.add_argument("--transfer-root", required=True, type=Path)
    args = parser.parse_args()
    value = prepare(
        Path(__file__).resolve().parents[1],
        args.run_dir.resolve(),
        args.parent_run.resolve(),
        args.bounded_run.resolve(),
        args.transfer_root.resolve(),
    )
    print(
        json.dumps(
            {
                "status": value["status"],
                "run": value["output_root"],
                "neural_fits": 15,
                "head_fits": 8,
                "comparisons": len(value["statistics"]["comparisons"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
