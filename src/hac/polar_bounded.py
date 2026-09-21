"""Read-only parent evidence and finite candidate contracts for bounded adaptation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file
from hac.polar_benchmark import (
    canonical_hash,
    check_completed,
    implementation_evidence,
    label_names,
    load_development,
    probability_metrics,
)

SEEDS = (42, 52, 62)
MODEL_KINDS = ("siglip2_base", "convnextv2_base")
TASKS = (9, 4)
INCUMBENT = "uniform_top_two_validation"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def implementation_hashes(root: Path) -> dict[str, str]:
    hashes = implementation_evidence(root)
    for path in sorted((root / "experiments").glob("*polar_bounded*")):
        if path.suffix in {".py", ".json"}:
            hashes[path.relative_to(root).as_posix()] = sha256_file(path)
    return hashes


def manifest_path(parent: Path, classes: int) -> Path:
    if classes not in TASKS:
        raise ValueError("Only the locked four/nine-class tasks are supported")
    filename = (
        "polar9_development_manifest.csv"
        if classes == 9
        else "polar4_legacy_development_manifest.csv"
    )
    return parent / "data" / filename


def development_rows(path: Path, classes: int) -> pd.DataFrame:
    # Split-only admission check: do not parse test labels and then filter them away.
    splits = pd.read_csv(path, usecols=["split"], dtype=str, keep_default_na=False).split
    if set(splits) != {"train", "val"}:
        raise ValueError("Test is forbidden; exactly train/val development splits required")
    return load_development(path, num_classes=classes)


def load_probabilities(path: Path, validation: pd.DataFrame, classes: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        if not np.array_equal(
            archive["image_ids"], validation.image_id.to_numpy(dtype=str)
        ) or not np.array_equal(archive["labels"], validation.label_index.to_numpy()):
            raise ValueError(f"Prediction row/label mismatch: {path}")
        values = archive["probabilities"].astype(np.float64)
    if (
        values.shape != (len(validation), classes)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-7, rtol=0)
    ):
        raise ValueError(f"Invalid normalized probabilities: {path}")
    return values


def audit_parent(parent: Path) -> dict:
    """Verify completed evidence without changing it or opening any test predictions."""
    parent = parent.resolve(strict=True)
    status = read_json(parent / "queue_status.json")
    if (
        status.get("status") != "DEVELOPMENT_COMPLETE_REVIEW_REQUIRED"
        or status.get("test_evaluated") is not False
    ):
        raise RuntimeError("Parent must have completed development without test evaluation")
    bindings = {}

    def bind(path):
        path = Path(path).resolve(strict=True)
        if not path.is_relative_to(parent):
            raise ValueError("Parent artifact escapes its run directory")
        bindings[path.relative_to(parent).as_posix()] = sha256_file(path)

    for filename in ("queue_status.json", "queue_receipts.json", "development_summary.json"):
        bind(parent / filename)
    receipts = read_json(parent / "queue_receipts.json")
    completed = 0
    for receipt in receipts["jobs"]:
        if receipt["status"] != "COMPLETE":
            if receipt["status"] != "NOT_RUN_CHECKPOINT_UNAVAILABLE":
                raise RuntimeError("Parent contains a failed or unfinished job")
            continue
        marker = Path(receipt["completion_marker"])
        bind(marker)
        if sha256_file(marker) != receipt["marker_sha256"]:
            raise RuntimeError(f"Parent completion marker drift: {receipt['job']}")
        completed += 1
    source_audit = parent / "data" / "polar9_data_audit.json"
    bind(source_audit)
    data = read_json(source_audit)
    if (
        data.get("status") != "LOCKED_BEFORE_NEW_BENCHMARK_FITTING"
        or data.get("legacy_four_class_additional_nine_class_quarantine_flags") != 0
    ):
        raise RuntimeError("Expected completed matching source audit")
    for relative, evidence in data["artifacts"].items():
        path = (source_audit.parent / relative).resolve(strict=True)
        if not path.is_relative_to(source_audit.parent) or sha256_file(path) != evidence["sha256"]:
            raise RuntimeError("Audited source artifact changed")
        bind(path)
    report = read_json(parent / "development_summary.json")
    tasks = {}
    for classes in TASKS:
        task = f"polar{classes}"
        manifest = manifest_path(parent, classes)
        frame = development_rows(manifest, classes)
        validation = frame.loc[frame.split.eq("val")]
        result = report["tasks"][task]
        analysis = parent / task / "development_analysis"
        request_path = analysis / "analysis_request.json"
        request = read_json(request_path)
        if (
            request["manifest_sha256"] != sha256_file(manifest)
            or result["request_sha256"] != canonical_hash(request)
            or result["test_rows_read"] != 0
        ):
            raise RuntimeError("Parent analysis cohort/request mismatch")
        bind(request_path)
        bind(analysis / "analysis.json")
        scores = {}
        values = {}
        for candidate in result["candidates"]:
            name = candidate["candidate"]
            path = analysis / f"{name}_validation.npz"
            values[name] = load_probabilities(path, validation, classes)
            metrics = probability_metrics(
                validation.label_index.to_numpy(), values[name], label_names(frame)
            )
            for metric in ("macro_f1", "accuracy", "errors", "log_loss", "brier_score"):
                if not np.isclose(metrics[metric], candidate[metric], atol=1e-12, rtol=0):
                    raise RuntimeError(f"Parent metric replay failed: {task}/{name}/{metric}")
            bind(path)
            scores[name] = {key: metrics[key] for key in ("macro_f1", "accuracy", "errors", "rows")}
        for name, hashes in request["inputs"].items():
            if name == "dinov2_adapted_3seeds":
                directory = parent / task / "adaptation" / "confirmation"
                source_request_path = directory / "selection_lock.json"
                source_request = read_json(source_request_path)
                if check_completed(directory, source_request) is None:
                    raise RuntimeError("Missing completed parent seed ensemble")
                prediction_path = directory / "validation_predictions.npz"
            else:
                directory = parent / task / "heads" / name
                source_request_path = directory / "request.json"
                source_request = read_json(source_request_path)
                head = check_completed(directory, source_request)
                if head is None:
                    raise RuntimeError("Missing completed parent feature head")
                child = directory / head["best_candidate"]
                child_request = read_json(child / "request.json")
                if check_completed(child, child_request) is None:
                    raise RuntimeError("Missing parent winning head artifacts")
                prediction_path = child / "validation_predictions.npz"
                bind(child / "request.json")
                bind(child / "summary.json")
            if (
                hashes["summary"] != sha256_file(directory / "summary.json")
                or hashes["predictions"] != sha256_file(prediction_path)
                or source_request["manifest_sha256"] != sha256_file(manifest)
                or not np.array_equal(
                    values[name], load_probabilities(prediction_path, validation, classes)
                )
            ):
                raise RuntimeError("Parent analysis differs from its completed input")
            bind(source_request_path)
            bind(directory / "summary.json")
            bind(prediction_path)
        for fusion, members in result["fusion_members"].items():
            expected = np.mean([values[name] for name in members], axis=0)
            expected /= expected.sum(axis=1, keepdims=True)
            if not np.allclose(expected, values[fusion], atol=1e-12, rtol=0):
                raise RuntimeError("Prior fusion does not replay from declared members")
        tasks[task] = {"scores": scores, "incumbent": INCUMBENT, "test_rows_read": 0}
    for seed in SEEDS:
        directory = parent / "polar9" / "adaptation" / f"person_preserving_seed{seed}"
        replay = read_json(directory / "replay_audit.json")
        seed_request = read_json(directory / "request.json")
        if check_completed(directory, seed_request) is None:
            raise RuntimeError("Missing complete prior adapted seed")
        if (
            replay["status"] != "PASS"
            or not replay["labels_identical"]
            or not replay["predictions_identical"]
            or replay["maximum_probability_difference"] > 1e-5
        ):
            raise RuntimeError("A prior adapted seed failed checkpoint replay")
        for filename in (
            "request.json",
            "summary.json",
            "replay_audit.json",
            "validation_predictions.npz",
        ):
            bind(directory / filename)
    return {
        "status": "PARENT_EVIDENCE_VERIFIED_BEFORE_BOUNDED_PHASE",
        "parent_directory": str(parent),
        "completed_markers_verified": completed,
        "artifacts": bindings,
        "tasks": tasks,
        "test_predictions_read": 0,
    }


def verify_bindings(parent: Path, evidence: dict) -> None:
    for relative, digest in evidence["artifacts"].items():
        path = (parent / relative).resolve(strict=True)
        if not path.is_relative_to(parent.resolve()) or sha256_file(path) != digest:
            raise RuntimeError(f"Locked prior evidence changed: {relative}")


def candidate_family(incumbent, anchor, siglip, convolution):
    replacement = (anchor + siglip) / 2
    return {
        "retain_incumbent": incumbent.copy(),
        "siglip2_adapted_3seeds": siglip.copy(),
        "convnextv2_adapted_3seeds": convolution.copy(),
        "replacement_fusion": replacement,
        "conservative_fusion": (incumbent + replacement) / 2,
    }


def transitions(labels, reference, candidate) -> dict:
    old = reference.argmax(axis=1) != labels
    new = candidate.argmax(axis=1) != labels
    rescued = int((old & ~new).sum())
    harmed = int((~old & new).sum())
    union = int((old | new).sum())
    return {
        "rescued": rescued,
        "harmed": harmed,
        "net_corrections": rescued - harmed,
        "error_jaccard": float((old & new).sum() / union) if union else 1.0,
    }


def promotion_gate(metrics, reference, transition, seed_comparisons, gate):
    class_names = metrics["class_names"]
    class_drop = max(
        reference["per_class"][name]["f1-score"] - metrics["per_class"][name]["f1-score"]
        for name in class_names
    )
    checks = {
        "macro_f1_gain": metrics["macro_f1"] - reference["macro_f1"]
        >= gate["minimum_macro_f1_gain"],
        "positive_net_corrections": transition["net_corrections"]
        >= gate["minimum_net_corrections"],
        "all_seeds_positive_f1": len(seed_comparisons) == 3
        and all(row["macro_f1_gain"] > 0 for row in seed_comparisons),
        "all_seeds_positive_net": len(seed_comparisons) == 3
        and all(row["net_corrections"] > 0 for row in seed_comparisons),
        "per_class_safety": class_drop <= gate["maximum_single_class_f1_drop"],
        "nll_safety": metrics["log_loss"] - reference["log_loss"] <= gate["maximum_nll_increase"],
        "brier_safety": metrics["brier_score"] - reference["brier_score"]
        <= gate["maximum_brier_increase"],
    }
    return {"eligible": all(checks.values()), "checks": checks, "maximum_class_f1_drop": class_drop}


def _macro_f1(labels, predictions, weights, classes):
    matrix = np.bincount(
        labels * classes + predictions, weights=weights, minlength=classes**2
    ).reshape(classes, classes)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(
        np.divide(
            2 * matrix.diagonal(), denominator, out=np.zeros(classes), where=denominator > 0
        ).mean()
    )


def paired_bootstrap(labels, reference, candidate, source_groups, *, draws=1000, seed=20260921):
    """Descriptive selected-validation intervals, not confirmatory inference."""
    classes = reference.shape[1]
    old, new = reference.argmax(axis=1), candidate.argmax(axis=1)
    rng = np.random.default_rng(seed)
    pools = [np.flatnonzero(labels == label) for label in range(classes)]
    _, groups = np.unique(np.asarray(source_groups).astype(str), return_inverse=True)
    group_count = int(groups.max() + 1)
    stratified, clustered, rejected = [], [], 0
    for _ in range(draws):
        indices = np.concatenate([rng.choice(pool, len(pool), replace=True) for pool in pools])
        weights = np.bincount(indices, minlength=len(labels))
        stratified.append(
            _macro_f1(labels, new, weights, classes) - _macro_f1(labels, old, weights, classes)
        )
        for _attempt in range(1000):
            multiplicities = np.bincount(
                rng.integers(group_count, size=group_count), minlength=group_count
            )
            weights = multiplicities[groups]
            if all(weights[pool].sum() > 0 for pool in pools):
                break
            rejected += 1
        else:
            raise ValueError("Source-group bootstrap cannot retain every class in bounded draws")
        clustered.append(
            _macro_f1(labels, new, weights, classes) - _macro_f1(labels, old, weights, classes)
        )
    return {
        "draws": draws,
        "seed": seed,
        "row_stratified_95_interval": np.quantile(stratified, [0.025, 0.975]).tolist(),
        "source_group_95_interval": np.quantile(clustered, [0.025, 0.975]).tolist(),
        "source_groups": group_count,
        "source_group_rejected_missing_class_draws": rejected,
        "source_group_conditioning": "every_declared_class_present",
        "role": "descriptive_selected_validation_not_confirmatory_or_multiplicity_adjusted",
    }
