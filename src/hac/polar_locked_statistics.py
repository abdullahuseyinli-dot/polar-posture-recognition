"""CPU statistics for predeclared POLAR predictions, never model selection.

All resampling is paired. Group inference conditions on detected source groups,
not on unobserved subject/session identities. Test exposure is a separate protocol
limitation that cannot be repaired by a small p-value.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from hac.metrics import classification_metrics

DEFAULT_SEED = 20260921
DEFAULT_BOOTSTRAP_DRAWS = 5000
DEFAULT_RANDOMIZATION_DRAWS = 10000


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained_path(root: Path, value: str | Path) -> Path:
    root = Path(root).resolve(strict=True)
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve(strict=True)
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Prediction artifact escapes its declared evaluation directory")
    return path


def validate_arrays(labels, probabilities, *, classes: int | None = None):
    raw_labels = np.asarray(labels)
    values = np.asarray(probabilities, dtype=np.float64)
    if raw_labels.ndim != 1 or not len(raw_labels):
        raise ValueError("Expected nonempty one-dimensional labels")
    if raw_labels.dtype.kind not in "iuf" or not np.isfinite(raw_labels).all():
        raise ValueError("Labels must be finite numeric integers")
    if not np.equal(raw_labels, np.floor(raw_labels)).all():
        raise ValueError("Fractional class labels are forbidden")
    labels = raw_labels.astype(np.int64)
    if values.ndim != 2 or len(values) != len(labels) or values.shape[1] < 2:
        raise ValueError("Expected aligned multiclass probability matrix")
    classes = values.shape[1] if classes is None else classes
    if values.shape[1] != classes or set(labels) != set(range(classes)):
        raise ValueError("Evaluation must contain exactly every declared class")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError("Probabilities must be finite and within [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-7, rtol=0):
        raise ValueError("Probability rows must sum to one; no silent renormalization")
    return labels, values


def load_prediction(
    root: Path,
    record: Mapping,
    *,
    image_ids: Sequence,
    labels: np.ndarray,
    classes: int,
) -> np.ndarray:
    path = contained_path(root, record["path"])
    if not isinstance(record.get("sha256"), str) or file_sha256(path) != record["sha256"]:
        raise ValueError("Prediction bytes differ from their locked manifest")
    expected_ids = np.asarray(image_ids, dtype=str)
    if len(expected_ids) != len(set(expected_ids.tolist())):
        raise ValueError("Evaluation image IDs must be unique")
    with np.load(path, allow_pickle=False) as saved:
        if not {"image_ids", "labels", "probabilities"}.issubset(saved.files):
            raise ValueError("Prediction archive has an incomplete schema")
        if not np.array_equal(saved["image_ids"], expected_ids):
            raise ValueError("Prediction image IDs or ordering differ from the locked cohort")
        if not np.array_equal(saved["labels"], labels):
            raise ValueError("Prediction labels differ from the locked cohort")
        _, values = validate_arrays(saved["labels"], saved["probabilities"], classes=classes)
    return values


def confusion_from_codes(codes: np.ndarray, classes: int, weights=None) -> np.ndarray:
    return np.bincount(codes, weights=weights, minlength=classes * classes).reshape(
        classes, classes
    )


def macro_f1_from_confusion(matrix: np.ndarray) -> float:
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(
        np.divide(
            2 * matrix.diagonal(),
            denominator,
            out=np.zeros(len(matrix), dtype=float),
            where=denominator > 0,
        ).mean()
    )


def _macro_accuracy(codes, classes, weights=None):
    matrix = confusion_from_codes(codes, classes, weights)
    return macro_f1_from_confusion(matrix), float(matrix.diagonal().sum() / matrix.sum())


def metrics_summary(labels, probabilities, class_names: Sequence[str]) -> dict:
    if len(class_names) != len(set(class_names)):
        raise ValueError("Class names must be unique")
    labels, values = validate_arrays(labels, probabilities, classes=len(class_names))
    classes = len(class_names)
    predictions = values.argmax(axis=1)
    matrix = confusion_from_codes(labels * classes + predictions, classes)
    precision = np.divide(
        matrix.diagonal(), matrix.sum(axis=0), out=np.zeros(classes), where=matrix.sum(axis=0) > 0
    )
    recall = matrix.diagonal() / matrix.sum(axis=1)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(classes),
        where=(precision + recall) > 0,
    )
    confidence = values.max(axis=1)
    ordered_values = np.partition(values, -2, axis=1)
    margin = ordered_values[:, -1] - ordered_values[:, -2]
    entropy = -np.sum(values * np.log(np.maximum(values, np.finfo(float).tiny)), axis=1)
    errors = predictions != labels
    order = np.argsort(-confidence, kind="stable")
    risks = np.cumsum(errors[order]) / np.arange(1, len(labels) + 1)
    selective = []
    for coverage in (0.5, 0.8, 0.9, 0.95, 1.0):
        count = int(np.ceil(coverage * len(labels)))
        selective.append(
            {
                "requested_coverage": coverage,
                "retained": count,
                "realized_coverage": count / len(labels),
                "risk": float(risks[count - 1]),
            }
        )
    return {
        **classification_metrics(labels, values),
        "rows": len(labels),
        "errors": int(errors.sum()),
        "class_names": list(class_names),
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(matrix[i].sum()),
            }
            for i, name in enumerate(class_names)
        },
        "confidence": {
            "mean": float(confidence.mean()),
            "mean_margin": float(margin.mean()),
            "mean_entropy": float(entropy.mean()),
            "mean_on_correct": float(confidence[~errors].mean()) if (~errors).any() else None,
            "mean_on_errors": float(confidence[errors].mean()) if errors.any() else None,
        },
        "selective_risk": selective,
        "area_under_risk_coverage_curve": float(risks.mean()),
        "selective_role": "fixed_coverage_diagnostic_no_threshold_selected",
    }


def error_transitions(labels, reference, candidate) -> dict:
    labels, reference = validate_arrays(labels, reference)
    _, candidate = validate_arrays(labels, candidate, classes=reference.shape[1])
    old_predictions, new_predictions = reference.argmax(axis=1), candidate.argmax(axis=1)
    old, new = old_predictions != labels, new_predictions != labels
    rescue, harm = int((old & ~new).sum()), int((~old & new).sum())
    both, union = int((old & new).sum()), int((old | new).sum())
    return {
        "reference_errors": int(old.sum()),
        "candidate_errors": int(new.sum()),
        "rescued": rescue,
        "harmed": harm,
        "net_corrections": rescue - harm,
        "both_wrong": both,
        "both_correct": int((~old & ~new).sum()),
        "error_jaccard": both / union if union else 1.0,
        "prediction_disagreement_count": int((old_predictions != new_predictions).sum()),
        "prediction_disagreement_rate": float((old_predictions != new_predictions).mean()),
        "candidate_error_given_reference_error": both / int(old.sum()) if old.any() else None,
        "candidate_error_given_reference_correct": harm / int((~old).sum())
        if (~old).any()
        else None,
    }


def _group_indices(groups, rows):
    values = np.asarray(groups)
    if values.ndim != 1 or len(values) != rows:
        raise ValueError("Source groups must align with evaluation rows")
    if any(
        value is None or (isinstance(value, float) and not np.isfinite(value)) for value in values
    ):
        raise ValueError("Source group identifiers cannot be missing")
    values = values.astype(str)
    if any(not value.strip() for value in values):
        raise ValueError("Source group identifiers cannot be empty")
    names, inverse = np.unique(values, return_inverse=True)
    return names, inverse


def paired_bootstrap(
    labels, reference, candidate, source_groups, *, draws=DEFAULT_BOOTSTRAP_DRAWS, seed=DEFAULT_SEED
) -> dict:
    """Paired percentile intervals, conditional on all classes in cluster draws."""
    if not isinstance(draws, int) or draws < 1:
        raise ValueError("Bootstrap draws must be a positive integer")
    labels, reference = validate_arrays(labels, reference)
    _, candidate = validate_arrays(labels, candidate, classes=reference.shape[1])
    classes = reference.shape[1]
    group_names, inverse = _group_indices(source_groups, len(labels))
    pools = [np.flatnonzero(labels == label) for label in range(classes)]
    codes = [labels * classes + matrix.argmax(axis=1) for matrix in (reference, candidate)]
    row_values, group_values = np.empty((draws, 2, 2)), np.empty((draws, 2, 2))
    row_rng, group_rng = np.random.default_rng(seed), np.random.default_rng(seed + 1)
    rejected = 0
    for draw in range(draws):
        indices = np.concatenate(
            [row_rng.choice(pool, size=len(pool), replace=True) for pool in pools]
        )
        weights = np.bincount(indices, minlength=len(labels))
        row_values[draw] = [_macro_accuracy(code, classes, weights) for code in codes]
        for _ in range(10000):
            multiplicity = np.bincount(
                group_rng.integers(len(group_names), size=len(group_names)),
                minlength=len(group_names),
            )
            weights = multiplicity[inverse]
            if all(weights[pool].sum() > 0 for pool in pools):
                break
            rejected += 1
        else:
            raise ValueError(
                "Cluster bootstrap cannot retain every class within its fixed retry cap"
            )
        group_values[draw] = [_macro_accuracy(code, classes, weights) for code in codes]
    points = np.array([_macro_accuracy(code, classes) for code in codes])
    output = {
        "draws": draws,
        "row_seed": seed,
        "source_group_seed": seed + 1,
        "source_groups": len(group_names),
        "source_group_missing_class_draws_rejected": rejected,
        "source_group_conditioning": "all_declared_classes_present",
        "interval": "paired_percentile_95_not_simultaneous_or_selection_adjusted",
    }
    for metric_index, name in enumerate(("macro_f1", "accuracy")):
        output[name] = {}
        for model_index, role in enumerate(("reference", "candidate")):
            output[name][role] = {
                "point_estimate": float(points[model_index, metric_index]),
                "row_stratified_ci95": np.quantile(
                    row_values[:, model_index, metric_index], [0.025, 0.975]
                ).tolist(),
                "source_group_ci95": np.quantile(
                    group_values[:, model_index, metric_index], [0.025, 0.975]
                ).tolist(),
            }
        output[name]["delta"] = {
            "point_estimate": float(points[1, metric_index] - points[0, metric_index]),
            "row_stratified_ci95": np.quantile(
                row_values[:, 1, metric_index] - row_values[:, 0, metric_index], [0.025, 0.975]
            ).tolist(),
            "source_group_ci95": np.quantile(
                group_values[:, 1, metric_index] - group_values[:, 0, metric_index], [0.025, 0.975]
            ).tolist(),
        }
    return output


def paired_randomization(
    labels,
    reference,
    candidate,
    source_groups,
    *,
    draws=DEFAULT_RANDOMIZATION_DRAWS,
    seed=DEFAULT_SEED,
) -> dict:
    """Two-sided groupwise prediction exchange, plus-one Monte Carlo p-value.

    Identical model predictions cancel. Only groups containing a discordant row
    need a random swap, while each group's swap applies to all its differing rows.
    """
    if not isinstance(draws, int) or draws < 1:
        raise ValueError("Randomization draws must be a positive integer")
    labels, reference = validate_arrays(labels, reference)
    _, candidate = validate_arrays(labels, candidate, classes=reference.shape[1])
    classes = reference.shape[1]
    _group_indices(source_groups, len(labels))
    old, new = reference.argmax(axis=1), candidate.argmax(axis=1)
    old_codes, new_codes = labels * classes + old, labels * classes + new
    old_matrix = confusion_from_codes(old_codes, classes)
    new_matrix = confusion_from_codes(new_codes, classes)
    observed = macro_f1_from_confusion(new_matrix) - macro_f1_from_confusion(old_matrix)
    different = old != new
    affected, inverse = np.unique(
        np.asarray(source_groups).astype(str)[different], return_inverse=True
    )
    extreme = 0
    if different.any():
        rng = np.random.default_rng(seed)
        for _ in range(draws):
            swapped = rng.integers(0, 2, size=len(affected))[inverse]
            change = confusion_from_codes(
                old_codes[different], classes, swapped
            ) - confusion_from_codes(new_codes[different], classes, swapped)
            delta = macro_f1_from_confusion(new_matrix + change) - macro_f1_from_confusion(
                old_matrix - change
            )
            extreme += abs(delta) >= abs(observed) - 1e-12
    else:
        extreme = draws
    return {
        "statistic": "macro_f1_candidate_minus_reference",
        "observed_delta": float(observed),
        "alternative": "two_sided",
        "exchange_unit": "detected_source_group",
        "draws": draws,
        "seed": seed,
        "affected_groups": len(affected),
        "discordant_prediction_rows": int(different.sum()),
        "extreme_draws": int(extreme),
        "p_value": float((extreme + 1) / (draws + 1)),
        "correction": "plus_one_monte_carlo",
        "assumption": "model_assignments_exchangeable_under_null_at_detected_group_level; unobserved_identity_dependence_not_resolved",
    }


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=float)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values > 1).any()
    ):
        raise ValueError("Holm adjustment requires a nonempty list of valid p-values")
    order = np.argsort(values, kind="stable")
    adjusted = np.minimum(
        1.0, np.maximum.accumulate(values[order] * (len(values) - np.arange(len(values))))
    )
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return restored.tolist()


def paired_seed_summary(
    labels, reference, candidates: Mapping[str, np.ndarray], *, expected_seeds=(42, 52, 62)
) -> dict:
    expected = [str(seed) for seed in expected_seeds]
    if set(candidates) != set(expected):
        raise ValueError(
            "All declared seeds must be present; favorable-seed selection is forbidden"
        )
    paired_references = isinstance(reference, Mapping)
    if paired_references and set(reference) != set(expected):
        raise ValueError("Every candidate seed requires its matching reference seed")
    references = reference if paired_references else {seed: reference for seed in expected}
    rows = []
    for seed in expected:
        labels, reference_values = validate_arrays(labels, references[seed])
        classes = reference_values.shape[1]
        reference_f1 = _macro_accuracy(labels * classes + reference_values.argmax(axis=1), classes)[
            0
        ]
        _, values = validate_arrays(labels, candidates[seed], classes=classes)
        score = _macro_accuracy(labels * classes + values.argmax(axis=1), classes)[0]
        rows.append(
            {
                "seed": int(seed),
                "macro_f1": score,
                "reference_macro_f1": reference_f1,
                "macro_f1_gain": score - reference_f1,
                **error_transitions(labels, reference_values, values),
            }
        )
    gains = np.asarray([row["macro_f1_gain"] for row in rows])
    return {
        "seeds": rows,
        "mean_paired_macro_f1_gain": float(gains.mean()),
        "sample_sd_paired_macro_f1_gain": float(gains.std(ddof=1)) if len(gains) > 1 else None,
        "all_seeds_positive_macro_f1_gain": bool((gains > 0).all()),
        "all_seeds_positive_net_corrections": all(row["net_corrections"] > 0 for row in rows),
        "role": (
            "matching_reference_seed_sensitivity_not_independent_test_replicates"
            if paired_references
            else "fixed_reference_seed_sensitivity_not_independent_test_replicates"
        ),
    }


def locked_promotion_gate(
    candidate_metrics: Mapping,
    reference_metrics: Mapping,
    comparison: Mapping,
    seed_diagnostic: Mapping,
    gates: Mapping,
) -> dict:
    """Evaluate fixed acceptance checks, without selecting another test candidate."""
    differences = {
        name: candidate_metrics["per_class"][name]["f1"] - values["f1"]
        for name, values in reference_metrics["per_class"].items()
    }
    delta = candidate_metrics["macro_f1"] - reference_metrics["macro_f1"]
    nll_delta = candidate_metrics["log_loss"] - reference_metrics["log_loss"]
    brier_delta = candidate_metrics["brier_score"] - reference_metrics["brier_score"]
    checks = {
        "positive_macro_f1": delta > 0,
        "positive_net_corrections": comparison["transitions"]["net_corrections"] > 0,
        "source_group_delta_ci_lower": comparison["bootstrap"]["macro_f1"]["delta"][
            "source_group_ci95"
        ][0]
        > gates["source_group_delta_ci_lower_gt"],
        "global_holm_p": comparison["holm_p_value"] < gates["holm_p_lt"],
        "no_excess_per_class_f1_drop": min(differences.values()) >= -gates["max_per_class_f1_drop"],
        "nll_increase_within_limit": nll_delta <= gates["max_nll_increase"],
        "brier_increase_within_limit": brier_delta <= gates["max_brier_increase"],
        "all_seed_counterparts_positive_macro_f1": seed_diagnostic[
            "all_seeds_positive_macro_f1_gain"
        ],
        "all_seed_counterparts_positive_net_corrections": seed_diagnostic[
            "all_seeds_positive_net_corrections"
        ],
    }
    return {
        "all_checks_passed": all(checks.values()),
        "checks": checks,
        "macro_f1_gain": delta,
        "per_class_f1_gain": differences,
        "nll_increase": nll_delta,
        "brier_increase": brier_delta,
        "decision_role": "evaluate_preselected_nominee_only_no_test_winner_switching",
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }
