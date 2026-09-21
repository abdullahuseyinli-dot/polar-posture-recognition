"""Analyze the sealed POLAR comparison family, never select a test winner."""

from __future__ import annotations

import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar_benchmark import lock_json
from hac.polar_locked_evaluation import (
    read_json,
    validate_test_rows,
    verify_selection_lock,
    verify_test_access,
)
from hac.polar_locked_statistics import (
    DEFAULT_BOOTSTRAP_DRAWS,
    DEFAULT_RANDOMIZATION_DRAWS,
    DEFAULT_SEED,
    contained_path,
    error_transitions,
    file_sha256,
    holm_adjust,
    load_prediction,
    locked_promotion_gate,
    metrics_summary,
    paired_bootstrap,
    paired_randomization,
    paired_seed_summary,
)

STATUS = "POLAR_LOCKED_STATISTICS_COMPLETE"
SEEDS = (42, 52, 62)
NOMINEE = "conservative_fusion"
GATES = {
    "positive_macro_f1": True,
    "positive_net_corrections": True,
    "source_group_delta_ci_lower_gt": 0,
    "holm_p_lt": 0.05,
    "max_per_class_f1_drop": 0.01,
    "max_nll_increase": 0.02,
    "max_brier_increase": 0.01,
    "all_seed_counterparts_positive": True,
}
COMMON_CANDIDATES = {
    NOMINEE,
    "prior_incumbent",
    "replacement_fusion",
    "frozen_dinov2_base",
    "frozen_siglip2_base",
    "frozen_dinov3_base",
    "frozen_convnextv2_base",
    "adapted_siglip2",
    "adapted_convnextv2",
}
SEED_CANDIDATES = {f"{prefix}_seed{seed}" for prefix in ("conservative", "prior") for seed in SEEDS}


def expected_candidates(task: str) -> set[str]:
    if task not in {"polar4", "polar9"}:
        raise ValueError("Undeclared task")
    return COMMON_CANDIDATES | {"adapted_dinov2" if task == "polar9" else "historical_ensemble"}


def validate_statistical_contract(lock: dict) -> list[dict]:
    """Reject changed families, nominees, seeds, thresholds, or analysis budgets."""
    statistics = lock["statistics"]
    expected = {
        "bootstrap_draws": DEFAULT_BOOTSTRAP_DRAWS,
        "randomization_draws": DEFAULT_RANDOMIZATION_DRAWS,
        "seed": DEFAULT_SEED,
        "alpha": 0.05,
        "multiplicity": "one_Holm_family_all_18_comparisons",
        "randomization_unit": "detected_source_group",
        "alternative": "two_sided",
        "gates": GATES,
    }
    if any(statistics.get(name) != value for name, value in expected.items()):
        raise ValueError("Statistics differ from the predeclared final analysis")
    if set(lock["tasks"]) != {"polar9", "polar4"}:
        raise ValueError("Both locked tasks are required")
    expected_comparisons = {}
    for task, specification in lock["tasks"].items():
        names = specification["candidates"]
        if (
            len(names) != 10
            or set(names) != expected_candidates(task)
            or specification["nominated_candidate"] != NOMINEE
        ):
            raise ValueError("The fixed nominee/comparator family cannot change")
        strongest = "adapted_dinov2" if task == "polar9" else "frozen_siglip2_base"
        for reference in names:
            if reference != NOMINEE:
                identifier = f"{task}__vs__{reference}"
                expected_comparisons[identifier] = {
                    "id": identifier,
                    "task": task,
                    "candidate": NOMINEE,
                    "reference": reference,
                    "primary": reference in {"prior_incumbent", strongest, "historical_ensemble"},
                }
    comparisons = statistics["comparisons"]
    if (
        len(comparisons) != 18
        or len({item["id"] for item in comparisons}) != 18
        or {item["id"]: item for item in comparisons} != expected_comparisons
    ):
        raise ValueError("The complete 18-comparison Holm family is mandatory")
    if (
        lock.get("test_selection_forbidden") is not True
        or lock.get("automatic_search_after_test") is not False
    ):
        raise ValueError("Test-based selection/search must remain forbidden")
    return comparisons


def load_locked_inputs(selection_lock: Path, prediction_manifest: Path) -> dict:
    """Verify the complete fit barrier before reading any test labels/predictions."""
    selection_lock = selection_lock.resolve(strict=True)
    run = selection_lock.parent
    lock = verify_selection_lock(selection_lock)
    comparisons = validate_statistical_contract(lock)
    verify_test_access(selection_lock, run)
    prediction_manifest = prediction_manifest.resolve(strict=True)
    if prediction_manifest != run / "evaluation" / "prediction_manifest.json":
        raise ValueError("Prediction manifest must be the locked run's evaluation manifest")
    manifest_digest = file_sha256(prediction_manifest)
    manifest = read_json(prediction_manifest)
    if (
        manifest.get("selection_lock_sha256") != file_sha256(selection_lock)
        or manifest.get("gate_sha256") != file_sha256(run / "test_access_gate.json")
        or set(manifest["tasks"]) != set(lock["tasks"])
    ):
        raise ValueError("Predictions are not bound to this selection lock and test gate")
    evaluation_root = prediction_manifest.parent
    loaded = {}
    for task, specification in lock["tasks"].items():
        record = manifest["tasks"][task]
        if (
            record["rows"] != specification["test_rows"]
            or record["manifest_sha256"] != specification["test_manifest_sha256"]
            or record["class_names"] != specification["class_names"]
            or set(record["candidates"]) != expected_candidates(task) | SEED_CANDIDATES
        ):
            raise ValueError(f"Task cohort or candidate family mismatch: {task}")
        dtype = {"image_id": str, "source_group": str}
        frame = validate_test_rows(
            pd.read_csv(specification["test_manifest"], dtype=dtype),
            specification,
            pd.read_csv(specification["development_manifest"], dtype=dtype),
        )
        groups_path = contained_path(evaluation_root, record["source_groups_path"])
        if file_sha256(groups_path) != record.get("source_groups_sha256"):
            raise ValueError("Source-group artifact integrity failure")
        groups = read_json(groups_path)
        if not isinstance(groups, list) or not np.array_equal(
            np.asarray(groups), frame.source_group.to_numpy(dtype=str)
        ):
            raise ValueError("Source groups/order differ from the audited test manifest")
        values = {}
        for name, candidate in record["candidates"].items():
            expected_kind = "seed_diagnostic" if name in SEED_CANDIDATES else "locked_comparison"
            if candidate.get("kind") != expected_kind:
                raise ValueError("Comparator/seed roles cannot be silently changed")
            values[name] = load_prediction(
                evaluation_root,
                candidate,
                image_ids=frame.image_id.to_numpy(dtype=str),
                labels=frame.label_index.to_numpy(),
                classes=specification["classes"],
            )
        # All seed artifacts are required and must average to the declared fusion.
        for prefix, aggregate in (("conservative", NOMINEE), ("prior", "prior_incumbent")):
            mean = np.mean([values[f"{prefix}_seed{seed}"] for seed in SEEDS], axis=0)
            if not np.allclose(mean, values[aggregate], atol=1e-7, rtol=0):
                raise ValueError("Seed diagnostic probabilities do not reproduce their aggregate")
        anchor = values["adapted_dinov2" if task == "polar9" else "frozen_dinov2_base"]
        formulas = {
            "prior_incumbent": 0.5 * anchor + 0.5 * values["frozen_siglip2_base"],
            "replacement_fusion": 0.5 * anchor + 0.5 * values["adapted_siglip2"],
            NOMINEE: 0.5 * anchor
            + 0.25 * values["frozen_siglip2_base"]
            + 0.25 * values["adapted_siglip2"],
        }
        if any(
            not np.allclose(values[name], formula, atol=1e-7, rtol=0)
            for name, formula in formulas.items()
        ):
            raise ValueError("Prediction probabilities violate the predeclared fusion formula")
        loaded[task] = {
            "image_ids": frame.image_id.to_numpy(dtype=str),
            "labels": frame.label_index.to_numpy(),
            "source_groups": np.asarray(groups, dtype=str),
            "probabilities": values,
            "class_names": specification["class_names"],
        }
    return {
        "selection_lock": selection_lock,
        "prediction_manifest": prediction_manifest,
        "selection_lock_sha256": file_sha256(selection_lock),
        "prediction_manifest_sha256": manifest_digest,
        "lock": lock,
        "manifest": manifest,
        "comparisons": comparisons,
        "tasks": loaded,
    }


def analyze_inputs(inputs: dict) -> dict:
    """Compute every locked comparison in declaration order, without score sorting."""
    tasks = {}
    for task, data in inputs["tasks"].items():
        probabilities, labels = data["probabilities"], data["labels"]
        tasks[task] = {
            "rows": len(labels),
            "source_groups": len(set(data["source_groups"].tolist())),
            "nominated_candidate": NOMINEE,
            "recorded_costs": inputs["manifest"]["tasks"][task].get("costs"),
            "metrics": {
                name: metrics_summary(labels, probabilities[name], data["class_names"])
                for name in inputs["lock"]["tasks"][task]["candidates"]
            },
            "seed_diagnostic": paired_seed_summary(
                labels,
                {str(seed): probabilities[f"prior_seed{seed}"] for seed in SEEDS},
                {str(seed): probabilities[f"conservative_seed{seed}"] for seed in SEEDS},
            ),
        }
    comparisons = []
    for index, specification in enumerate(inputs["comparisons"], 1):
        print(f"Locked comparison {index}/18: {specification['id']}", flush=True)
        data = inputs["tasks"][specification["task"]]
        reference = data["probabilities"][specification["reference"]]
        candidate = data["probabilities"][specification["candidate"]]
        resampling = (data["labels"], reference, candidate, data["source_groups"])
        comparisons.append(
            {
                **specification,
                "transitions": error_transitions(data["labels"], reference, candidate),
                "bootstrap": paired_bootstrap(
                    *resampling, draws=DEFAULT_BOOTSTRAP_DRAWS, seed=DEFAULT_SEED
                ),
                "randomization": paired_randomization(
                    *resampling, draws=DEFAULT_RANDOMIZATION_DRAWS, seed=DEFAULT_SEED
                ),
            }
        )
    adjusted = holm_adjust([row["randomization"]["p_value"] for row in comparisons])
    for row, corrected in zip(comparisons, adjusted, strict=True):
        row["holm_p_value"] = corrected
        row["holm_family_size"] = len(comparisons)
        row["superiority_conditions_met_on_this_cohort"] = (
            row["bootstrap"]["macro_f1"]["delta"]["point_estimate"] > 0
            and row["bootstrap"]["macro_f1"]["delta"]["source_group_ci95"][0] > 0
            and corrected < 0.05
        )
    for task, result in tasks.items():
        prior_comparison = next(
            row for row in comparisons if row["id"] == f"{task}__vs__prior_incumbent"
        )
        result["promotion_gate"] = locked_promotion_gate(
            result["metrics"][NOMINEE],
            result["metrics"]["prior_incumbent"],
            prior_comparison,
            result["seed_diagnostic"],
            inputs["lock"]["statistics"]["gates"],
        )
        result["nominee_unchanged_after_test"] = True
        for comparison in (row for row in comparisons if row["task"] == task):
            reference, candidate = comparison["reference"], comparison["candidate"]
            for name, role in ((reference, "reference"), (candidate, "candidate")):
                result["metrics"][name]["uncertainty"] = {
                    metric: comparison["bootstrap"][metric][role]
                    for metric in ("macro_f1", "accuracy")
                }
    return {
        "status": STATUS,
        "selection_lock_sha256": inputs["selection_lock_sha256"],
        "prediction_manifest_sha256": inputs["prediction_manifest_sha256"],
        "statistics": inputs["lock"]["statistics"],
        "tasks": tasks,
        "comparisons": comparisons,
        "historical_exposure": inputs["lock"]["historical_exposure"],
        "cost_records": inputs["manifest"].get("cost_records"),
        "cost_scope": "recorded_panel_or_session_costs_only; resumed_sessions_may_omit_prior_runtime; feature_extraction_may_be_excluded; missing_is_not_zero; not_deployable_per_image_latency",
        "inference_scope": "preselected_models_on_historically_exposed_cohorts_not_fresh_independent_confirmation",
        "interval_scope": "paired_percentile_95; not_simultaneous_or_selection_adjusted; groups_not_verified_subjects",
        "test_used_for_model_or_threshold_selection": False,
        "automatic_model_promotion": False,
        "limitations": inputs["lock"].get("limitations", []),
    }


def _write_once(path: Path, content: bytes) -> None:
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Refusing to replace a different analysis artifact: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(content)


def _csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def render_reports(summary: dict, output: Path) -> list[str]:
    """Fixed-order tables and two diagnostic plots, without a test-ranked winner."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_rows, comparison_rows = [], []
    lines = [
        "# Locked POLAR evaluation",
        "",
        "The development-nominated model remains `conservative_fusion` for both tasks. "
        "This analysis does not choose a new winner, tune a threshold, or modify historical results.",
        "",
        "The four-class test was previously inspected; the nine-class test includes that "
        "exposed subset. These results are not fresh independent confirmation or a verified "
        "state-of-the-art claim. Detected source groups are not verified subject identities.",
        "",
        "## Predeclared promotion checks",
        "",
        "| Task | Nominee macro-F1 | Accuracy | Errors | All checks vs prior incumbent |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for task, result in summary["tasks"].items():
        score = result["metrics"][NOMINEE]
        passed = result["promotion_gate"]["all_checks_passed"]
        lines.append(
            f"| {task} | {100 * score['macro_f1']:.3f}% | "
            f"{100 * score['accuracy']:.3f}% | {score['errors']} | {'PASS' if passed else 'FAIL'} |"
        )
        for name, metrics in result["metrics"].items():
            row_ci = metrics["uncertainty"]["macro_f1"]["row_stratified_ci95"]
            group_ci = metrics["uncertainty"]["macro_f1"]["source_group_ci95"]
            metric_rows.append(
                {
                    "task": task,
                    "candidate": name,
                    "development_nominee": name == NOMINEE,
                    **{
                        key: metrics[key]
                        for key in (
                            "rows",
                            "macro_f1",
                            "accuracy",
                            "errors",
                            "log_loss",
                            "brier_score",
                            "ece",
                        )
                    },
                    "macro_f1_row_ci_low": row_ci[0],
                    "macro_f1_row_ci_high": row_ci[1],
                    "macro_f1_source_group_ci_low": group_ci[0],
                    "macro_f1_source_group_ci_high": group_ci[1],
                }
            )
    lines.extend(
        ["", "Each failed check is reported explicitly; no alternate model is selected.", ""]
    )
    for task, result in summary["tasks"].items():
        failed = result["promotion_gate"]["failed_checks"]
        lines.append(
            f"- {task}: {', '.join(failed) if failed else 'all predeclared checks passed'}."
        )
    lines.extend(
        [
            "",
            "## Fixed comparison family",
            "",
            "All 18 two-sided source-group randomization tests share one Holm correction "
            "(10,000 draws, +1 correction). Intervals are paired percentile 95% intervals "
            "from 5,000 source-group draws; they are not simultaneous or selection-adjusted. "
            "Row-stratified intervals are also recorded in `summary.json`. All seeds are included.",
            "",
            "| Task | Reference | Primary | F1 gain (pp) | Group 95% interval (pp) | Rescue / harm | Holm p |",
            "| --- | --- | --- | ---: | --- | ---: | ---: |",
        ]
    )
    for row in summary["comparisons"]:
        delta = row["bootstrap"]["macro_f1"]["delta"]
        low, high = delta["source_group_ci95"]
        changes = row["transitions"]
        lines.append(
            f"| {row['task']} | {row['reference']} | {'yes' if row['primary'] else 'no'} | "
            f"{100 * delta['point_estimate']:+.3f} | [{100 * low:+.3f}, {100 * high:+.3f}] | "
            f"{changes['rescued']} / {changes['harmed']} | {row['holm_p_value']:.5f} |"
        )
        comparison_rows.append(
            {
                **{key: row[key] for key in ("id", "task", "candidate", "reference", "primary")},
                "macro_f1_gain": delta["point_estimate"],
                "group_ci_low": low,
                "group_ci_high": high,
                "randomization_p": row["randomization"]["p_value"],
                "holm_p": row["holm_p_value"],
                **changes,
            }
        )
    lines.extend(
        [
            "",
            "![Fixed comparison deltas](comparison_f1.png)",
            "",
            "![Nominated-model confusion matrices](nominee_confusion.png)",
            "",
            "Complete classwise precision/recall/F1, probability metrics, fixed-coverage risks, "
            "seed sensitivity, runtime records (when supplied), hashes, and all checks are in "
            "`summary.json`. No inference-latency benchmark is inferred from training runtime.",
            "",
        ]
    )
    artifacts = {
        "report.md": "\n".join(lines).encode("utf-8"),
        "metrics.csv": _csv_bytes(metric_rows),
        "comparisons.csv": _csv_bytes(comparison_rows),
    }
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    for axis, (task, result) in zip(axes, summary["tasks"].items(), strict=True):
        score = result["metrics"][NOMINEE]
        matrix = np.asarray(score["confusion_matrix"])
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        axis.set_xticks(range(len(matrix)), score["class_names"], rotation=60, ha="right")
        axis.set_yticks(range(len(matrix)), score["class_names"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.set_title(
            f"{task}: development-nominated conservative fusion\nCounts; color = within-class recall"
        )
        for i, j in np.ndindex(matrix.shape):
            axis.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized[i, j] > 0.5 else "black",
            )
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    artifacts["nominee_confusion.png"] = buffer.getvalue()
    figure, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    for axis, task in zip(axes, summary["tasks"], strict=True):
        rows = [row for row in summary["comparisons"] if row["task"] == task]
        for index, row in enumerate(rows):
            delta = row["bootstrap"]["macro_f1"]["delta"]
            low, high = np.asarray(delta["source_group_ci95"]) * 100
            axis.plot([low, high], [index, index], color="#246b93", linewidth=2)
            axis.plot(100 * delta["point_estimate"], index, "o", color="#153e59")
        axis.axvline(0, color="#555555", linewidth=1, linestyle="--")
        axis.set_yticks(range(len(rows)), [row["reference"] for row in rows])
        axis.invert_yaxis()
        axis.set_xlabel("Nominee minus reference: macro-F1 percentage points")
        axis.set_title(
            f"{task}: fixed-order comparisons\nPaired source-group 95% intervals (not simultaneous)"
        )
        axis.grid(axis="x", alpha=0.2)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    artifacts["comparison_f1.png"] = buffer.getvalue()
    for name, content in artifacts.items():
        _write_once(output / name, content)
    return list(artifacts)


def run_analysis(selection_lock: Path, prediction_manifest: Path, output_dir: Path) -> dict:
    inputs = load_locked_inputs(selection_lock, prediction_manifest)
    run = inputs["selection_lock"].parent
    output = output_dir.resolve()
    if (
        not output.is_relative_to(run)
        or output == run
        or output.is_relative_to(run / "evaluation")
        or output.is_relative_to(run / "inputs")
    ):
        raise ValueError("Analysis output must be a new non-input subdirectory of the locked run")
    output.mkdir(parents=True, exist_ok=True)
    request = {
        "selection_lock_sha256": inputs["selection_lock_sha256"],
        "prediction_manifest_sha256": inputs["prediction_manifest_sha256"],
        "statistics": inputs["lock"]["statistics"],
        "implementation_sha256": file_sha256(Path(__file__)),
        "purpose": "evaluate_fixed_nominees_no_model_selection",
    }
    lock_json(output / "request.json", request)
    if (output / "summary.json").exists():
        integrity_path = output / "analysis_integrity.json"
        if not integrity_path.is_file() or read_json(integrity_path) != {
            "summary_sha256": file_sha256(output / "summary.json"),
            "request_sha256": file_sha256(output / "request.json"),
        }:
            raise RuntimeError(
                "Completed statistical summary integrity receipt is missing or changed"
            )
        summary = read_json(output / "summary.json")
        if (
            summary.get("status") != STATUS
            or summary.get("selection_lock_sha256") != inputs["selection_lock_sha256"]
            or summary.get("prediction_manifest_sha256") != inputs["prediction_manifest_sha256"]
            or not summary.get("artifacts")
        ):
            raise RuntimeError("Existing analysis summary does not match this locked request")
        for relative, digest in summary["artifacts"].items():
            if file_sha256(contained_path(output, relative)) != digest:
                raise RuntimeError("Completed statistical report artifact changed")
        return summary
    summary = analyze_inputs(inputs)
    # Recheck gate, source, and probability bytes after computation, before completion.
    checked = load_locked_inputs(selection_lock, prediction_manifest)
    if (
        checked["prediction_manifest_sha256"] != inputs["prediction_manifest_sha256"]
        or checked["selection_lock_sha256"] != inputs["selection_lock_sha256"]
    ):
        raise RuntimeError("Locked evaluation inputs changed during analysis")
    artifacts = render_reports(summary, output)
    summary["artifacts"] = {
        name: file_sha256(output / name) for name in ("request.json", *artifacts)
    }
    lock_json(output / "summary.json", summary)
    lock_json(
        output / "analysis_integrity.json",
        {
            "summary_sha256": file_sha256(output / "summary.json"),
            "request_sha256": file_sha256(output / "request.json"),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--prediction-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run_analysis(args.selection_lock, args.prediction_manifest, args.output_dir)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "comparisons": len(summary["comparisons"]),
                "nominee": NOMINEE,
                "automatic_model_promotion": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
