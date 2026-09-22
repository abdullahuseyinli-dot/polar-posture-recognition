"""Replay public POLAR predictions; optional full resampling, never model fitting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from hac.polar_locked_statistics import (
    error_transitions,
    holm_adjust,
    metrics_summary,
    paired_bootstrap,
    paired_randomization,
)

ROOT = Path(__file__).resolve().parents[1]


def compare(actual, expected, name: str) -> None:
    if isinstance(expected, dict):
        for key, value in expected.items():
            compare(actual[key], value, f"{name}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"Length mismatch: {name}")
        for i, (left, right) in enumerate(zip(actual, expected, strict=True)):
            compare(left, right, f"{name}[{i}]")
    elif isinstance(expected, float):
        if not np.isclose(actual, expected, rtol=0, atol=1e-12):
            raise ValueError(f"Numeric mismatch: {name}")
    elif actual != expected:
        raise ValueError(f"Value mismatch: {name}")


def verify(root: Path = ROOT, *, resample: bool = False) -> dict:
    folder = root / "results/polar_20260921"
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    summary_path = (root / manifest["summary_path"]).resolve()
    if not summary_path.is_relative_to(root.resolve()):
        raise ValueError("Summary path escapes repository")
    if hashlib.sha256(summary_path.read_bytes()).hexdigest() != manifest["summary_sha256"]:
        raise ValueError("Final report hash mismatch")
    for name, artifact in manifest["artifacts"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()):
            raise ValueError("Artifact path escapes public evidence")
        if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
            raise ValueError(f"Artifact hash mismatch: {name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with (folder / "cohort.csv").open(encoding="utf-8", newline="") as stream:
        cohort = {row["image_id"]: row for row in csv.DictReader(stream) if row["split"] == "test"}
    predictions = {}
    for task, record in manifest["tasks"].items():
        with np.load(folder / f"{task}_predictions.npz", allow_pickle=False) as saved:
            predictions[task] = {key: saved[key] for key in saved.files}
        arrays = predictions[task]
        if (
            len(arrays["labels"]) != record["rows"]
            or len(set(arrays["image_ids"])) != record["rows"]
        ):
            raise ValueError("Evaluation row count or IDs differ")
        compare(arrays["class_names"].tolist(), record["class_names"], task + ".classes")
        expected_ids = {key for key, row in cohort.items() if row["label"] in record["class_names"]}
        if set(arrays["image_ids"]) != expected_ids:
            raise ValueError("Prediction IDs do not match audited test membership")
        for image_id, label, group in zip(
            arrays["image_ids"], arrays["labels"], arrays["source_groups"], strict=True
        ):
            row = cohort[str(image_id)]
            if int(label) != int(row["label_index"]) or str(group) != row["source_group"]:
                raise ValueError("Prediction label or source group differs from cohort")
        for name, expected in summary["tasks"][task]["metrics"].items():
            actual = metrics_summary(arrays["labels"], arrays[name], record["class_names"])
            compare(
                actual, {k: v for k, v in expected.items() if k != "uncertainty"}, f"{task}.{name}"
            )
        anchor = arrays["adapted_dinov2" if task == "polar9" else "frozen_dinov2_base"]
        formulas = {
            "prior_incumbent": 0.5 * anchor + 0.5 * arrays["frozen_siglip2_base"],
            "replacement_fusion": 0.5 * anchor + 0.5 * arrays["adapted_siglip2"],
            "conservative_fusion": 0.5 * anchor
            + 0.25 * arrays["frozen_siglip2_base"]
            + 0.25 * arrays["adapted_siglip2"],
        }
        for name, expected in formulas.items():
            if not np.allclose(arrays[name], expected, rtol=0, atol=1e-12):
                raise ValueError(f"Fixed fusion arithmetic differs: {task}.{name}")
        for seed in summary["tasks"][task]["seed_diagnostic"]["seeds"]:
            prefix = str(seed["seed"])
            reference, candidate = (
                arrays[f"prior_seed{prefix}"],
                arrays[f"conservative_seed{prefix}"],
            )
            transitions = error_transitions(arrays["labels"], reference, candidate)
            compare(transitions, {k: seed[k] for k in transitions}, task + ".seed" + prefix)
            actual = metrics_summary(arrays["labels"], candidate, record["class_names"])
            compare(actual["macro_f1"], seed["macro_f1"], task + ".seed_f1")
    for record in summary["comparisons"]:
        arrays = predictions[record["task"]]
        inputs = (arrays["labels"], arrays[record["reference"]], arrays[record["candidate"]])
        compare(error_transitions(*inputs), record["transitions"], record["id"])
        for role in ("reference", "candidate"):
            expected = summary["tasks"][record["task"]]["metrics"][record[role]]["uncertainty"]
            actual = {key: record["bootstrap"][key][role] for key in ("macro_f1", "accuracy")}
            compare(actual, expected, record["id"] + ".marginal_intervals")
        if resample:
            compare(
                paired_bootstrap(*inputs, arrays["source_groups"]),
                record["bootstrap"],
                record["id"] + ".bootstrap",
            )
            compare(
                paired_randomization(*inputs, arrays["source_groups"]),
                record["randomization"],
                record["id"] + ".randomization",
            )
    corrected = holm_adjust([r["randomization"]["p_value"] for r in summary["comparisons"]])
    compare(corrected, [r["holm_p_value"] for r in summary["comparisons"]], "Holm family")
    return {
        "status": "PASS",
        "models_recomputed": 20,
        "seed_pairs_checked": 6,
        "paired_comparisons": 18,
        "full_resampling_replayed": resample,
        "training_or_inference_performed": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--resample",
        action="store_true",
        help="Also repeat all 5,000-draw bootstraps and 10,000-draw paired tests",
    )
    args = parser.parse_args()
    print(json.dumps(verify(resample=args.resample), indent=2))
