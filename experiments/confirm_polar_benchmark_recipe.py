"""Lock the better of two predeclared seed-42 recipes, then confirm seeds 52/62."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--adaptation-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", type=int, choices=(9,), required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    frame = load_development(args.manifest, num_classes=args.classes)
    validation = frame.loc[frame.split.eq("val")]
    candidates = []
    for recipe in ("historical_mild", "person_preserving"):
        directory = args.adaptation_root / f"{recipe}_seed42"
        request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        if (
            request["manifest_sha256"] != sha256_file(args.manifest)
            or request["role"] != "development_adaptation"
            or request["seed"] != 42
        ):
            raise RuntimeError("Recipe selection inputs are not matched development runs")
        if check_completed(directory, request) is None:
            raise RuntimeError("Both recipe screens must complete before seed confirmation")
        metrics = json.loads((directory / "best_metrics.json").read_text(encoding="utf-8"))
        candidates.append(
            {
                "recipe": recipe,
                "macro_f1": metrics["macro_f1"],
                "log_loss": metrics["log_loss"],
                "summary_sha256": sha256_file(directory / "summary.json"),
            }
        )
    candidates.sort(key=lambda row: (-row["macro_f1"], row["log_loss"], row["recipe"]))
    selected = candidates[0]["recipe"]
    request = {
        "selection": "two_fixed_seed42_recipes; macro_f1_then_nll_then_recipe_name",
        "manifest_sha256": sha256_file(args.manifest),
        "candidates": candidates,
        "selected_recipe": selected,
        "confirmation_seeds": [52, 62],
        "test_rows_read": 0,
    }
    if check_completed(output, request):
        return
    lock_json(output / "selection_lock.json", request)
    probabilities, seed_metrics, artifacts = [], [], {}
    for seed in (42, 52, 62):
        directory = args.adaptation_root / f"{selected}_seed{seed}"
        if seed != 42:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("train_polar_benchmark.py")),
                    "--manifest",
                    str(args.manifest),
                    "--output-dir",
                    str(directory),
                    "--classes",
                    str(args.classes),
                    "--recipe",
                    selected,
                    "--seed",
                    str(seed),
                    "--workers",
                    "4",
                ],
                check=True,
            )
        seed_request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
        if check_completed(directory, seed_request) is None:
            raise RuntimeError("Missing completed seed run")
        with np.load(directory / "validation_predictions.npz", allow_pickle=False) as archive:
            if not np.array_equal(
                archive["image_ids"], validation.image_id.to_numpy(dtype=str)
            ) or not np.array_equal(archive["labels"], validation.label_index.to_numpy()):
                raise RuntimeError("Seed prediction row/label mismatch")
            probabilities.append(archive["probabilities"].copy())
        seed_metrics.append(
            json.loads((directory / "best_metrics.json").read_text(encoding="utf-8"))["macro_f1"]
        )
        artifacts[str(seed)] = {
            "summary_sha256": sha256_file(directory / "summary.json"),
            "predictions_sha256": sha256_file(directory / "validation_predictions.npz"),
        }
    mean_probabilities = np.mean(probabilities, axis=0)
    mean_probabilities /= mean_probabilities.sum(axis=1, keepdims=True)
    np.savez_compressed(
        output / "validation_predictions.npz",
        image_ids=validation.image_id.to_numpy(dtype=str),
        labels=validation.label_index.to_numpy(),
        probabilities=mean_probabilities,
    )
    metrics = probability_metrics(
        validation.label_index.to_numpy(), mean_probabilities, label_names(frame)
    )
    atomic_json(
        output / "summary.json",
        {
            "status": "COMPLETE",
            "request_sha256": canonical_hash(request),
            "selected_recipe": selected,
            "metrics": metrics,
            "individual_seed_macro_f1": seed_metrics,
            "individual_seed_mean": float(np.mean(seed_metrics)),
            "individual_seed_sd": float(np.std(seed_metrics, ddof=1)),
            "seed_evidence": artifacts,
            "artifacts": {
                "validation_predictions.npz": sha256_file(output / "validation_predictions.npz"),
                "selection_lock.json": sha256_file(output / "selection_lock.json"),
            },
            "test_rows_read": 0,
            "completed_utc": utc_now(),
        },
    )


if __name__ == "__main__":
    main()
