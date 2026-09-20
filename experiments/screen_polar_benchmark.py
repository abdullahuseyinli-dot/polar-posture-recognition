"""Run the predeclared development-only frozen-feature challenger panel."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from hac.polar import sha256_file
from hac.polar_benchmark import (
    aligned_features,
    atomic_json,
    canonical_hash,
    check_completed,
    environment_evidence,
    implementation_evidence,
    label_names,
    load_development,
    lock_json,
    probability_metrics,
    utc_now,
)
from hac.polar_benchmark_heads import fit_group_calibrated_rbf
from hac.vcoco_v3_cuda_heads import CudaStandardizedLogisticRegression


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--model-kind", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", type=int, choices=(4, 9), required=True)
    parser.add_argument("--include-rbf", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This screen requires a working CUDA runtime")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    frame = load_development(args.manifest, num_classes=args.classes)
    names = label_names(frame)
    train = frame.split.eq("train").to_numpy()
    validation = frame.split.eq("val").to_numpy()
    labels = frame.label_index.to_numpy(dtype=int)
    views, evidence = {}, {}
    for view in ("full_frame", "person_context_10"):
        views[view], evidence[view] = aligned_features(
            args.cache_root / args.model_kind / view, frame
        )
    views["two_view_concat"] = np.concatenate(list(views.values()), axis=1)
    request = {
        "task_classes": args.classes,
        "class_names": names,
        "manifest_sha256": sha256_file(args.manifest),
        "model_kind": args.model_kind,
        "features": evidence,
        "include_rbf": args.include_rbf,
        "selection": "validation_macro_f1_then_nll_then_candidate_id",
        "test_rows_read": 0,
        "linear_C": [0.001, 0.01, 0.1, 1.0],
        "linear_class_weight": "balanced",
        "linear_max_iter": 500,
        "linear_tolerance": 1e-5,
        "rbf_C": 10.0,
        "rbf_gamma": "1 / feature_dimensions",
        "rbf_calibration": "5_fold_source_grouped_sigmoid",
        "seed": 42,
        "implementation": implementation_evidence(root),
        "environment": environment_evidence(),
    }
    if previous := check_completed(output, request):
        print(
            json.dumps(
                {
                    "status": "VERIFIED_REUSE",
                    "output": str(output),
                    "best": previous["best_candidate"],
                }
            ),
            flush=True,
        )
        return
    lock_json(output / "request.json", request)
    candidates = []
    configurations = [(view, "cuda_logistic", c) for view in views for c in (0.001, 0.01, 0.1, 1.0)]
    if args.include_rbf:
        configurations.append(("two_view_concat", "cuda_kernel_rbf", 10.0))
    for view, family, c_value in configurations:
        candidate = f"{view}__{family}__c{c_value:g}"
        candidate_dir = output / candidate
        candidate_request = {
            "parent_request_sha256": canonical_hash(request),
            "candidate": candidate,
        }
        if previous := check_completed(candidate_dir, candidate_request):
            candidates.append(
                {
                    "candidate": candidate,
                    "view": view,
                    "family": family,
                    "C": c_value,
                    **previous["metrics"],
                }
            )
            continue
        candidate_dir.mkdir(parents=True, exist_ok=True)
        lock_json(candidate_dir / "request.json", candidate_request)
        values = views[view]
        print(
            json.dumps(
                {
                    "event": "FIT_START",
                    "candidate": candidate,
                    "train_rows": int(train.sum()),
                    "dimensions": values.shape[1],
                    "utc": utc_now(),
                }
            ),
            flush=True,
        )
        started = time.perf_counter()
        if family == "cuda_logistic":
            model = CudaStandardizedLogisticRegression(
                c_value=c_value,
                class_weight="balanced",
                maximum_iterations=500,
                tolerance=1e-5,
                seed=42,
            )
            model.fit(values[train], labels[train])
            optimization = model.optimization_
            if (
                optimization["iteration_limit_reached"]
                and optimization["final_gradient_max"] > 1e-3
            ):
                raise RuntimeError(
                    "Logistic optimization did not converge adequately; stop rather than rank an underfit challenger"
                )
        else:
            model = fit_group_calibrated_rbf(
                values[train],
                labels[train],
                frame.loc[train, "source_group"].to_numpy(),
                seed=42,
                folds=5,
            )
            optimization = {
                "kernel": "CUDA_FP32_without_TF32",
                "optimizer": "CPU_libsvm",
                "calibration": "CPU_sigmoid",
            }
        fit_seconds = time.perf_counter() - started
        predicted = np.asarray(model.predict_proba(values[validation]), dtype=np.float64)
        if not np.array_equal(model.classes_, np.arange(args.classes)):
            raise RuntimeError("Unexpected estimator class order")
        predicted /= predicted.sum(axis=1, keepdims=True)
        metrics = probability_metrics(labels[validation], predicted, names)
        np.savez_compressed(
            candidate_dir / "validation_predictions.npz",
            image_ids=frame.loc[validation, "image_id"].to_numpy(dtype=str),
            labels=labels[validation],
            probabilities=predicted,
        )
        joblib.dump(model, candidate_dir / "head.joblib", compress=1)
        artifacts = {
            name: sha256_file(candidate_dir / name)
            for name in ("head.joblib", "validation_predictions.npz")
        }
        summary = {
            "status": "COMPLETE",
            "request_sha256": canonical_hash(candidate_request),
            "metrics": metrics,
            "fit_seconds": fit_seconds,
            "optimization": optimization,
            "artifacts": artifacts,
            "test_rows_read": 0,
            "completed_utc": utc_now(),
        }
        atomic_json(candidate_dir / "summary.json", summary)
        candidates.append(
            {"candidate": candidate, "view": view, "family": family, "C": c_value, **metrics}
        )
        atomic_json(
            output / "progress.json",
            {
                "status": "RUNNING",
                "completed": len(candidates),
                "total": len(configurations),
                "last_candidate": candidate,
                "utc": utc_now(),
            },
        )
        print(
            json.dumps(
                {
                    "event": "FIT_COMPLETE",
                    "candidate": candidate,
                    "validation_macro_f1": metrics["macro_f1"],
                    "fit_seconds": fit_seconds,
                }
            ),
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()
    ranking = sorted(
        candidates, key=lambda row: (-row["macro_f1"], row["log_loss"], row["candidate"])
    )
    pd.DataFrame(
        [
            {key: value for key, value in item.items() if not isinstance(value, (dict, list))}
            for item in ranking
        ]
    ).to_csv(output / "validation_ranking.csv", index=False)
    artifacts = {"validation_ranking.csv": sha256_file(output / "validation_ranking.csv")}
    for candidate in candidates:
        for name in ("summary.json", "head.joblib", "validation_predictions.npz"):
            relative = f"{candidate['candidate']}/{name}"
            artifacts[relative] = sha256_file(output / relative)
    atomic_json(
        output / "summary.json",
        {
            "status": "COMPLETE",
            "request_sha256": canonical_hash(request),
            "best_candidate": ranking[0]["candidate"],
            "best_validation_metrics": {
                key: value for key, value in ranking[0].items() if isinstance(value, (float, int))
            },
            "artifacts": artifacts,
            "test_rows_read": 0,
            "evaluation_role": "development_selection_not_confirmatory",
            "completed_utc": utc_now(),
        },
    )


if __name__ == "__main__":
    main()
