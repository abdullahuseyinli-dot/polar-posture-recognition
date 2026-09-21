"""Produce only the frozen final candidate/seed panel after the all-fits barrier."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from hac.polar import image_view, sha256_file
from hac.polar_benchmark import atomic_json, canonical_hash, lock_json, probability_metrics
from hac.polar_locked_evaluation import (
    INFERENCE_COLUMNS,
    SEEDS,
    fusion_candidates,
    normalized,
    read_json,
    verify_file,
    verify_selection_lock,
    verify_test_access,
)
from hac.polar_locked_features import load_aligned_test_features, predict_locked_head
from hac.polar_locked_neural import load_final_model


class TestPixels(Dataset):
    """Inference uses audited pixels and the annotated box, never label metadata."""

    def __init__(self, frame, transform):
        if set(frame) != set(INFERENCE_COLUMNS):
            raise ValueError("Neural inference accepts the exact label-free input schema only")
        self.rows, self.transform = frame.to_dict("records"), transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        verify_file(Path(row["image_path"]), row["image_sha256"])
        with Image.open(row["image_path"]) as image:
            return self.transform(image_view(image.convert("RGB"), row, "person_context_25"))


def save_prediction(path: Path, ids, labels, probabilities, request: dict) -> dict:
    values = normalized(probabilities)
    request_hash = canonical_hash(request)
    marker = path.with_suffix(".json")
    if marker.exists():
        previous = read_json(marker)
        if previous["request_sha256"] != request_hash:
            raise RuntimeError("Existing prediction has a different locked request")
        verify_file(path, previous["sha256"])
        with np.load(path, allow_pickle=False) as saved:
            if (
                not np.array_equal(saved["image_ids"], ids)
                or not np.array_equal(saved["labels"], labels)
                or not np.array_equal(saved["probabilities"], values)
            ):
                raise RuntimeError("An existing test prediction would change")
        return previous
    if path.exists():
        raise RuntimeError("Orphan test predictions require integrity review, not blind overwrite")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, image_ids=ids, labels=labels, probabilities=values)
    value = {
        "path": str(path),
        "sha256": sha256_file(path),
        "request_sha256": request_hash,
        "request": request,
    }
    lock_json(marker, value)
    return value


def cached_prediction(path: Path, ids, labels, request: dict):
    marker = path.with_suffix(".json")
    if not marker.exists():
        return None
    meta = read_json(marker)
    if meta["request_sha256"] != canonical_hash(request):
        raise RuntimeError("Cached prediction request changed")
    verify_file(path, meta["sha256"])
    with np.load(path, allow_pickle=False) as saved:
        if not np.array_equal(saved["image_ids"], ids) or not np.array_equal(
            saved["labels"], labels
        ):
            raise RuntimeError("Cached prediction membership changed")
        values = saved["probabilities"].copy()
        normalized(values)  # Validate without repeated floating-point renormalization.
        return values


@torch.inference_mode()
def neural_probabilities(directory, inputs, lock_sha, output, progress):
    # RBF kernels deliberately set highest precision; reset the neural training
    # policy on every call so a cache hit cannot change inference numerics.
    torch.set_float32_matmul_precision("high")
    model, transform, _ = load_final_model(directory, expected_lock_sha256=lock_sha)
    loader = DataLoader(
        TestPixels(inputs, transform), batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )
    values = []
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Final inference requires the locked CUDA bfloat16 path")
    started = time.perf_counter()
    for index, pixels in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(pixels.to("cuda", non_blocking=True))
        values.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        if index % 25 == 0:
            atomic_json(
                output / "inference_progress.json",
                {
                    "status": "PREDICTING",
                    "component": progress,
                    "batch": index + 1,
                    "batches": len(loader),
                    "seconds": time.perf_counter() - started,
                },
            )
    result = normalized(np.concatenate(values))
    del model, loader, transform
    gc.collect()
    torch.cuda.empty_cache()
    return result


def historical_probabilities(spec: dict, ids, labels, names) -> np.ndarray:
    path = Path(spec["path"])
    verify_file(path, spec["sha256"])
    with np.load(path, allow_pickle=False) as archive:
        old_ids = archive["image_ids"].astype(str)
        if len(set(old_ids)) != len(ids) or set(old_ids) != set(ids):
            raise RuntimeError("Historical comparison is not the identical four-class cohort")
        position = {identifier: index for index, identifier in enumerate(old_ids)}
        order = [position[identifier] for identifier in ids]
        if not np.array_equal(archive[spec["label_key"]][order], labels) or not np.array_equal(
            archive["class_names_4"], names
        ):
            raise RuntimeError("Historical class mapping differs")
        values = normalized(archive[spec["probability_key"]][order])
    replay = probability_metrics(labels, values, names)
    if not np.isclose(replay["macro_f1"], spec["expected_macro_f1"], atol=1e-12, rtol=0):
        raise RuntimeError("Historical reference metrics could not be reproduced")
    return values


def evaluate(selection_lock: Path) -> dict:
    run = selection_lock.resolve().parent
    gate = verify_test_access(selection_lock, run)
    lock = verify_selection_lock(selection_lock)
    lock_sha = sha256_file(selection_lock)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this locked evaluation")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.benchmark = False
    output = run / "evaluation"
    output.mkdir(exist_ok=True)
    inference = pd.read_csv(gate["inference_manifest"], dtype={"image_id": str})
    inference = inference.set_index("image_id", drop=False)
    result = {
        "selection_lock_sha256": lock_sha,
        "gate_sha256": sha256_file(run / "test_access_gate.json"),
        "tasks": {},
    }
    for task, spec in lock["tasks"].items():
        print(f"Evaluating fixed {task} panel; no selection or fitting", flush=True)
        frame = pd.read_csv(
            spec["test_manifest"], dtype={"image_id": str, "source_group": str}
        ).sort_values("image_id", ignore_index=True)
        ids, labels = frame.image_id.to_numpy(dtype=str), frame.label_index.to_numpy(dtype=int)
        inputs = inference.loc[ids, list(INFERENCE_COLUMNS)].reset_index(drop=True)
        task_dir = output / task
        task_dir.mkdir(exist_ok=True)
        group_path = task_dir / "source_groups.json"
        lock_json(group_path, frame.source_group.astype(str).tolist())
        components, seeds = {}, {}
        costs = {"neural_training_seconds": 0.0, "head_fit_seconds": 0.0}
        started = time.perf_counter()

        def request(name, source, task_name=task, manifest_sha=spec["test_manifest_sha256"]):
            return {
                "selection_lock_sha256": lock_sha,
                "task": task_name,
                "candidate": name,
                "test_manifest_sha256": manifest_sha,
                "source": source,
            }

        for model, fit in spec["head_fits"].items():
            name = f"frozen_{model}"
            directory = run / fit["output_dir"]
            req = request(name, {"head_summary_sha256": sha256_file(directory / "summary.json")})
            path = task_dir / f"{name}.npz"
            probabilities = cached_prediction(path, ids, labels, req)
            if probabilities is None:
                features = load_aligned_test_features(
                    run / "test_features", model, ids, expected_lock_sha256=lock_sha
                )
                probabilities = predict_locked_head(
                    directory, features, expected_lock_sha256=lock_sha
                )
                save_prediction(path, ids, labels, probabilities, req)
                probabilities = cached_prediction(path, ids, labels, req)
                del features
            components[name] = probabilities
            costs["head_fit_seconds"] += read_json(directory / "summary.json")["fit_seconds"]
        for model, fit in spec["neural_fits"].items():
            alias = "adapted_" + model.removesuffix("_base")
            seed_values = {}
            for seed in SEEDS:
                name = f"{alias}_seed{seed}"
                directory = run / fit["output_dir_pattern"].format(seed=seed)
                req = request(
                    name, {"neural_summary_sha256": sha256_file(directory / "summary.json")}
                )
                path = task_dir / f"{name}.npz"
                probabilities = cached_prediction(path, ids, labels, req)
                if probabilities is None:
                    probabilities = neural_probabilities(
                        directory, inputs, lock_sha, output, f"{task}/{name}"
                    )
                    save_prediction(path, ids, labels, probabilities, req)
                    probabilities = cached_prediction(path, ids, labels, req)
                seed_values[seed] = probabilities
                costs["neural_training_seconds"] += float(
                    pd.read_csv(directory / "history.csv").epoch_seconds.sum()
                )
            components[alias] = normalized(np.mean(list(seed_values.values()), axis=0))
            seeds[alias] = seed_values
        if task == "polar4":
            components["historical_ensemble"] = historical_probabilities(
                lock["historical_reference"], ids, labels, spec["class_names"]
            )
        panel = fusion_candidates(components, classes=spec["classes"])
        if set(panel) != set(spec["candidates"]):
            raise RuntimeError("Final candidate panel differs from the prespecified comparisons")
        # Matching-seed sensitivity is diagnostic and does not change the nominee.
        for seed in SEEDS:
            anchor = (
                seeds["adapted_dinov2"][seed]
                if task == "polar9"
                else components["frozen_dinov2_base"]
            )
            panel[f"prior_seed{seed}"] = normalized(
                0.5 * anchor + 0.5 * components["frozen_siglip2_base"]
            )
            panel[f"conservative_seed{seed}"] = normalized(
                0.5 * anchor
                + 0.25 * components["frozen_siglip2_base"]
                + 0.25 * seeds["adapted_siglip2"][seed]
            )
        records = {}
        for name, probabilities in panel.items():
            path = task_dir / f"{name}.npz"
            if name.startswith("frozen_"):
                record = read_json(path.with_suffix(".json"))
            else:
                record = save_prediction(
                    path,
                    ids,
                    labels,
                    probabilities,
                    request(name, "locked_formula_and_verified_components"),
                )
            records[name] = {
                "path": str(path),
                "sha256": record["sha256"],
                "kind": "locked_comparison" if name in spec["candidates"] else "seed_diagnostic",
            }
        costs["inference_seconds"] = time.perf_counter() - started
        result["tasks"][task] = {
            "rows": len(frame),
            "manifest_sha256": spec["test_manifest_sha256"],
            "source_groups_path": str(group_path),
            "source_groups_sha256": sha256_file(group_path),
            "class_names": spec["class_names"],
            "candidates": records,
            "costs": costs,
        }
    # All probability artifacts are complete before any statistical claim is made.
    manifest_path = output / "prediction_manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        for task in result["tasks"]:
            result["tasks"][task]["costs"] = previous["tasks"][task]["costs"]
    result["cost_records"] = {task: spec["costs"] for task, spec in result["tasks"].items()}
    lock_json(manifest_path, result)
    lock_json(
        output / "summary.json",
        {
            "status": "LOCKED_PREDICTIONS_COMPLETE",
            "selection_lock_sha256": lock_sha,
            "prediction_manifest_sha256": sha256_file(manifest_path),
            "test_used_for_selection": False,
        },
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.selection_lock.resolve())


if __name__ == "__main__":
    main()
