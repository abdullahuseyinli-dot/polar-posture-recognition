"""Immutable final-evaluation contracts, fit barrier, and label-free test access.

No model selection is implemented here. A test gate can only be created after
every prespecified train+validation refit has completed with verified artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from hac.polar import sha256_file
from hac.polar_benchmark import canonical_hash, lock_json, utc_now

LOCK_STATUS = "POLAR_FINAL_EVALUATION_LOCKED"
SEEDS = (42, 52, 62)
MODELS = ("dinov2_base", "siglip2_base", "dinov3_base", "convnextv2_base")
INFERENCE_COLUMNS = (
    "image_id",
    "image_path",
    "split",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "image_sha256",
)


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def inside(root: Path, relative: str) -> Path:
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Artifact path escapes its run: {relative}")
    return result


def verify_file(path: Path, digest: str) -> None:
    if not path.is_file() or sha256_file(path) != digest:
        raise RuntimeError(f"Artifact is missing or changed: {path}")


def verify_selection_lock(path: Path) -> dict:
    """Validate implementation/development bindings without parsing test files."""
    path = path.resolve(strict=True)
    value = read_json(path)
    if value.get("status") != LOCK_STATUS or Path(value["output_root"]).resolve() != path.parent:
        raise ValueError("Invalid final-selection lock or output root")
    root = Path(value["repository_root"]).resolve(strict=True)
    if not value.get("implementation"):
        raise ValueError("Missing implementation bindings")
    for relative, digest in value["implementation"].items():
        verify_file(inside(root, relative), digest)
    if set(value["tasks"]) != {"polar9", "polar4"}:
        raise ValueError("Both prespecified tasks are required")
    for task, specification in value["tasks"].items():
        if specification["classes"] != int(task[5:]):
            raise ValueError("Task/class count mismatch")
        verify_file(
            Path(specification["development_manifest"]),
            specification["development_manifest_sha256"],
        )
        if specification["nominated_candidate"] != "conservative_fusion":
            raise ValueError("The development-selected candidate cannot change")
    for binding in value.get("development_evidence", []):
        verify_file(Path(binding["path"]), binding["sha256"])
    return value


def expected_fits(lock: dict):
    for task, spec in lock["tasks"].items():
        for model, fit in spec["neural_fits"].items():
            for seed in fit["seeds"]:
                yield (
                    f"{task}_neural_{model}_seed{seed}",
                    fit["output_dir_pattern"].format(seed=seed),
                    "neural",
                )
        for model, fit in spec["head_fits"].items():
            yield f"{task}_head_{model}", fit["output_dir"], "head"


def fit_expectation(lock: dict, relative: str, kind: str) -> dict:
    parts = Path(relative).parts
    task, model = parts[0], parts[2]
    spec = lock["tasks"][task]
    expected = {
        "task": task,
        "model_kind": model,
        "classes": spec["classes"],
        "train_rows": spec["development_rows"],
        "development_manifest_sha256": spec["development_manifest_sha256"],
    }
    if kind == "neural":
        expected.update(
            seed=int(parts[3].removeprefix("seed")), epochs=spec["neural_fits"][model]["epochs"]
        )
    return expected


def verify_fit(directory: Path, selection_lock_sha256: str, kind: str, expected=None) -> dict:
    summary_path = directory / "summary.json"
    request_path = directory / "request.json"
    summary, request = read_json(summary_path), read_json(request_path)
    if (
        summary.get("status") != "COMPLETE"
        or summary.get("role") != "final_train_plus_validation_fit"
        or request.get("smoke") is not False
    ):
        raise RuntimeError(f"Incomplete or smoke fit cannot unlock test: {directory}")
    if request.get("selection_lock_sha256") != selection_lock_sha256:
        raise RuntimeError(f"Fit is not bound to this selection lock: {directory}")
    if summary.get("request_sha256") != canonical_hash(request):
        raise RuntimeError(f"Fit request has changed: {directory}")
    if any(
        item.get("test_rows_read") != 0 or item.get("test_labels_read") is not False
        for item in (summary, request)
    ):
        raise RuntimeError("Final fitting must never open test rows")
    if expected and any(request.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"Final fit identity/budget differs from locked component: {directory}")
    if kind == "neural" and summary.get("epochs_completed") != request.get("epochs"):
        raise RuntimeError("Final checkpoint did not finish its fixed epoch budget")
    artifacts = summary.get("artifacts", {})
    required = {"replay_audit.json", "final.pt" if kind == "neural" else "head.joblib"}
    if not required <= set(artifacts):
        raise RuntimeError(f"Missing final fit/replay artifacts: {directory}")
    for relative, digest in artifacts.items():
        verify_file(inside(directory, relative), digest)
    replay = read_json(directory / "replay_audit.json")
    if replay.get("status") != "PASS" or replay.get("predictions_identical") is not True:
        raise RuntimeError(f"Final checkpoint replay failed: {directory}")
    return {
        "directory": str(directory),
        "kind": kind,
        "summary_sha256": sha256_file(summary_path),
        "request_sha256": sha256_file(request_path),
        "artifacts": artifacts,
    }


def verify_all_fits(selection_lock: Path) -> dict:
    lock = verify_selection_lock(selection_lock)
    digest = sha256_file(selection_lock)
    run = selection_lock.resolve().parent
    records = {}
    for identifier, relative, kind in expected_fits(lock):
        records[identifier] = verify_fit(
            inside(run, relative), digest, kind, fit_expectation(lock, relative, kind)
        )
    if len(records) != 23:
        raise ValueError("Exactly fifteen neural and eight frozen-head refits are required")
    value = {
        "status": "ALL_23_FINAL_FITS_VERIFIED",
        "selection_lock_sha256": digest,
        "fits": records,
        "test_predictions_opened": False,
    }
    lock_json(run / "final_fits_verified.json", value)
    return value


def validate_test_rows(frame: pd.DataFrame, spec: dict, development: pd.DataFrame) -> pd.DataFrame:
    required = set(INFERENCE_COLUMNS) | {"label", "label_index", "source_group"}
    if "image_sha256" not in frame and "sha256" in frame:
        frame = frame.rename(columns={"sha256": "image_sha256"})
    if not required <= set(frame) or frame[list(required)].isna().any().any():
        raise ValueError("Test manifest fields missing")
    if set(frame.split) != {"test"} or len(frame) != spec["test_rows"]:
        raise ValueError("Test partition or locked row count changed")
    if frame.image_id.duplicated().any() or set(frame.label_index) != set(range(spec["classes"])):
        raise ValueError("Test IDs or class indices invalid")
    expected_names = dict(enumerate(spec["class_names"]))
    if not all(expected_names.get(int(row.label_index)) == row.label for row in frame.itertuples()):
        raise ValueError("Test class mapping differs from development")
    if set(frame.image_id) & set(development.image_id) or set(frame.source_group) & set(
        development.source_group
    ):
        raise ValueError("Detected source identity crosses development and test")
    dev_hash = "image_sha256" if "image_sha256" in development else "sha256"
    if set(frame.image_sha256) & set(development[dev_hash]):
        raise ValueError("Exact image duplicate crosses development and test")
    return frame.sort_values("image_id", ignore_index=True)


def open_test_gate(selection_lock: Path) -> dict:
    lock = verify_selection_lock(selection_lock)
    run = selection_lock.resolve().parent
    if (run / "test_access_gate.json").exists():
        return verify_test_access(selection_lock, run)
    verify_all_fits(selection_lock)
    frames = {}
    for task, spec in lock["tasks"].items():
        manifest = Path(spec["test_manifest"])
        verify_file(manifest, spec["test_manifest_sha256"])
        development = pd.read_csv(
            spec["development_manifest"], dtype={"image_id": str, "source_group": str}
        )
        frames[task] = validate_test_rows(
            pd.read_csv(manifest, dtype={"image_id": str, "source_group": str}),
            spec,
            development,
        )
    nine, four = frames["polar9"].set_index("image_id"), frames["polar4"].set_index("image_id")
    if not set(four.index) <= set(nine.index):
        raise ValueError("Four-class cohort must be a subset of the nine-class test cohort")
    for column in ("label_index", "image_sha256", "source_group", *INFERENCE_COLUMNS[3:7]):
        if not np.array_equal(nine.loc[four.index, column], four[column]):
            raise ValueError(f"Shared task cohort disagrees in {column}")
    inference_path = run / "inference_manifest.csv"
    inference = frames["polar9"][list(INFERENCE_COLUMNS)]
    csv_text = inference.to_csv(index=False, lineterminator="\n")
    if inference_path.exists():
        if inference_path.read_text(encoding="utf-8") != csv_text:
            raise RuntimeError("Existing inference manifest differs")
    else:
        with inference_path.open("x", encoding="utf-8", newline="") as handle:
            handle.write(csv_text)
    gate = {
        "status": "FINAL_TEST_ACCESS_OPENED",
        "opened_utc": utc_now(),
        "selection_lock_sha256": sha256_file(selection_lock),
        "barrier_sha256": sha256_file(run / "final_fits_verified.json"),
        "inference_manifest": str(inference_path),
        "inference_manifest_sha256": sha256_file(inference_path),
        "inference_rows": len(inference),
        "test_used_for_selection": False,
        "official_open_count_this_locked_phase": 1,
        "historical_exposure": lock["historical_exposure"],
    }
    lock_json(run / "test_access_gate.json", gate)
    return gate


def verify_test_access(selection_lock: Path, run_dir: Path) -> dict:
    lock = verify_selection_lock(selection_lock)
    run = run_dir.resolve(strict=True)
    if run != Path(lock["output_root"]).resolve():
        raise ValueError("Test run does not match selection lock")
    gate = read_json(run / "test_access_gate.json")
    barrier = read_json(run / "final_fits_verified.json")
    digest = sha256_file(selection_lock)
    if (
        gate.get("status") != "FINAL_TEST_ACCESS_OPENED"
        or gate.get("selection_lock_sha256") != digest
        or gate.get("barrier_sha256") != sha256_file(run / "final_fits_verified.json")
        or barrier.get("status") != "ALL_23_FINAL_FITS_VERIFIED"
        or barrier.get("selection_lock_sha256") != digest
        or len(barrier.get("fits", {})) != 23
    ):
        raise RuntimeError("Test gate/barrier does not match the locked final fits")
    # Recheck summary+checkpoint bindings, not merely existence of a gate token.
    for identifier, relative, kind in expected_fits(lock):
        if verify_fit(
            inside(run, relative), digest, kind, fit_expectation(lock, relative, kind)
        ) != barrier["fits"].get(identifier):
            raise RuntimeError("Final fit changed after barrier")
    verify_file(Path(gate["inference_manifest"]), gate["inference_manifest_sha256"])
    for spec in lock["tasks"].values():
        verify_file(Path(spec["test_manifest"]), spec["test_manifest_sha256"])
    return gate


def normalized(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if (
        values.ndim != 2
        or not np.isfinite(values).all()
        or (values < 0).any()
        or (values.sum(1) <= 0).any()
    ):
        raise ValueError("Invalid probability array")
    if not np.allclose(values.sum(1), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Unnormalized probabilities")
    return values / values.sum(1, keepdims=True)


def fusion_candidates(components: dict[str, np.ndarray], *, classes: int) -> dict:
    """Fixed formulas only; this function deliberately has no labels argument."""
    anchor = components["adapted_dinov2" if classes == 9 else "frozen_dinov2_base"]
    frozen, adapted = components["frozen_siglip2_base"], components["adapted_siglip2"]
    return {
        **components,
        "prior_incumbent": normalized(0.5 * anchor + 0.5 * frozen),
        "replacement_fusion": normalized(0.5 * anchor + 0.5 * adapted),
        "conservative_fusion": normalized(0.5 * anchor + 0.25 * frozen + 0.25 * adapted),
    }
