"""Verify the standalone benchmark using only Python's standard library."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tomllib
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    "README.md",
    "CONTRIBUTING.md",
    "docs/README.md",
    "docs/RESULTS.md",
    "docs/ARCHITECTURE.md",
    "docs/REPRESENTATIONS.md",
    "docs/MODEL_CARD.md",
    "docs/REPRODUCIBILITY.md",
    "docs/PROJECT_HISTORY.md",
    "docs/VALIDATION.md",
    "assets/README.md",
    "results/README.md",
    "data/README.md",
    "experiments/README.md",
    "output/pdf/README.md",
    "docs/POLAR_BENCHMARK_REPORT.md",
    "docs/COMPARISONS.md",
    "docs/PORTFOLIO_ARTICLE.md",
    "docs/releases/PROJECT_1.1.0.md",
    "results/polar_20260921/README.md",
    "THIRD_PARTY_NOTICES.md",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path, normalized: bool = True) -> str:
    data = path.read_bytes()
    if normalized:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def within(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes the repository: {name}")
    return path


def preserved_files(root: Path) -> int:
    origin = read_json(root / "results/project_origin.json")
    for name, record in origin["frozen_files"].items():
        if digest(within(root, name), record["normalized_lf"]) != record["sha256"]:
            raise ValueError(f"Frozen historical source/evidence changed: {name}")
    for name in origin["excluded_third_party_media"]:
        if within(root, name).exists():
            raise ValueError(f"Excluded third-party photograph is present: {name}")
    return len(origin["frozen_files"])


def confusion_metrics(matrix: list[list[int]]) -> dict:
    n = len(matrix)
    if n < 2 or any(len(row) != n for row in matrix):
        raise ValueError("Expected a square multiclass confusion matrix")
    if any(type(value) is not int or value < 0 for row in matrix for value in row):
        raise ValueError("Confusion counts must be nonnegative integers")
    total = sum(map(sum, matrix))
    if total == 0:
        raise ValueError("Empty evaluation population")
    f1 = []
    for i in range(n):
        denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1.append(2 * matrix[i][i] / denominator if denominator else 0.0)
    correct = sum(matrix[i][i] for i in range(n))
    return {
        "rows": total,
        "accuracy": correct / total,
        "macro_f1": sum(f1) / n,
        "errors": total - correct,
    }


def close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Metric arithmetic differs: {name}")


def evidence(root: Path) -> dict:
    checked = 0
    for table, matrices, key, population in (
        ("polar_test_metrics.csv", "polar_test_confusions.json", "candidate", 3329),
        (
            "vcoco_v2/official_test_metrics.csv",
            "vcoco_v2/official_test_confusions.json",
            "method",
            6077,
        ),
    ):
        confusions = read_json(root / "results" / matrices)
        with (root / "results" / table).open(encoding="utf-8", newline="") as stream:
            scores = {row[key]: row for row in csv.DictReader(stream)}
        for name, row in scores.items():
            actual = confusion_metrics(confusions[name]["counts"])
            if actual["rows"] != population:
                raise ValueError(f"Evaluation population differs: {name}")
            for metric in ("macro_f1", "accuracy"):
                close(actual[metric], float(row[metric]), name + ":" + metric)
            checked += 1
    uncertainty = read_json(root / "results/polar_test_uncertainty.json")
    primary = uncertainty["locked_ensemble"]["point_estimate"]
    for name, record in uncertainty["locked_ensemble_paired_deltas"].items():
        close(primary - uncertainty[name]["point_estimate"], record["point_estimate"], name)
    summary = read_json(root / "results/polar_test_summary.json")
    gate = read_json(root / "results/polar_test_access_gate.json")
    if summary["test_used_for_selection"] or gate["official_test_manifest_open_count"] != 1:
        raise ValueError("POLAR test gate differs")
    for name in ("selection_lock_sha256", "test_manifest_sha256", "test_rows_read"):
        if gate[name] != summary[name]:
            raise ValueError(f"POLAR gate/summary mismatch: {name}")
    vcoco_gate = read_json(root / "results/vcoco_v2/test_access_gate.json")
    if vcoco_gate["official_test_label_open_count"] != 1 or vcoco_gate["test_rows_read"] != 6077:
        raise ValueError("V-COCO test gate differs")
    delta = read_json(root / "results/vcoco_v2/official_test_uncertainty.json")
    close(
        float(scores["scale_conditioned_stacking"]["macro_f1"])
        - float(scores["historical_v1_dino"]["macro_f1"]),
        delta["point_estimate"],
        "V-COCO paired gain",
    )
    decisions = read_json(root / "results/vcoco_v3/source_tag_promotion_decisions.json")
    if decisions["representations"]["decisions"]["dinov3_base"]["general_candidate"]:
        raise ValueError("DINOv3 promotion claim changed")
    return {
        "confusion_based_systems_recomputed": checked,
        "polar_test_rows": 3329,
        "vcoco_test_people": 6077,
        "scope": "Proper scores and intervals are preserved exports, not recomputed",
    }


def metadata(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    version = re.search(r'^version:\s*[\'"]?([^\'"\s]+)', cff, re.MULTILINE)
    if project["name"] != "polar-posture-recognition" or not version:
        raise ValueError("Project identity differs")
    if (
        project["version"] != version.group(1)
        or read_json(root / ".zenodo.json")["version"] != project["version"]
    ):
        raise ValueError("Version metadata disagrees")
    return project["version"]


def benchmark_evidence(root: Path) -> dict:
    """Check the portable release without importing numerical or GPU packages."""
    folder = root / "results/polar_20260921"
    manifest = read_json(folder / "manifest.json")
    summary_path = within(root, manifest["summary_path"])
    expected_hash = "5beab5852a8aea3375a2b7425308f4c89974acc2f135c1411d6994a4e52b813a"
    if manifest["summary_sha256"] != expected_hash or digest(summary_path, False) != expected_hash:
        raise ValueError("Locked final summary hash differs")
    if manifest["schema_version"] != 1:
        raise ValueError("Public evidence schema version differs")
    for name in manifest["artifacts"]:
        within(folder, name)
    required_artifacts = {
        "cohort.csv",
        "data_audit.json",
        "polar4_predictions.npz",
        "polar9_predictions.npz",
        "quarantine.csv",
    }
    if set(manifest["artifacts"]) != required_artifacts:
        raise ValueError("Public evidence artifact inventory differs")
    if manifest["project_version"] != metadata(root):
        raise ValueError("Evidence release version differs")
    for name, item in manifest["artifacts"].items():
        path = within(folder, name)
        if digest(path, False) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"Public evidence bytes differ: {name}")
    summary = read_json(summary_path)
    if summary["selection_lock_sha256"] != manifest["selection_lock_sha256"]:
        raise ValueError("Selection-lock binding differs")
    if summary["prediction_manifest_sha256"] != manifest["source_prediction_manifest_sha256"]:
        raise ValueError("Original prediction-manifest binding differs")
    if (
        summary["test_used_for_model_or_threshold_selection"]
        or summary["automatic_model_promotion"]
    ):
        raise ValueError("Post-test selection or promotion is not permitted")
    tasks = {"polar4", "polar9"}
    if set(manifest["tasks"]) != tasks or set(summary["tasks"]) != tasks:
        raise ValueError("Locked task inventory differs")
    common_candidates = {
        "adapted_convnextv2",
        "adapted_siglip2",
        "conservative_fusion",
        "frozen_convnextv2_base",
        "frozen_dinov2_base",
        "frozen_dinov3_base",
        "frozen_siglip2_base",
        "prior_incumbent",
        "replacement_fusion",
    }
    candidates = {
        "polar4": common_candidates | {"historical_ensemble"},
        "polar9": common_candidates | {"adapted_dinov2"},
    }
    expected_comparisons = {
        (f"{task}__vs__{reference}", task, reference, "conservative_fusion")
        for task in tasks
        for reference in candidates[task] - {"conservative_fusion"}
    }
    observed_comparisons = [
        (record["id"], record["task"], record["reference"], record["candidate"])
        for record in summary["comparisons"]
    ]
    if (
        len(observed_comparisons) != len(expected_comparisons)
        or set(observed_comparisons) != expected_comparisons
    ):
        raise ValueError("Locked comparison inventory differs")
    class_names = [
        "sitting",
        "standing",
        "walking",
        "running",
        "bending",
        "jumping",
        "lying",
        "squatting",
        "stretching",
    ]
    seeds = [42, 52, 62]
    seed_candidates = {f"{role}_seed{seed}" for role in ("prior", "conservative") for seed in seeds}
    checked = 0
    prediction_sets = 0
    for task, rows in (("polar4", 3329), ("polar9", 6984)):
        record = summary["tasks"][task]
        public_task = manifest["tasks"][task]
        if record["rows"] != rows or set(record["metrics"]) != candidates[task]:
            raise ValueError("Final evaluation population or panel differs")
        expected_classes = class_names[:4] if task == "polar4" else class_names
        if public_task["rows"] != rows or public_task["class_names"] != expected_classes:
            raise ValueError("Public evaluation population or class order differs")
        if [item["seed"] for item in record["seed_diagnostic"]["seeds"]] != seeds:
            raise ValueError("Locked seed inventory differs")
        if (
            record["nominated_candidate"] != "conservative_fusion"
            or not record["nominee_unchanged_after_test"]
        ):
            raise ValueError("Development nominee changed")
        if record["promotion_gate"]["all_checks_passed"]:
            raise ValueError("Locked prior retention differs")
        if set(public_task["candidates"]) != candidates[task] | seed_candidates:
            raise ValueError("Public prediction inventory differs")
        prediction_sets += len(public_task["candidates"])
        for name, expected in record["metrics"].items():
            if expected["class_names"] != expected_classes:
                raise ValueError("Final metric class order differs")
            actual = confusion_metrics(expected["confusion_matrix"])
            for key in ("rows", "errors", "macro_f1", "accuracy"):
                close(actual[key], expected[key], f"{task}.{name}.{key}")
            if actual["rows"] != rows:
                raise ValueError("Confusion population differs")
            checked += 1
    audit = read_json(folder / "data_audit.json")
    with (folder / "cohort.csv").open(encoding="utf-8", newline="") as stream:
        cohort = list(csv.DictReader(stream))
    with (folder / "quarantine.csv").open(encoding="utf-8", newline="") as stream:
        quarantine = list(csv.DictReader(stream))
    if len(cohort) != 35007 or len(quarantine) != 317:
        raise ValueError("Audited cohort/quarantine size differs")
    ids = [row["image_id"] for row in cohort + quarantine]
    if len(set(ids)) != 35324:
        raise ValueError("Cohort/quarantine IDs overlap or repeat")
    counts = Counter((row["split"], row["label"]) for row in cohort)
    expected_counts = Counter(
        {
            (split, label): count
            for split, values in audit["source_audited_counts"].items()
            for label, count in values.items()
        }
    )
    if counts != expected_counts:
        raise ValueError("Audited class/split counts differ")
    groups = {}
    for row in cohort:
        if int(row["label_index"]) != audit["class_to_index"][row["label"]]:
            raise ValueError("Audited class mapping differs")
        group = row["source_group"]
        if groups.setdefault(group, row["split"]) != row["split"]:
            raise ValueError("Detected source group crosses retained splits")
    return {
        "confusion_based_systems_recomputed": checked,
        "prediction_sets_hashed": prediction_sets,
        "audited_rows": len(cohort),
        "quarantined_rows": len(quarantine),
        "retention": "prior_incumbents_unchanged",
    }


def release_artifacts(root: Path) -> int:
    """Check current charts and report separately from imported historical assets."""
    count = 0
    for name in (
        "assets/polar_20260921/figure_manifest.json",
        "output/pdf/polar_benchmark_report_v1.1.0.manifest.json",
    ):
        manifest = read_json(root / name)
        if manifest["project_version"] != metadata(root):
            raise ValueError("Presentation release version differs")
        for section in ("sources", "artifacts"):
            for path, item in manifest.get(section, {}).items():
                if digest(within(root, path), item["normalized_lf"]) != item["sha256"]:
                    raise ValueError(f"Release source/artifact differs: {path}")
                count += section == "artifacts"
    if count != 14:
        raise ValueError("Current release artifact inventory differs")
    return count


def navigation(root: Path) -> int:
    checked = 0
    for name in CURRENT_DOCS:
        path = root / name
        content = path.read_text(encoding="utf-8")
        if re.search(r"[A-Za-z]:[\\/]Users[\\/]|/home/[^/]+/", content):
            raise ValueError(f"Workstation-local path in current guide: {name}")
        for target in re.findall(r"\]\(([^\s)]+)(?:\s+[^)]*)?\)", content):
            parsed = urlsplit(target.strip("<>"))
            if parsed.scheme or not parsed.path:
                continue
            resolved = (path.parent / unquote(parsed.path)).resolve()
            if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                raise ValueError(f"Broken current link: {name} -> {target}")
            checked += 1
    return checked


def figures(root: Path) -> int:
    manifest = read_json(root / "assets/project_figure_manifest.json")
    for group, folder in (("sources", root), ("artifacts", root / "assets")):
        for name, item in manifest[group].items():
            if digest(within(folder, name), Path(name).suffix != ".png") != item["sha256"]:
                raise ValueError(f"Figure source/artifact differs: {name}")
    if len(manifest["artifacts"]) != 6:
        raise ValueError("Current figure inventory differs")
    return 6


def main() -> None:
    print(
        json.dumps(
            {
                "status": "PASS",
                "version": metadata(ROOT),
                "preserved_historical_files": preserved_files(ROOT),
                "evidence": evidence(ROOT),
                "current_links_checked": navigation(ROOT),
                "current_figure_files": figures(ROOT),
                "current_benchmark": benchmark_evidence(ROOT),
                "release_artifact_bindings": release_artifacts(ROOT),
                "checkpoint_replay": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
