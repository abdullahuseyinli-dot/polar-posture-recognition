"""Verify the standalone benchmark using only Python's standard library."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import tomllib
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
CURRENT_DOCS = (
    "README.md", "CONTRIBUTING.md", "docs/README.md", "docs/RESULTS.md",
    "docs/ARCHITECTURE.md", "docs/REPRESENTATIONS.md", "docs/MODEL_CARD.md",
    "docs/REPRODUCIBILITY.md", "docs/PROJECT_HISTORY.md", "docs/VALIDATION.md",
    "assets/README.md", "results/README.md", "data/README.md",
    "experiments/README.md", "output/pdf/README.md",
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
    return {"rows": total, "accuracy": correct / total, "macro_f1": sum(f1) / n,
            "errors": total - correct}


def close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Metric arithmetic differs: {name}")


def evidence(root: Path) -> dict:
    checked = 0
    for table, matrices, key, population in (
        ("polar_test_metrics.csv", "polar_test_confusions.json", "candidate", 3329),
        ("vcoco_v2/official_test_metrics.csv", "vcoco_v2/official_test_confusions.json", "method", 6077),
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
    close(float(scores["scale_conditioned_stacking"]["macro_f1"])
          - float(scores["historical_v1_dino"]["macro_f1"]),
          delta["point_estimate"], "V-COCO paired gain")
    decisions = read_json(root / "results/vcoco_v3/source_tag_promotion_decisions.json")
    if decisions["representations"]["decisions"]["dinov3_base"]["general_candidate"]:
        raise ValueError("DINOv3 promotion claim changed")
    return {"confusion_based_systems_recomputed": checked,
            "polar_test_rows": 3329, "vcoco_test_people": 6077,
            "scope": "Proper scores and intervals are preserved exports, not recomputed"}


def metadata(root: Path) -> str:
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    cff = (root / "CITATION.cff").read_text(encoding="utf-8")
    version = re.search(r'^version:\s*[\'"]?([^\'"\s]+)', cff, re.MULTILINE)
    if project["name"] != "polar-posture-recognition" or not version:
        raise ValueError("Project identity differs")
    if project["version"] != version.group(1) or read_json(root / ".zenodo.json")["version"] != project["version"]:
        raise ValueError("Version metadata disagrees")
    return project["version"]


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
    print(json.dumps({
        "status": "PASS", "version": metadata(ROOT),
        "preserved_historical_files": preserved_files(ROOT),
        "evidence": evidence(ROOT), "current_links_checked": navigation(ROOT),
        "current_figure_files": figures(ROOT),
        "checkpoint_replay": False,
    }, indent=2))


if __name__ == "__main__":
    main()
