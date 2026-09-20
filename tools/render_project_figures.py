"""Render standalone benchmark figures from preserved public results, without fitting."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "results/polar_test_metrics.csv",
    "results/polar_test_uncertainty.json",
    "results/vcoco_v2/official_test_metrics.csv",
    "results/vcoco_v2/official_test_uncertainty.json",
    "results/vcoco_v3/source_tag_development_metrics.csv",
    "tools/render_project_figures.py",
)
FIGURES = ("benchmark_results", "transfer_results", "representation_results")
NAMES = {
    "locked_ensemble": "Locked ensemble",
    "dinov2_base_multilayer_rbf": "DINOv2-B + RBF SVM",
    "dinov2_base_multilayer_logistic": "DINOv2-B + logistic",
    "dinov2_base_top4": "DINOv2-B · top 4 adapted",
    "dinov2_small_moderate": "DINOv2-S · fully adapted",
    "convnext_small_full": "ConvNeXt-S · fully adapted",
    "scale_conditioned_stacking": "Target-trained DINO stack",
    "historical_v1_dino": "Source-only DINO baseline",
    "dinov2_base": "DINOv2-B",
    "dinov3_base": "DINOv3-B",
    "siglip2_base": "SigLIP2-B",
    "dino_siglip_factorized_reliability_stack": "DINO + SigLIP · reliability",
    "dino_siglip_linear_svm_control": "DINO + SigLIP · linear SVM",
    "dino_factorized_probability_stack": "DINO · factorized stack",
    "dino_flat_probability_stack": "DINO · flat stack",
}


def digest(path: Path) -> str:
    data = path.read_bytes()
    if path.suffix != ".png":
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def rows(name: str) -> list[dict]:
    with (ROOT / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(fig, stem: str) -> None:
    import matplotlib.pyplot as plt

    folder = ROOT / "assets"
    fig.savefig(folder / f"{stem}.png", dpi=190, facecolor="white")
    path = folder / f"{stem}.svg"
    fig.savefig(path, facecolor="white", metadata={"Date": None})
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plt.close(fig)


def point_chart(axis, data: list[dict], key: str, limits: tuple, uncertainty=None) -> None:
    for index, row in enumerate(data):
        name = row[key]
        value = float(row["macro_f1"]) * 100
        color = "#087f75" if index == 0 else "#52657f"
        if uncertainty is not None:
            interval = uncertainty[name]
            errors = [[value - interval["ci_95_low"] * 100],
                      [interval["ci_95_high"] * 100 - value]]
            axis.errorbar(value, index, xerr=errors, fmt="o", capsize=4,
                          color=color, markersize=7, linewidth=2)
        else:
            axis.scatter(value, index, color=color, s=65, zorder=3)
        axis.text(1.01, index, f"{value:.2f}%", transform=axis.get_yaxis_transform(),
                  va="center", color=color, fontweight="bold", fontsize=10)
    axis.set_yticks(range(len(data)), [NAMES[r[key]] for r in data])
    axis.set_ylim(len(data) - 0.4, -0.6)
    axis.set_xlim(*limits)
    axis.set_xlabel("Macro-F1 (%) · zoomed point scale")
    axis.grid(axis="x", alpha=0.2)
    axis.tick_params(axis="y", length=0, pad=10)
    for spine in axis.spines.values():
        spine.set_visible(False)


def main() -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Imported result hashes are checked before constructing any chart.
    from check_project import preserved_files

    preserved_files(ROOT)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.labelcolor": "#334155", "text.color": "#172b44",
        "svg.hashsalt": "polar-posture-recognition-1.0.0",
    })
    data = rows(SOURCES[0])
    uncertainty = json.loads((ROOT / SOURCES[1]).read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(10.7, 5.6))
    fig.subplots_adjust(left=0.32, right=0.87, bottom=0.23, top=0.74)
    point_chart(axis, data, "candidate", (87.5, 95.5), uncertainty)
    fig.text(0.04, 0.92, "POLAR · locked four-class test", fontsize=20, weight="bold")
    fig.text(0.04, 0.855, "3,329 held-out images  |  Six predeclared systems", fontsize=11)
    fig.text(0.04, 0.085, "Ensemble versus strongest component: +1.25 pp  [95% paired CI: +0.65, +1.86]",
             fontsize=11, color="#087f75", weight="bold")
    fig.text(0.04, 0.035, "Whiskers: marginal class-stratified 95% bootstrap intervals. Custom subset; no standard leaderboard claim.",
             fontsize=9, color="#52657f")
    save(fig, FIGURES[0])

    data = rows(SOURCES[2])
    delta = json.loads((ROOT / SOURCES[3]).read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(10.7, 4.3))
    fig.subplots_adjust(left=0.32, right=0.87, bottom=0.32, top=0.67)
    point_chart(axis, data, "method", (65, 90))
    fig.text(0.04, 0.9, "V-COCO · person-level transfer follow-up", fontsize=19, weight="bold")
    fig.text(0.04, 0.81, "6,077 test people in 3,708 images  |  Custom three-class posture mapping", fontsize=10.5)
    fig.text(0.04, 0.12,
             f"Paired gain: +{delta['point_estimate'] * 100:.2f} pp  "
             f"[95% image-cluster CI: +{delta['ci_95_low'] * 100:.2f}, +{delta['ci_95_high'] * 100:.2f}]",
             fontsize=11, color="#087f75", weight="bold")
    fig.text(0.04, 0.055, "The new stack uses target-domain training; this is not a zero-shot or architecture-only gain.", fontsize=9)
    save(fig, FIGURES[1])

    all_rows = rows(SOURCES[4])
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1))
    fig.subplots_adjust(left=0.11, right=0.92, bottom=0.25, top=0.67, wspace=1.45)
    for axis, stage, limits, title in zip(
        axes,
        ("representations", "nested_stacks"),
        ((82.8, 84.6), (84.8, 87.5)),
        ("Matched representation screen", "Separate nested-fusion screen"),
        strict=True,
    ):
        data = [row for row in all_rows if row["stage"] == stage]
        point_chart(axis, data, "family", limits)
        axis.set_title(title, fontsize=11, pad=18, weight="bold")
    fig.text(0.04, 0.92, "Representation quality and complementary evidence", fontsize=20, weight="bold")
    fig.text(0.04, 0.845, "6,640 V-COCO development people  |  Nested image-grouped evaluation, not held-out test", fontsize=11)
    fig.text(0.04, 0.09, "DINOv3 did not displace DINOv2. The DINO + SigLIP reliability stack passed its development promotion rules.", fontsize=10)
    fig.text(0.04, 0.04, "Point estimates only; these panels are different comparisons. Neither updates the locked POLAR or V-COCO test score.", fontsize=9, color="#52657f")
    save(fig, FIGURES[2])
    payload = {
        "schema_version": 1,
        "matplotlib_version": mpl.__version__,
        "scope": "Public aggregates; no training or new performance evaluation",
        "sources": {name: {"sha256": digest(ROOT / name)} for name in SOURCES},
        "artifacts": {
            f"{stem}.{ext}": {"sha256": digest(ROOT / "assets" / f"{stem}.{ext}")}
            for stem in FIGURES for ext in ("png", "svg")
        },
    }
    (ROOT / "assets/project_figure_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps({"figures": len(FIGURES), "output_files": 6}))


if __name__ == "__main__":
    main()
