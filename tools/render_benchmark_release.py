"""Render the four-/nine-class release figures from locked numerical evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "docs/research/20260921_final_evaluation/results/summary.json"
OUT = ROOT / "assets/polar_20260921"
NAVY, TEAL, GREY, ORANGE = "#23435c", "#007f78", "#718096", "#b56b25"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelcolor": NAVY,
        "text.color": NAVY,
        "axes.edgecolor": "#d3dce4",
        "svg.hashsalt": "polar-benchmark-1.1.0",
        "savefig.facecolor": "white",
    }
)


def save(fig: plt.Figure, name: str) -> None:
    for extension in ("png", "svg"):
        kwargs = {"metadata": {"Date": None}} if extension == "svg" else {}
        target = OUT / f"{name}.{extension}"
        fig.savefig(target, dpi=180, **kwargs)
        if extension == "svg":
            # Matplotlib path commands contain insignificant trailing spaces.
            text = "\n".join(
                line.rstrip() for line in target.read_text(encoding="utf-8").splitlines()
            )
            target.write_text(text + "\n", encoding="utf-8", newline="\n")
    plt.close(fig)


def comparisons(summary: dict, *, report: bool = False) -> None:
    if report:
        # The PDF renderer caps image height at 158 mm: 10.5 pt labels on a
        # 7.2-inch square remain just over 9 pt after that reduction.
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2))
        fig.subplots_adjust(left=0.34, right=0.97, top=0.865, bottom=0.20, hspace=0.42)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14.4, 6.6))
        fig.subplots_adjust(left=0.17, right=0.97, top=0.80, bottom=0.19, wspace=0.90)
    label_size = 10.5 if report else 10
    labels = {
        "conservative_fusion": "Conservative fusion",
        "prior_incumbent": "Prior fusion",
        "replacement_fusion": "Replacement control",
        "adapted_siglip2": "Adapted SigLIP2-B",
        "adapted_dinov2": "Adapted DINOv2-B",
        "historical_ensemble": "Historical ensemble",
        "frozen_dinov2_base": "Frozen DINOv2-B",
        "frozen_dinov3_base": "Frozen DINOv3-B",
        "frozen_siglip2_base": "Frozen SigLIP2-B",
        "adapted_convnextv2": "Adapted ConvNeXt V2-B",
        "frozen_convnextv2_base": "Frozen ConvNeXt V2-B",
    }
    for ax, task, title in zip(
        axes,
        ("polar4", "polar9"),
        ("Four classes · 3,329 images", "Nine classes · 6,984 images"),
        strict=True,
    ):
        metrics = summary["tasks"][task]["metrics"]
        names = [
            "conservative_fusion",
            "prior_incumbent",
            "replacement_fusion",
            "adapted_siglip2",
            "adapted_dinov2" if task == "polar9" else "historical_ensemble",
            "frozen_siglip2_base",
            "frozen_dinov3_base",
            "frozen_dinov2_base",
            "adapted_convnextv2",
            "frozen_convnextv2_base",
        ]
        for y, name in enumerate(names):
            row = metrics[name]
            value = 100 * row["macro_f1"]
            lo, hi = np.array(row["uncertainty"]["macro_f1"]["source_group_ci95"]) * 100
            color = {
                "conservative_fusion": TEAL,
                "prior_incumbent": NAVY,
                "replacement_fusion": ORANGE,
            }.get(name, GREY)
            if name == "conservative_fusion":
                ax.axhspan(y - 0.44, y + 0.44, color="#e9f5f2", zorder=0)
            ax.errorbar(
                value,
                y,
                xerr=[[value - lo], [hi - value]],
                fmt="o",
                color=color,
                capsize=3,
                markersize=6,
                lw=1.5,
            )
            ax.text(
                99.6, y, f"{value:.2f}", ha="right", va="center", color=color, fontsize=label_size
            )
        ax.set_yticks(range(len(names)), [labels[name] for name in names])
        ax.tick_params(axis="y", length=0, pad=8, labelsize=label_size)
        ax.tick_params(axis="x", labelsize=label_size)
        ax.set_ylim(9.7, -0.8)
        ax.set_xlim(82, 100)
        ax.set_xticks([85, 90, 95])
        ax.grid(axis="x", color="#e6ebef", zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel("Macro-F1 (%)", fontsize=label_size)
        ax.set_title(
            title,
            loc="left",
            fontsize=11 if report else 12,
            fontweight="bold",
            pad=9 if report else 16,
        )
    if report:
        axes[0].set_xlabel("")
        fig.text(0.04, 0.965, "POLAR | locked final evaluation", fontsize=14, weight="bold")
        fig.text(
            0.04,
            0.931,
            "All predeclared systems; separate four- and nine-class tasks.",
            fontsize=label_size,
            color=GREY,
        )
    else:
        fig.text(0.05, 0.95, "POLAR | locked final evaluation", fontsize=20, weight="bold")
        fig.text(
            0.05,
            0.895,
            "All predeclared systems. Separate taxonomies; no cross-task ranking.",
            fontsize=11,
            color=GREY,
        )
    legend = [
        Line2D([0], [0], marker="o", color=c, lw=0, label=label)
        for c, label in (
            (TEAL, "Development nominee"),
            (NAVY, "Retained prior"),
            (ORANGE, "Diagnostic control"),
            (GREY, "Fixed comparator"),
        )
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.065 if report else 0.075),
        ncol=2 if report else 4,
        frameon=False,
        fontsize=label_size,
    )
    fig.text(
        0.04 if report else 0.05,
        0.018 if report else 0.038,
        (
            "Bars: marginal 95% source-group bootstrap intervals.\nPromotion uses paired tests and every locked gate."
            if report
            else "Bars: marginal 95% source-group bootstrap intervals. Promotion uses paired tests and all locked gates."
        ),
        fontsize=label_size,
        color=GREY,
    )
    save(fig, "report_comparison" if report else "benchmark_comparison")


def confusion(summary: dict) -> None:
    row = summary["tasks"]["polar9"]["metrics"]["conservative_fusion"]
    matrix = np.array(row["confusion_matrix"])
    recall = matrix / matrix.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(9.6, 8.7))
    fig.subplots_adjust(left=0.16, bottom=0.20, right=0.90, top=0.82)
    chart = ax.imshow(recall, cmap="Blues", vmin=0, vmax=100)
    for i in range(9):
        for j in range(9):
            if matrix[i, j]:
                ax.text(
                    j,
                    i,
                    f"{recall[i, j]:.1f}%\n{matrix[i, j]}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color="white" if recall[i, j] > 55 else NAVY,
                )
    names = [name.capitalize() for name in row["class_names"]]
    ax.set_xticks(range(9), names, rotation=45, ha="right")
    ax.set_yticks(range(9), names)
    ax.set_xlabel("Predicted posture", labelpad=12, fontsize=11)
    ax.set_ylabel("Annotated posture", labelpad=12, fontsize=11)
    ax.tick_params(length=0, labelsize=11)
    colorbar = fig.colorbar(chart, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Row share (%)", fontsize=11)
    colorbar.ax.tick_params(labelsize=11)
    fig.text(0.06, 0.94, "Nine-class error structure", fontsize=20, weight="bold")
    fig.text(
        0.06, 0.89, "Conservative fusion · 94.58% macro-F1 · 369 errors / 6,984 images", fontsize=11
    )
    fig.text(
        0.06,
        0.04,
        "Cells show row-normalized percentage and count; blank cells have zero count.\nLargest pairwise confusion: standing ↔ walking (61 images). No relabeling or test-driven fitting.",
        fontsize=10.5,
        color=GREY,
    )
    save(fig, "nine_class_confusion")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 5.7))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.84, bottom=0.11)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, title, body, color=NAVY, fill="#f2f5f8"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.10,rounding_size=0.12",
                ec=color,
                fc=fill,
                lw=1.1,
            )
        )
        ax.text(
            x + w / 2,
            y + h * 0.72,
            title,
            ha="center",
            va="center",
            weight="bold",
            fontsize=11,
            color=color,
        )
        ax.text(
            x + w / 2, y + h * 0.34, body, ha="center", va="center", fontsize=10, linespacing=1.45
        )

    def arrow(start, end, color=GREY):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                lw=1.3,
                color=color,
                connectionstyle="arc3",
            )
        )

    box(0.15, 1.85, 2.0, 1.3, "Given-person input", "RGB image\n+ annotated target box")
    box(
        3.0,
        3.5,
        4.0,
        1.2,
        "A · DINOv2 anchor",
        "4 classes: frozen two-view RBF\n9 classes: adapted, three-seed mean",
    )
    box(
        3.0,
        1.9,
        4.0,
        1.2,
        "F · Frozen SigLIP2",
        "Full frame + 10% context crop\nCalibrated RBF classifier",
    )
    box(
        3.0,
        0.3,
        4.0,
        1.2,
        "S · Adapted SigLIP2",
        "25% context, aspect-preserving view\nTop-block adaptation, three-seed mean",
    )
    for y in (4.1, 2.5, 0.9):
        arrow((2.28, 2.5), (2.86, y))
        arrow((7.13, y), (8.25, 2.5))
    box(
        8.4,
        1.55,
        4.0,
        1.9,
        "Conservative fusion",
        "p = 0.50 A + 0.25 F + 0.25 S\nWeights fixed in development\nPredict argmax(p)",
        TEAL,
        "#e9f5f2",
    )
    ax.text(
        10.4,
        0.6,
        "Retained prior: 0.50 A + 0.50 F\nNo test-trained router or threshold",
        ha="center",
        va="center",
        fontsize=10,
    )
    fig.text(
        0.035,
        0.945,
        "Person-centric views + complementary representations",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.035,
        0.025,
        "Source audit precedes fitting. Training labels supervise models; labels and source groups are not inference inputs.",
        fontsize=10,
        color=GREY,
    )
    save(fig, "system_overview")


def report_architecture() -> None:
    """Compact, vertically routed counterpart to the wide web diagram."""
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.89, bottom=0.08)
    ax.set_xlim(0, 7.2)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, color=NAVY, fill="#f2f5f8"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.06,rounding_size=0.08",
                ec=color,
                fc=fill,
                lw=1.1,
            )
        )
        ax.text(
            x + w / 2,
            y + h - 0.22,
            title,
            ha="center",
            va="center",
            weight="bold",
            fontsize=11,
            color=color,
        )
        ax.text(
            x + w / 2,
            y + h * 0.39,
            body,
            ha="center",
            va="center",
            fontsize=10.5,
            linespacing=1.25,
        )

    def arrow(start, end):
        ax.add_patch(
            FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11, lw=1.2, color=GREY)
        )

    ax.add_patch(
        FancyBboxPatch(
            (0.3, 4.9),
            6.6,
            0.5,
            boxstyle="round,pad=0.06,rounding_size=0.08",
            ec=NAVY,
            fc="#f2f5f8",
            lw=1.1,
        )
    )
    ax.text(
        3.6,
        5.15,
        "RGB image + annotated target-person box",
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
    )
    branches = (
        (0.12, "A · DINOv2", "4-class: frozen\ntwo-view RBF\n9-class: adapted\nthree-seed mean"),
        (2.57, "F · Frozen SigLIP2", "Full frame +\n10% context crop\nCalibrated RBF"),
        (
            5.02,
            "S · Adapted SigLIP2",
            "25% context crop\nAspect-preserving\nTop-block adaptation\nThree-seed mean",
        ),
    )
    for x, title, body in branches:
        box(x, 2.7, 2.05, 1.7, title, body)
        arrow((3.6, 4.82), (x + 1.025, 4.5))
        arrow((x + 1.025, 2.61), (3.6, 2.16))
    box(
        0.3,
        0.9,
        6.6,
        1.15,
        "Conservative fusion",
        "p = 0.50 A + 0.25 F + 0.25 S\nDevelopment-fixed weights; predict argmax(p)",
        TEAL,
        "#e9f5f2",
    )
    ax.text(
        3.6,
        0.35,
        "Retained prior: 0.50 A + 0.50 F\nNo test-trained router or threshold",
        ha="center",
        va="center",
        fontsize=10.5,
        linespacing=1.3,
    )
    fig.text(0.045, 0.953, "Person-centric views and fixed fusion", fontsize=14, weight="bold")
    fig.text(
        0.045,
        0.025,
        "Training labels and source groups are not inference inputs.",
        fontsize=10.5,
        color=GREY,
    )
    save(fig, "report_system")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = json.loads((ROOT / SOURCE).read_text(encoding="utf-8"))
    comparisons(summary)
    comparisons(summary, report=True)
    confusion(summary)
    architecture()
    report_architecture()
    manifest = {"schema_version": 1, "project_version": "1.1.0", "sources": {}, "artifacts": {}}
    for section, paths in (
        ("sources", [ROOT / SOURCE, Path(__file__).resolve()]),
        ("artifacts", sorted(OUT.glob("*.png")) + sorted(OUT.glob("*.svg"))),
    ):
        for path in paths:
            data = path.read_bytes()
            normalized = path.suffix in {".py", ".svg"}
            if normalized:
                data = data.replace(b"\r\n", b"\n")
            manifest[section][path.relative_to(ROOT).as_posix()] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "normalized_lf": normalized,
            }
    (OUT / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(f"Rendered {len(manifest['artifacts'])} figure files")


if __name__ == "__main__":
    main()
