# Figures

## Current four-/nine-class study

| Figure | Scope |
| --- | --- |
| [Comparison](polar_20260921/benchmark_comparison.png) ([SVG](polar_20260921/benchmark_comparison.svg)) | All 20 predeclared candidates and marginal source-group intervals |
| [System](polar_20260921/system_overview.png) ([SVG](polar_20260921/system_overview.svg)) | Person-centric branches, fixed blend and retained prior |
| [Nine-class confusion](polar_20260921/nine_class_confusion.png) ([SVG](polar_20260921/nine_class_confusion.svg)) | Nominee row-normalized percentages and counts |

```bash
python tools/render_benchmark_release.py
python tools/check_project.py
```

The [manifest](polar_20260921/figure_manifest.json) binds source, renderer and
outputs. Plot axes may be zoomed; points and interval lengths are not zero-based
bars. Marginal intervals do not replace paired significance tests.
[Public probabilities](../results/polar_20260921/README.md) support metric replay.

## Historical overview figures

| Figure | Scope |
| --- | --- |
| [Original POLAR comparison](benchmark_results.png) ([SVG](benchmark_results.svg)) | Six historical four-class candidates |
| [V-COCO transfer](transfer_results.png) ([SVG](transfer_results.svg)) | Two fixed person-level test systems |
| [V-COCO representations](representation_results.png) ([SVG](representation_results.svg)) | Nested development, including DINOv3 |
| [Original confusion](polar_confusion_matrix.png) ([SVG](polar_confusion_matrix.svg)) | Original four-class ensemble |
| [Learning curve](polar_scale_curve.png) ([SVG](polar_scale_curve.svg)) | Deterministic nested subsets; validation only |

The first three are rebuilt by `tools/render_project_figures.py` and bound by
the [overview manifest](project_figure_manifest.json). Original study figures
remain byte-preserved in the [origin record](../results/project_origin.json);
do not overwrite them using legacy renderers.

Only aggregate diagrams/charts are distributed, not source photographs.
