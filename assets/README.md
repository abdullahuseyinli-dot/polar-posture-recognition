# Figures

Current figures are generated from the frozen public evidence; no inference or
new experiment is run.

| Figure | Scope |
| --- | --- |
| [Held-out comparison](benchmark_results.png) ([SVG](benchmark_results.svg)) | Six POLAR candidates with marginal 95% bootstrap intervals |
| [V-COCO transfer](transfer_results.png) ([SVG](transfer_results.svg)) | Two predeclared systems on the same held-out person rows |
| [Representation and fusion screens](representation_results.png) ([SVG](representation_results.svg)) | Separate nested-development comparisons, including DINOv3 |
| [Confusion matrix](polar_confusion_matrix.png) ([SVG](polar_confusion_matrix.svg)) | Four-class locked ensemble |
| [Learning curve](polar_scale_curve.png) ([SVG](polar_scale_curve.svg)) | Deterministic nested training subsets; validation only |

Regenerate only the new figures:

```bash
python tools/render_project_figures.py
python tools/check_project.py
```

[Manifest](project_figure_manifest.json) binds sources, renderer and output hashes.
Rendering requires Matplotlib and NumPy, not a GPU. Confidence intervals are exported
quantities; the public package does not contain the row probabilities to recompute them.
Point-chart axes may be zoomed, as labeled; their markers do not encode bar lengths.

Original figures, including attribution, calibration, scale and robustness diagnostics,
remain byte-preserved in the [origin record](../results/project_origin.json).
Do not run legacy renderers over these frozen assets; use a new output directory.
Only aggregate diagrams/charts are distributed, not qualitative source photographs.
