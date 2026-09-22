# Benchmark evidence

## Current four-/nine-class evaluation

| Evidence | Contents |
| --- | --- |
| [Public probability package](polar_20260921/README.md) | 32 prediction sets; class order, source groups and row IDs |
| [Cohort](polar_20260921/cohort.csv) / [quarantine](polar_20260921/quarantine.csv) | Retained/excluded memberships, labels, boxes and hashes |
| [Audit](polar_20260921/data_audit.json) / [manifest](polar_20260921/manifest.json) | Source-audit counts and exact evidence bindings |
| [Metrics](../docs/research/20260921_final_evaluation/results/metrics.csv) | All 20 fixed comparison candidates |
| [Comparisons](../docs/research/20260921_final_evaluation/results/comparisons.csv) | All 18 paired tests |
| [Summary](../docs/research/20260921_final_evaluation/results/summary.json) | Class-wise, probability, uncertainty and promotion records |

```bash
python tools/check_project.py
python tools/verify_benchmark_predictions.py
python tools/verify_benchmark_predictions.py --resample
```

The first needs only the standard library. Prediction replay needs installed
project dependencies, not images, weights or a GPU. The final command repeats
paired bootstrap and randomization calculations. [Interpretation](../docs/RESULTS.md).

## Preserved earlier evidence

| Family | Artifacts | Boundary |
| --- | --- | --- |
| Original POLAR | [Metrics](polar_test_metrics.csv), [confusions](polar_test_confusions.json), [uncertainty](polar_test_uncertainty.json) | Original four-class ensemble and controls |
| Original gate | [Selection](polar_final_selection_lock.json), [access](polar_test_access_gate.json), [summary](polar_test_summary.json) | Original locked test opening |
| Original data audit | [Audit](polar_data_audit.json), [quarantine](polar_quarantine.csv) | 125 pre-fit exclusions |
| Secondary task | [Metrics](polar_test_secondary_metrics.csv) | Three-class collapse, not a replacement headline |
| Development | [Scale](polar_extension_summary.json), [training](polar_training_summary.json) | Validation-only comparisons |
| V-COCO follow-up | [Metrics](vcoco_v2/official_test_metrics.csv), [uncertainty](vcoco_v2/official_test_uncertainty.json) | 6,077 held-out people |
| V-COCO representations/fusion | [Metrics](vcoco_v3/source_tag_development_metrics.csv), [decisions](vcoco_v3/source_tag_promotion_decisions.json) | Nested development, including DINOv3 |
| Extraction provenance | [Origin](project_origin.json) | Original Git revisions and 330 frozen files |

The original study manifests describe their historical trees. Their hashes are
not refreshed for new results. Earlier COCO-pilot tables and post-lock exploratory
studies remain provenance, not current POLAR headline measurements.
