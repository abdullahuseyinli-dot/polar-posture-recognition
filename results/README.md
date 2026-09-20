# Benchmark evidence

| Evidence family | Main artifacts | Evaluation boundary |
| --- | --- | --- |
| POLAR primary | [Metrics](polar_test_metrics.csv), [confusions](polar_test_confusions.json), [uncertainty](polar_test_uncertainty.json) | Four classes, 3,329 held-out images |
| Test gate | [Selection](polar_final_selection_lock.json), [access](polar_test_access_gate.json), [summary](polar_test_summary.json) | One test opening after selection |
| Data audit | [Audit](polar_data_audit.json), [quarantine](polar_quarantine.csv) | Pre-fit source-overlap exclusions |
| Secondary task | [Metrics](polar_test_secondary_metrics.csv) | Three-class collapse, not a replacement headline |
| Development | [Scale](polar_extension_summary.json), [training](polar_training_summary.json) | Validation-only comparisons |
| V-COCO follow-up | [Metrics](vcoco_v2/official_test_metrics.csv), [uncertainty](vcoco_v2/official_test_uncertainty.json), [selection](vcoco_v2/final_selection_lock.json) | 6,077 held-out people; custom posture labels |
| Later representations/fusion | [Metrics](vcoco_v3/source_tag_development_metrics.csv), [decisions](vcoco_v3/source_tag_promotion_decisions.json) | Nested, image-grouped development; includes DINOv3 |
| Extraction provenance | [Origin record](project_origin.json) | Original Git revisions and frozen-file hashes |

```bash
python tools/check_project.py
```

The check recomputes confusion-derived metrics and paired-delta arithmetic and
verifies preserved source/evidence hashes. NLL, Brier, ECE, bootstrap intervals and
development predictions are exported quantities, not recomputed from aggregate matrices.

The original study manifests remain unmodified. They describe historical trees
(including old metadata and excluded photo composites), not the standalone 1.0.0 tree.
The new origin record is the verification boundary for this extraction.

Earlier 285-image COCO pilot tables remain for provenance; they are not POLAR results.
Post-lock exploratory tables are hypothesis-generating and do not change test selection.
