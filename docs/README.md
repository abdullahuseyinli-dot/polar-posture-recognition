# Documentation

| Guide | Purpose |
| --- | --- |
| [Results](RESULTS.md) | All 20 final systems, paired gains, uncertainty and error structure |
| [Technical report](POLAR_BENCHMARK_REPORT.md) | Current four-/nine-class methods and interpretation |
| [Report PDF](../output/pdf/polar_benchmark_report_v1.1.0.pdf) | Version 1.1.0 technical report |
| [Architecture](ARCHITECTURE.md) | Views, adaptation, fusion and experimental evidence map |
| [Model card](MODEL_CARD.md) | Inputs, training, evaluation and limitations |
| [Reproducibility](REPRODUCIBILITY.md) | Public prediction replay, tests and full-training boundaries |
| [External comparisons](COMPARISONS.md) | Related results and protocol differences |
| [Representations](REPRESENTATIONS.md) | DINOv2, DINOv3, SigLIP2 and ConvNeXt V2 |
| [Validation](VALIDATION.md) | Release checks and their scope |
| [Project history](PROJECT_HISTORY.md) | Versions and preserved imported evidence |

[Prediction package](../results/polar_20260921/README.md) ·
[Evidence inventory](../results/README.md) · [Figures](../assets/README.md) ·
[Engineering case study](PORTFOLIO_ARTICLE.md) · [Release notes](releases/PROJECT_1.1.0.md).

## Current study records

Dated research records preserve the original plans, decisions and run paths:

- [Benchmark plan](research/20260920_polar_benchmark/PLAN.md).
- [Bounded development evidence](research/20260921_bounded_development/EVIDENCE_REVIEW.md).
- [Locked final protocol](research/20260921_final_evaluation/PROTOCOL.md).
- [Completed final evaluation](research/20260921_final_evaluation/RESULTS.md).
- [Generated metrics](research/20260921_final_evaluation/results/metrics.csv)
  and [summary](research/20260921_final_evaluation/results/summary.json).

## Historical studies

| Study | Read online | Preserved PDF |
| --- | --- | --- |
| Original four-class POLAR | [Report](POLAR_PUBLIC_REPORT.md) | [Study 1.0.0](../output/pdf/polar_public_report_v1.0.0.pdf) |
| V-COCO person-level follow-up | [Report](VCOCO_V2_EXTERNAL_TRANSFER.md) | [Study 2.0.0](../output/pdf/vcoco_v2_external_transfer_v2.0.0.pdf) |
| V-COCO representations | [DINOv3 record](DINOV3_ACCESS.md) | See [representation guide](REPRESENTATIONS.md) |

The [executed notebook](../human_activity_classification.ipynb) covers the earlier
POLAR/V-COCO studies, not the September nine-class evaluation.
The [COCO pilot](LEGACY_COCO_STUDY.md), [lineage](RESULT_LINEAGE.md),
[scale protocol](POLAR_SCALE_STUDY_PROTOCOL.md) and
[study release notes](releases/POLAR_STUDY_V2.0.0.md) remain provenance records,
not instructions to overwrite the current presentation or reopen a test gate.

Temporal modeling is maintained separately in
[ARFTR](https://github.com/abdullahuseyinli-dot/arftr).
