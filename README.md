# POLAR Posture Recognition

Person-centric visual representations for four- and nine-class posture recognition.

[![Quality gates](https://github.com/abdullahuseyinli-dot/polar-posture-recognition/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/abdullahuseyinli-dot/polar-posture-recognition/actions/workflows/ci.yml?query=branch%3Amain)
[![Python](https://img.shields.io/badge/Python-3.11%E2%80%933.12-3776AB.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/Code-MIT-0F766E.svg)](LICENSE)

This benchmark classifies a specified person in a still image using RGB and a
supplied bounding box. The four-class task covers **sitting, standing, walking and
running**; the nine-class task adds **bending, jumping, lying, squatting and stretching**.

The system combines person-context views, calibrated nonlinear classifiers and
partial backbone adaptation. DINOv2, DINOv3, SigLIP2 and ConvNeXt V2 are compared
under a source-overlap-audited POLAR protocol, with locked evaluation and public
per-example predictions.

## Results

| Audited POLAR task | Retained baseline F1 | Evaluated fusion F1 | Fusion accuracy | Test images |
| --- | ---: | ---: | ---: | ---: |
| Four classes | 94.75% | **95.21%** | **95.76%** | 3,329 |
| Nine classes | 94.43% | **94.58%** | **94.72%** | 6,984 |

F1 is macro-averaged. The conservative fusion was selected in development before
final evaluation. It improves on the original four-class
ensemble by **+1.22 percentage points**, and on the nine-class adapted DINOv2
reference by **+0.68 points**, with positive paired intervals and Holm-adjusted
p-values below 0.05. Its smaller increments over the immediate prior do not pass
all locked promotion gates, so the prior models remain retained. The table
distinguishes those decisions from measured scores.

[Complete results](docs/RESULTS.md) · [Technical report](docs/POLAR_BENCHMARK_REPORT.md) ·
[PDF](output/pdf/polar_benchmark_report_v1.1.0.pdf) ·
[Prediction evidence](results/polar_20260921/README.md) · [Comparison scope](docs/COMPARISONS.md)

![All 20 predeclared four- and nine-class systems, with source-group confidence intervals.](assets/polar_20260921/benchmark_comparison.png)

## System design

The model uses an RGB image and a supplied target-person box. Frozen encoders
see both the full frame and a person-context crop; adapted encoders use an
aspect-preserving person view. A fixed probability blend combines complementary
representations without learning a router on test labels.

![DINOv2 anchor, frozen SigLIP2 and adapted SigLIP2 combined with development-fixed probability weights.](assets/polar_20260921/system_overview.png)

For four classes the DINOv2 anchor is a calibrated frozen-feature classifier;
for nine it is a three-seed adapted model. The conservative blend assigns 50% to
the anchor and 25% to each SigLIP2 branch. The retained prior uses an equal blend
of the anchor and frozen SigLIP2.

The engineering contribution is the audited data protocol, person-preserving
adaptation, representation comparison and reproducible fusion evaluation, built
on the [original pretrained encoders](docs/REPRESENTATIONS.md#original-methods).

[Architecture and evidence map](docs/ARCHITECTURE.md) ·
[Representation study](docs/REPRESENTATIONS.md) · [Model card](docs/MODEL_CARD.md)

## What the experiments establish

- **Complementarity matters.** On nine classes, conservative fusion rescues 80
  adapted-DINOv2 errors while harming 33 correct predictions: 47 fewer errors.
- **More components are not automatically better.** On four classes, frozen
  SigLIP2 alone reaches 95.19% macro-F1; the nominated fusion's 95.21% is not a
  supported improvement over it.
- **DINOv3 is included.** Its frozen two-view classifier reaches 93.05% on four
  classes and 91.19% on nine under the documented encoder-specific preprocessing.

The nine-class nominee's remaining errors concentrate in plausible posture
boundaries, including standing/walking and bending/stretching.
[Per-class results and confusion matrix](docs/RESULTS.md#nine-class-error-structure)
show the counts without changing labels or selecting a new model after evaluation.

## Verify the results

A standard-library check needs no dataset, model weights or GPU:

```bash
git clone https://github.com/abdullahuseyinli-dot/polar-posture-recognition.git
cd polar-posture-recognition
python tools/check_project.py
```

With the [project dependencies installed](docs/REPRODUCIBILITY.md), replay the
published probabilities and, optionally, all paired statistical tests:

```bash
python tools/verify_benchmark_predictions.py
python tools/verify_benchmark_predictions.py --resample
```

The evidence package includes 32 prediction sets, audited split membership and
source hashes. It supports recalculating F1, accuracy, NLL, Brier, calibration,
confusions, uncertainty and rescue/harm counts. Full checkpoint replay additionally
requires locally obtained images and weights; neither is redistributed.

## Evaluation scope

These are **given-person-box still-image classification** results, not video
recognition or person-detection scores. The original nine-class dataset contains
35,324 images; the audited cohort contains 35,007 after source-overlap quarantine.
The four-class test was used historically and is part of the nine-class test.
Source grouping does not prove subject or scene independence.

No directly matched external nine-class baseline has been verified, so no
state-of-the-art claim is made. See the [external comparison table](docs/COMPARISONS.md)
for the differences in task, taxonomy and evaluation population.

## Earlier studies

| Study | Macro-F1 | Evaluation boundary |
| --- | ---: | --- |
| Original four-class POLAR ensemble | 93.99% | Same 3,329 test images; historical study |
| Three-class POLAR collapse | 96.11% | Walking/running merged; different task |
| V-COCO scale-conditioned DINO stack | 86.63% | Target-trained, person-level held-out follow-up |
| V-COCO DINO + SigLIP reliability stack | 86.97% | Nested development only; not a later test score |

[Historical reports](docs/README.md#historical-studies) ·
[Executed historical notebook](human_activity_classification.ipynb) ·
[Full evidence inventory](results/README.md)

The separate [ARFTR project](https://github.com/abdullahuseyinli-dot/arftr)
studies temporal actor-centered modeling on Okutama video. Its results are not
POLAR baselines.

## Project information

Abdulla Huseyinli · Version **1.1.0** · [Documentation](docs/README.md) ·
[Reproducibility](docs/REPRODUCIBILITY.md) · [Validation](docs/VALIDATION.md) ·
[Citation](CITATION.cff) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

[MIT code licence](LICENSE) · [Dataset and model terms](THIRD_PARTY_NOTICES.md).
Original study versions and all 330 imported source/evidence files remain preserved.
[Project history](docs/PROJECT_HISTORY.md).
