# Results

The primary metric is macro-F1. All percentages below are rounded for display;
the [machine-readable table](research/20260921_final_evaluation/results/metrics.csv)
preserves full precision. Four- and nine-class tasks are separate evaluations.

## Locked final evaluation

| System | Four-class F1 | Accuracy | Nine-class F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| Conservative fusion: development nominee | **95.21%** | **95.76%** | **94.58%** | **94.72%** |
| Prior fusion: retained under locked rule | 94.75% | 95.34% | 94.43% | 94.56% |
| Replacement fusion: diagnostic control | 95.69% | 96.19% | 94.53% | 94.67% |
| Frozen DINOv2-B, two-view RBF | 93.09% | 93.72% | 90.89% | 90.99% |
| Frozen DINOv3-B, two-view RBF | 93.05% | 93.75% | 91.19% | 91.29% |
| Frozen SigLIP2-B, two-view RBF | 95.19% | 95.70% | 91.92% | 91.90% |
| Frozen ConvNeXt V2-B, two-view RBF | 87.75% | 88.59% | 84.87% | 84.89% |
| Adapted SigLIP2-B, three-seed mean | 95.23% | 95.76% | 92.85% | 92.96% |
| Adapted ConvNeXt V2-B, three-seed mean | 89.39% | 90.30% | 85.52% | 85.62% |
| Adapted DINOv2-B, three-seed mean | — | — | 93.90% | 94.04% |
| Historical five-component ensemble | 93.99% | 94.56% | — | — |

Ten predeclared systems per task; a dash means absent from this comparison panel,
not a failed fit. The replacement control was not the development nominee; its
four-class point estimate is not used to select a new winner after test inspection.

![Final comparison panel](../assets/polar_20260921/benchmark_comparison.png)

## Uncertainty and model decisions

The nominated four-class model has 141 errors and a source-group 95% macro-F1
interval of **94.43–95.94%**. The nine-class model has 369 errors and an interval
of **94.03–95.11%**. Marginal intervals are not significance tests between systems.

| Paired comparison | F1 change (pp) | 95% source-group interval (pp) | Rescue / harm | Net corrections | Holm p |
| --- | ---: | --- | ---: | ---: | ---: |
| Four: nominee − historical ensemble | +1.223 | +0.573 to +1.940 | 75 / 35 | +40 | 0.00420 |
| Four: nominee − immediate prior | +0.465 | +0.064 to +0.890 | 27 / 13 | +14 | 0.19858 |
| Nine: nominee − adapted DINOv2 | +0.679 | +0.381 to +0.987 | 80 / 33 | +47 | 0.00180 |
| Nine: nominee − immediate prior | +0.158 | −0.091 to +0.408 | 44 / 33 | +11 | 0.90831 |

These comparisons use 5,000 source-group bootstrap draws and 10,000 two-sided
groupwise randomization draws. Holm adjustment covers the complete family of
18 comparisons, not only the four displayed. Intervals are neither simultaneous
nor adjusted for historical development selection.

Both nominees have positive matching-seed F1 and net corrections and pass the
per-class, NLL and Brier safety limits. Neither passes every locked promotion
condition against its immediate prior. **Retain both prior incumbents.** This does
not invalidate the measured nominee scores or the supported comparisons against
older references. [Full decisions](research/20260921_final_evaluation/RESULTS.md) ·
[All comparisons](research/20260921_final_evaluation/results/comparisons.csv).

## Probability quality

| Task and model | NLL | Summed multiclass Brier | ECE |
| --- | ---: | ---: | ---: |
| Four: nominee | 0.1370 | 0.0664 | 0.0390 |
| Nine: nominee | 0.1684 | 0.0812 | 0.0228 |
| Nine: prior | 0.2021 | 0.0908 | 0.0447 |

These metrics, confidence summaries and fixed-coverage selective-risk curves are
recomputed from the [public probabilities](../results/polar_20260921/README.md).
Selective risk is diagnostic; no abstention threshold was selected on test labels.

## Nine-class error structure

| Class | Test support | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Sitting | 1,034 | 97.29% | 97.10% | 97.19% |
| Standing | 921 | 93.49% | 93.59% | 93.54% |
| Walking | 638 | 91.34% | 90.91% | 91.12% |
| Running | 736 | 95.04% | 93.75% | 94.39% |
| Bending | 769 | 93.51% | 93.76% | 93.64% |
| Jumping | 679 | 93.10% | 97.35% | 95.18% |
| Lying | 668 | 98.65% | 98.35% | 98.50% |
| Squatting | 836 | 95.97% | 96.77% | 96.37% |
| Stretching | 703 | 92.93% | 89.76% | 91.32% |

![Nine-class nominee confusion, percentages and counts](../assets/polar_20260921/nine_class_confusion.png)

The largest bidirectional pairs are standing/walking (61), bending/stretching
(46), running/jumping (37), standing/stretching (36), and walking/running (35).
These counts describe errors, not their causes. Static ambiguity, viewpoint,
background and annotation semantics remain rival explanations; none was isolated
by relabeling or an intervention study in this evaluation.

## What changed during development

| Observation | Evidence | What it supports |
| --- | --- | --- |
| Nonlinear frozen heads beat the best screened linear heads | Validation-selected RBF advantage: +0.73–1.36 pp across four families on four classes; +0.75–2.29 pp on nine | Representation/classifier interaction; not a post-test tuning instruction |
| Person-preserving DINOv2 adaptation improved the initial nine-class recipe | Matched seed-42 validation: 90.73% → 93.23% | Combined view/augmentation recipe; not an isolated crop-only effect |
| Adaptation was stable across the tested seeds | Nine-class DINOv2 validation: 93.23%, 93.41%, 93.34% | Limited seed sensitivity within this development setup |
| Selective fusion beat indiscriminate averaging | Nine-class development: DINO + frozen SigLIP2 94.00%; uniform all-family blend 93.33% | Complementarity matters more than component count |
| Additional adaptation offered a small final increment | Nominee versus prior: +0.46 pp four / +0.16 pp nine | Measured positive differences, insufficient for locked promotion |

Sources: [development evidence review](research/20260921_bounded_development/EVIDENCE_REVIEW.md)
and [final analysis](research/20260921_final_evaluation/results/summary.json).
Development gains are not interchangeable with confirmatory test comparisons.

## Historical POLAR and V-COCO studies

| Study / system | Macro-F1 | Accuracy | Boundary |
| --- | ---: | ---: | --- |
| Original four-class ensemble | 93.99% | 94.56% | Same 3,329 POLAR test images |
| Original DINOv2-B multilayer RBF | 92.74% | 93.42% | Original frozen-feature recipe, not the current two-view comparator |
| Original DINOv2-B multilayer logistic regression | 92.58% | 93.24% | Same original four-class protocol |
| Original top-four-block DINOv2-B | 92.52% | 93.27% | Original four-class recipe |
| Original fully adapted DINOv2-S | 91.31% | 92.10% | Original four-class recipe |
| Original fully adapted ConvNeXt-S | 89.14% | 89.94% | Original four-class recipe |
| Three-class collapse of original ensemble | 96.11% | 96.22% | Walking/running merged after inference |
| Direct three-class frozen probe | 95.31% | 95.43% | Separate declared secondary system |
| V-COCO source-only DINO | 70.71% | 70.10% | Custom three-class target task |
| V-COCO target-trained scale-conditioned stack | 86.63% | 87.95% | 6,077 people / 3,708 held-out images |
| V-COCO DINO + SigLIP reliability stack | 86.97% | — | Later nested development, not a test result |

The original POLAR ensemble exceeded its strongest component by **+1.25 pp
[+0.65, +1.86]**. The frozen-DINOv2 learning curve rose from **84.87% to 91.50%**
validation F1 as training size grew from 242 to 9,958 images; repeated seeds did
not provide independent subset replications. The historical selected validation
blend reached 94.65%, a development score rather than a test result.

V-COCO's **+15.92 pp [+14.54, +17.35]** gain includes target supervision and
preprocessing changes, not just architecture. Its more controlled development
comparisons supported multiview stacking (+1.18 pp [0.37, 2.01]) and factorization
(+1.11 pp [0.56, 1.66]). Those findings motivate representations; they do not prove
the same effect on nine-class POLAR.

[Original POLAR report](POLAR_PUBLIC_REPORT.md) ·
[V-COCO report](VCOCO_V2_EXTERNAL_TRANSFER.md) ·
[Representation controls](REPRESENTATIONS.md) · [Evidence inventory](../results/README.md).

## Claim boundary

This is an audited-cohort, known-box posture benchmark. The historical four-class
test is nested in the nine-class test; it is not a wholly untouched new holdout.
Source groups are not subject/session identifiers. Neither these results nor
unmatched external reports establish a standard POLAR leaderboard rank.
See [external comparisons](COMPARISONS.md) and the [model card](MODEL_CARD.md).
