# Independent evidence review: bounded adapted challengers

Date: 2026-09-21. Status: **prior development predictions replayed; next phase
specified before its results**. No new test predictions, training or human-review
labels were used in this review. The executable
[bounded protocol](../../../experiments/polar_bounded_protocol_20260921.json) is
authoritative for the next run's budget, candidates and gates.

This extends the [previous evidence review](../20260920_polar_benchmark/EVIDENCE_AND_COMPARATORS.md)
and [knowledge graph](../20260920_polar_benchmark/knowledge_graph.json). It does not
replace the historical four-class test result or establish a nine-class SOTA claim.

## 1. Recomputed parent results

Source run: `.runs/benchmark_20260920_2250/`. Every prediction array was checked
against the appropriate audited validation manifest, including exact image IDs,
labels and class ordering, before metrics and transitions were recalculated.

| Task and candidate | Validation macro-F1 | Errors | NLL | Summed Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Nine-class incumbent: equal adapted DINOv2 / frozen SigLIP2 fusion | **94.00454%** | **408 / 6,966** | **0.22648** | **0.09725** | 0.04677 |
| Nine-class adapted DINOv2, three-seed probability mean | 93.47106% | 445 / 6,966 | 0.38826 | 0.10668 | 0.04354 |
| Nine-class uniform blend of all five systems | 93.33395% | 455 / 6,966 | 0.29283 | 0.12467 | 0.11953 |
| Four-class incumbent: equal frozen DINOv2 / SigLIP2 fusion | **95.15074%** | **146 / 3,327** | **0.16205** | **0.07391** | 0.04967 |
| Four-class frozen SigLIP2 alone | 94.96680% | 153 / 3,327 | 0.16792 | 0.07619 | **0.02702** |
| Four-class uniform blend of all four systems | 94.61639% | 162 / 3,327 | 0.19068 | 0.08524 | 0.07215 |

These scores are **development-selected validation measurements**, not new held-out
results. The historical 93.98833% four-class test score remains unchanged and
cannot be subtracted from 95.15074% validation as an achieved improvement.

Artifact families in the parent run:

- `polar{4,9}/development_analysis/*_validation.npz`: actual probability arrays.
- `polar{4,9}/development_analysis/analysis.json`: declared members and metrics.
- `polar{4,9}/development_analysis/analysis_request.json`: input hashes and cohorts.
- `polar{4,9}/heads/*/validation_ranking.csv`: complete frozen-head comparisons.
- `polar9/adaptation/confirmation/{selection_lock,summary}.json`: recipe and seeds.
- `data/polar9_data_audit.json`: original/audited populations and limitations.

### Rescue versus harm: what actually improved?

| Comparison | Rescued | Harmed | Net corrections | Macro-F1 change |
| --- | ---: | ---: | ---: | ---: |
| Nine-class incumbent vs adapted DINOv2 | 82 | 45 | **+37** | **+0.53348 pp** |
| Four-class incumbent vs frozen SigLIP2 | 38 | 31 | **+7** | **+0.18394 pp** |
| Nine-class uniform-all vs incumbent | 85 | 132 | **-47** | **-0.67059 pp** |
| Four-class uniform-all vs incumbent | 23 | 39 | **-16** | **-0.53435 pp** |

A 3,000-draw paired source-group bootstrap, seed 20260921, gave intervals of
**[+0.214, +0.851] pp** for the nine-class fusion gain and **[-0.368, +0.743] pp**
for the four-class fusion gain. There were 6,964 detected validation source groups
for nine classes and 3,327 for four classes; no resample omitted a class. These
are **conditional, descriptive intervals after validation selection**, not
independent confirmation or multiplicity-adjusted discovery evidence. In
particular, the four-class fusion advantage remains weak evidence of superiority
over frozen SigLIP2 despite having the best selected point estimate.

Nine-class fusion improves all nine class F1 scores over adapted DINOv2, most
strongly stretching (+1.662 pp). Four-class fusion loses 0.394 pp running F1
relative to SigLIP2 while improving the other three classes. Gains are not uniform
across tasks or classes.

### Probability quality is not uniformly improved

Both selected fusions improve NLL and Brier relative to their strongest components,
but their ECE is worse. Their mean maximum probabilities are 0.8980 and 0.9072,
below their respective accuracies 0.9414 and 0.9561: aggregate underconfidence is
present. Do not call this a general calibration improvement on every criterion.

The nine-class fusion's error rate among its most-confident half is 0.947%, versus
0.316% for adapted DINOv2. Thus the fusion's better full-coverage classification
does not establish better selective classification. No confidence threshold or
router is learned from these diagnostic findings in the bounded phase.

## 2. Mechanisms supported by the evidence

1. **Nonlinear, two-view heads matter across families.** All eight frozen winners
   (four encoders times two tasks) are two-view RBF heads with C=10. The best
   linear alternative in every family is two-view logistic regression with
   C=0.001. The RBF advantage ranges from +0.729 to +1.358 pp in four classes and
   +0.749 to +2.289 pp in nine classes. This is stronger repository evidence for
   head/view engineering than for automatically replacing a model with a newer
   backbone name.
2. **The person-preserving adaptation recipe merits transfer.** In the fixed
   seed-42 nine-class comparison, DINOv2 rises from 90.72558% with the historical
   mild recipe to 93.23034% with the person-preserving recipe (+2.50476 pp).
   Aspect handling and augmentation change together; this is a recipe-level
   intervention, not isolated proof about either component. Its selected recipe
   then yields 93.23034%, 93.41102% and 93.33546% across seeds 42/52/62; sample
   SD is 0.09074 pp. Their probability average reaches 93.47106%.
3. **SigLIP2 supplies useful complementary information.** It is the strongest
   frozen family on both tasks and helps the adapted DINOv2 nine-class system.
   Whether adaptation improves it without eliminating that complementarity is
   now a specific falsifiable question.
4. **ConvNeXtV2 needs a bounded adapted control, not automatic fusion.** Its frozen
   predictions correctly classify 132 of the nine-class incumbent's errors, but
   would damage 779 incumbent-correct rows if substituted wholesale. In four
   classes the corresponding counts are 34 rescued / 239 harmed. Complementarity
   alone is not permission to overwrite a much stronger system. A last-stage
   control tests a concrete alternative while avoiding another broad sweep.

### Residual opportunity map

| True class | Nine-class incumbent errors | Four-class incumbent errors |
| --- | ---: | ---: |
| Sitting | 37 | 6 |
| Standing | 61 | 56 |
| Walking | 51 | 59 |
| Running | 45 | 25 |
| Bending | 60 | — |
| Jumping | 54 | — |
| Lying | 11 | — |
| Squatting | 37 | — |
| Stretching | 52 | — |
| **Total** | **408** | **146** |

Standing/walking account for 115 of 146 four-class errors (78.8%), but nine-class
errors are broader. The additional classes motivate preserving person geometry
and limb context; this is a mechanism hypothesis, not proof of the causes of any
individual error. No annotation-derived pose category or hidden support label is
an inference input.

## 3. Exact next bounded experiment

The main protocol fixes **12 production fits**: two tasks, two foundation-model
families, three seeds each. Four separate engineering smokes verify the model/task
interfaces without making performance-selection decisions.

- SigLIP2-B: adapt the last four vision blocks, final normalization and pretrained
  attention pooler, plus a new classification head.
- ConvNeXtV2-B: adapt the last encoder stage and final normalization, plus a new
  classification head; it is a control, not an automatic fusion ingredient.
- Both use target-person box plus 25% context, person-preserving 224-square
  geometry, model-specific normalization and padding, original pinned foundation
  weights, and the unchanged audited training/validation manifests.
- Each has at most 20 epochs, minimum three, patience four, AdamW, backbone LR
  5e-6, new-head LR 1e-3, unweighted cross-entropy, no smoothing/mixup, effective
  batch size 64 and seeds 42/52/62. Frozen prefixes remain in evaluation mode.
- Pretrained SigLIP pooler parameters are backbone parameters; a module named
  `head` must not accidentally receive the new-classifier learning rate. SigLIP's
  normalization is not ImageNet mean/std. These are explicit implementation-review
  checks, not new hyperparameter proposals.

The schedules match epoch budgets, not FLOPs or exact trainable-parameter counts.
This phase cannot prove that every fully tuned convolutional or vision-language
competitor has been exhausted.

### Five candidates per task; only two new fusions

Let I be the unchanged incumbent; A the prior DINO anchor (frozen for four classes,
adapted three-seed mean for nine); S and C the new three-seed SigLIP2 and ConvNeXtV2
probability means. The prior incumbent is 0.5 A + 0.5 frozen SigLIP2.

| Candidate | Exact probabilities |
| --- | --- |
| Retain incumbent | I |
| Adapted SigLIP2 | S |
| Adapted ConvNeXtV2 control | C |
| Replacement fusion | 0.5 A + 0.5 S |
| Conservative fusion | 0.5 I + 0.5 replacement = 0.5 A + 0.25 frozen SigLIP2 + 0.25 S |

No learned router, threshold, arbitrary weight grid, selected seed or ConvNeXt
fusion is admitted. The conservative option explicitly keeps the frozen SigLIP2
evidence whose complementarity motivated the experiment.

### Predeclared nomination gates and stop rules

Relative to I, a new candidate must satisfy **all** protocol gates:

- Macro-F1 gain at least **0.20 pp** and positive net corrections.
- Each of its three seed-specific counterparts must improve macro-F1 and net
  corrections, holding the old anchor/incumbent fixed. These are seed checks,
  not independent outer folds or a hidden confirmation set.
- No individual class F1 reduction greater than **1.00 pp**.
- NLL increase no greater than **0.02** and summed Brier increase no greater
  than **0.01**. Report actual changes: the Brier tolerance still permits about
  10.3% relative worsening for the nine-class incumbent and 13.5% for four classes.
- Integrity checks and exact-prediction checkpoint replay pass for all required
  fits; probability replay tolerance is at most 1e-5.

Choose among eligible candidates by validation macro-F1, then NLL, then name.
Otherwise retain I. Recomputed intervals are descriptive diagnostics and do not
turn this adaptive development process into confirmatory inference. There is no
automatic new recipe, performance retry or test opening after the 12-fit budget.

## 4. Falsifying concerns and remaining boundary

- Adapted SigLIP2 may become more accurate alone but more correlated with DINOv2,
  eliminating fusion benefit. The replacement and conservative controls expose
  this outcome without changing weights after seeing it.
- A macro-F1 gain may be one seed, a single class, excessive intervention harm or
  degraded probability quality. The locked gates test those rival explanations.
- The strongest four-class fusion currently beats SigLIP2 by only seven net
  corrections. Do not turn that selected small edge into a strong causal claim.
- Full nine-class validation and the earlier four-class validation have already
  influenced recipe decisions. This phase selects **development finalists** only.
  Its results do not replace the retained published four-class test metrics.
- The nine-class test partly overlaps previously inspected four-class test rows;
  there is no wholly fresh nine-class confirmation claim. Auditing does not prove
  identity/session independence or absence of foundation-pretraining overlap.
- Original POLAR paper numerical/protocol equivalence remains unverified. Known
  person boxes are supplied; no person detector or action-detection AP is assessed.
- After this phase, freeze final recipes/refit budgets/comparisons before the
  fixed test evaluation. If none pass, retain the incumbent and complete that
  comparison instead of launching another validation search.

Initial review of `polar_bounded.py` found the finite candidate formulas and
promotion logic consistent with the declared protocol. Subsequent read-only
review covered the new model, trainer, queue and summarizer: original foundation
hashes, model-specific normalization, pretrained-pooler learning rate, frozen-prefix
evaluation mode, split-only admission, parent evidence bindings, resume integrity
and selected-checkpoint replay are explicit. The source-group bootstrap now
rejects and records draws omitting a class.

Independent CPU-only prelaunch checks passed **32 tests** covering the shared
protocol, model and training contracts. The review also identified a summarizer
field-name mismatch (`model_evidence` versus the trainer's `parameters`) that
would omit trainable-parameter counts; this was reported for correction before
the queue lock. It does not change model predictions or expand the experiment
budget. The production queue must additionally pass its four declared CUDA
engineering smokes. No score from the new 12 fits is claimed in this document.
