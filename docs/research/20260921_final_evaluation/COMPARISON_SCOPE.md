# Locked POLAR evaluation: what the comparisons establish

This phase evaluates a development-selected model; it does not conduct another test-set model search. **No new test score is asserted by this document.** The historical four-class result remains intact until separately reported paired evidence establishes the scope of any improvement.

## Cohorts and inference information

| Task | Locked development | Locked test | Scope |
| --- | ---: | ---: | --- |
| Four-class POLAR | 13,285 | 3,329 | Sitting, standing, walking, running; unchanged historical clean membership |
| Nine-class POLAR | 28,023 | 6,984 | All nine classes; independently restored/audited source data |

The official source has 35,324 images in nine posture classes. This is image/posture classification, not video recognition or bounding-box detection AP. The source is the authors' [POLAR release, version 1](https://data.mendeley.com/datasets/hvnsh7rwz7/1). Our audit found exactly one annotated person and one active action in every source record; this describes annotations, not a guarantee that every image contains only one visible human.

The audited nine-class set contains 35,007 images (21,057 train / 6,966 validation / 6,984 test) after quarantining 317 images in 154 cross-split source components. Of those exclusions, 125 were inherited from the prior four-class audit and 192 concern the added five classes; there are no new four-class exclusions. Exact/canonical source identity and verified perceptual-hash matches were checked. Exhaustive nine-class embedding-based duplicate discovery, subject IDs, session IDs, and foundation-model pretraining overlap are **not** established. Evidence: `.runs/benchmark_20260920_2250/data/polar9_data_audit.json` and its hash-bound manifests.

Input information includes image pixels and the annotated person box used for deterministic crop/context views. Thus a claim concerns **given-box posture classification**, not an end-to-end system that detects people itself. Action labels and source groups are never inference routing inputs. Source groups are used for split/calibration safeguards and resampling only.

The four-class test has been inspected historically. The nine-class test contains that previously inspected subset and is not wholly unseen. Freezing this phase prevents new selection on these test outcomes; it cannot retroactively produce a fresh independent holdout. Source-group confidence intervals do not establish subject-independent generalization.

## Fixed model panel and primary questions

Let `A` be the adapted DINOv2 three-seed probability mean for nine classes, or the frozen two-view DINOv2 RBF model for four classes. Let `F` be frozen SigLIP2 RBF and `S` the newly adapted SigLIP2 three-seed probability mean. The development-nominated candidate is fixed:

```text
prior incumbent       = 0.50 A + 0.50 F
replacement diagnostic = 0.50 A + 0.50 S
conservative nominee  = 0.50 A + 0.25 F + 0.25 S
```

Weights apply to probability vectors; no test-dependent gate, threshold, temperature, or class-specific weighting is fitted. The nominee remains the nominee even if a diagnostic model later scores higher.

Each task has exactly ten comparison candidates: the nominee, prior incumbent, replacement fusion, frozen two-view DINOv2 / SigLIP2 / DINOv3 / ConvNeXtV2 RBF models, adapted SigLIP2 / ConvNeXtV2, and either adapted DINOv2 (nine-class) or the retained historical ensemble (four-class). All train-plus-validation refits start from the pinned original foundation weights, use development-locked settings, and never select a checkpoint using test labels. Final neural epoch budgets are the median of the three relevant development best epochs; the original 20-epoch learning-rate schedule horizon remains unchanged.

| Predeclared primary comparison | Scientific question |
| --- | --- |
| Nominee vs prior incumbent, separately for four and nine classes | Does the new adaptation retain a net benefit after final refitting? |
| Nine-class nominee vs adapted DINOv2 | Does the selected complement add value beyond the strongest development standalone? |
| Four-class nominee vs frozen SigLIP2 | Is the selected fusion preferable to the strongest development standalone? |
| Four-class nominee vs original historical ensemble | Is there a paired improvement on the exact previously reported cohort? |

The historical ensemble's original prediction artifact is recovered by exact SHA-256 `0f96ecb7411abf8b3385004380a5fa001965113f0855015db4337fb76078c5f9`, aligned by image ID and checked for identical labels/class order. Its known macro-F1 is 0.9398833285904803 on 3,329 test images. It is not a newly refitted or validation-selected candidate.

There are **18** nominee-versus-reference comparisons: all nine references in each of the two tasks. The five primary contrasts above are identified in advance; the other thirteen are secondary. One global Holm correction covers all 18, including the overlapping four/nine cohorts. Overlap does not invalidate Holm's arbitrary-dependence protection, but it prevents treating the two tasks as independent replications. There is no separate favorable subset of p-values and no SOTA inference from an internal reference panel.

## Statistics and interpretation

- Macro-F1 uses every declared class. Accuracy, per-class precision/recall/F1/support, confusion, negative log-likelihood in nats, summed multiclass Brier score, and 15-bin ECE are reported. Adaptive/classwise ECE, confidence/margin/entropy, and fixed-coverage selective risk are descriptive diagnostics, not sources of a new rejection threshold.
- Paired, class-stratified row bootstrap and paired source-group bootstrap each use 5,000 draws. The source-group bootstrap resamples detected groups with replacement and rejects/counts draws omitting a declared class. Both are percentile 95% intervals, not simultaneous or selection-adjusted intervals. Fixed RNG seed is 20260921; the independent group-bootstrap stream uses 20260922.
- Macro-F1 paired randomization uses 10,000 groupwise model-assignment exchanges, a two-sided absolute-delta statistic, and `(extreme_draws + 1) / 10001`; identical predictions cancel without changing the null distribution. This tests exchangeability under the stated null at detected-source-group level, not independence of unknown subjects. All raw and Holm-adjusted p-values are retained.
- Rescue, harm, net corrections, error intersection/Jaccard, conditional error rates, and prediction disagreement describe what changed. They do not establish why a particular representation caused a rescue.
- All three matching-seed conservative/prior pairs (42, 52, 62) are retained. Nine-class pairs use the same matching adapted DINO seed on both sides; four-class priors are identical frozen models. Seed deltas are sensitivity evidence on the same test images, not three independent test samples or an opportunity to choose a favorable seed.
- Recorded runtime costs are reported only with their actual scope. Session time can omit pre-resume work; panel inference time can include cache replay and omit feature extraction. Neither is silently relabeled as end-to-end deployment latency. Missing costs are not zero costs.

For each task, the fixed nominee passes the predeclared promotion checks against its prior incumbent only if **all** hold: positive macro-F1 delta; positive net corrections; source-group macro-F1 delta interval lower bound above zero; global Holm p below 0.05; no class F1 decline exceeding 1 percentage point; NLL increase at most 0.02; Brier increase at most 0.01; and positive F1/net corrections in every matching seed pair. The NLL/Brier tolerances permit small deterioration and must not be described as proof of calibration improvement.

These gates evaluate a fixed nomination; they do not automatically publish or replace an artifact. Failure means report the failure and preserve the incumbent. Passing every other reference contrast is not required for valid metrics or incumbent-relative acceptance. Superiority relative to the historical ensemble or a standalone reference is separately reported under the fixed family; it is not inferred from an incumbent-relative pass alone.

Implementation: [statistical core](../../../src/hac/polar_locked_statistics.py), [analysis entry point](../../../experiments/analyze_polar_locked.py), [synthetic tests](../../../tests/test_polar_locked_statistics.py), and [selection lock builder](../../../experiments/prepare_polar_locked.py). The CLI verifies the lock, all-fit barrier, candidate family, source-group hashes, exact IDs/labels, probability bytes, fixed fusion formulas, and seed-mean reconstruction before analysis. Reports list candidates in the declared order, not a test-chosen ranking. Completion requires `POLAR_LOCKED_STATISTICS_COMPLETE` and hashed report artifacts.

## External baseline status: no unsupported state-of-the-art claim

The original paper is Ma and Liang, [POLAR: Posture-level Action Recognition Dataset, ICSAI 2019](https://doi.org/10.1109/ICSAI48974.2019.9010160). On 2026-09-21, a bounded search checked the DOI, the [official IEEE record](https://ieeexplore.ieee.org/document/9010160), author/title manuscript searches, and available primary dataset material. IEEE returned an access/JavaScript verification page; no legally accessible paper text with a baseline table was recovered. Search failure is **not** evidence that no stronger result exists.

The official dataset page establishes the nine-class taxonomy and image count, not a leaderboard or a numerical classification baseline. Consequently the original paper's model, split details, metric definitions, numeric results, use of person boxes, and external training data remain unverified. No paper number is fabricated or inferred from a secondary snippet. Broader multimodal benchmarks and adaptation/novelty-detection studies citing POLAR do not automatically supply matching nine-class closed-set classification results.

| Unknown | Evidence required before a stronger comparison claim |
| --- | --- |
| Original paper's numeric baseline and protocol | Obtain an authorized full text/authors' manuscript and extract the exact table, cohort, input, metric, and training protocol |
| Direct comparability to the original, unquarantined split | Reproduce its documented model on both official and audited cohorts, with exclusions fixed before scores; report both rather than mix protocols |
| Strongest independent POLAR9 competitor | Verify primary paper/official code and comparable class set, split, given-box policy, external data, and metric; reproduce if feasible |
| Unobserved identity or pretraining overlap | Obtain provenance or independent new-source evaluation; source hashes alone cannot establish absence |
| Generalization beyond an exposed benchmark | Evaluate the frozen nominee on a genuinely new, independently collected/source-separated holdout without selecting it there |
| Full deployment efficiency | Measure fixed hardware, batch size, preprocessing, box acquisition, all component feature extraction, and p50/p95 end-to-end latency |

The defensible claim, if the recorded evidence supports it, is a **paired improvement over specified internal baselines on the stated audited POLAR cohort**. A small corrected p-value is not proof of novelty, real-world robustness, label correctness, or global state of the art.
