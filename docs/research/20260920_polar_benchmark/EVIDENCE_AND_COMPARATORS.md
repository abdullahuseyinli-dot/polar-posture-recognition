# POLAR benchmark expansion: evidence and comparator audit

Date: 2026-09-20. Status: **data audit complete; pre-result engineering review**, not a
claim of new performance. This document keeps historical results, proposed tests,
and unresolved external comparisons separate. The executable run protocol is the
authority for the configuration actually launched; the recommendations here must
not be silently described as completed experiments.

## 1. What the existing experiments establish

| Observation | Evidence | What it supports; what it does not |
| --- | --- | --- |
| Four-class locked ensemble: **93.9883% macro-F1, 94.5629% accuracy**, 3,329 images | [Test metrics](../../../results/polar_test_metrics.csv) | Strong retained four-class result; not a nine-class result or independently established SOTA |
| Ensemble minus strongest standalone multilayer RBF: **+1.2529 pp**, paired 95% interval **[+0.6531, +1.8619] pp** | [Paired uncertainty](../../../results/polar_test_uncertainty.json) | Complementary predictors improved the retained system on the old test; interval does not include all seed or source-dependence uncertainty |
| Same 1,536-dimensional frozen DINOv2 feature family: RBF **92.6200%** vs logistic **91.4863%** validation macro-F1 | [Classifier screen](../../../results/polar_classifier_validation.csv) | A nonlinear head is a serious inexpensive comparator; not proof RBF always dominates every representation |
| Final 7,680-dimensional multilayer RBF **92.7354%** vs multilayer logistic **92.5753%** on test | [Test metrics](../../../results/polar_test_metrics.csv), [architecture](../../ARCHITECTURE.md) | Retain both linear and nonlinear controls; different feature screens must not be pooled as one matched comparison |
| Frozen-probe validation curve: **84.8651% to 91.5020%** for 242 to 9,958 training images | [Scale summary](../../../results/polar_scale_summary.csv) | Using available labeled data materially helped; nested deterministic subsets are not independent seed replications |
| RBF/adapted-DINOv2-B validation disagreement **6.5524%**, with **112** rows uniquely correct for RBF and **97** uniquely correct for adapted DINO | [Complementarity](../../../results/polar_validation_complementarity.csv) | Error diversity motivates fusion; the pair's 96.844% correctness oracle is diagnostic only, not an achieved classifier |
| DINOv2-B top-four adaptation **92.40%** vs full adaptation **92.45%** validation F1 at seed 42; partial fit about **20.9% faster** | [Historical report, section 5.2](../../POLAR_PUBLIC_REPORT.md) | Partial adaptation is a sensible resource control, not a universal best recipe |
| Matched V-COCO DINOv2/DINOv3/SigLIP2 representation F1: **83.9495 / 83.6713 / 83.5787%** | [Nested development metrics](../../../results/vcoco_v3/source_tag_development_metrics.csv) | A newer backbone alone did not automatically improve this separate domain |
| Separate V-COCO nested DINO+SigLIP reliability stack **86.9682%** vs DINO factorized stack **85.6010%** | [Nested development metrics](../../../results/vcoco_v3/source_tag_development_metrics.csv) | Complementary representation fusion is worth testing on POLAR; these are not POLAR scores and not a newly opened external test |
| Source-only external performance falls sharply; annotation-only geometry gets **43.0967%** POLAR macro-F1 but **30.7344%** on mapped V-COCO | [Geometry diagnostic](../../../results/polar_exploratory_geometry_baseline.csv), [external analysis](../../POLAR_PUBLIC_REPORT.md) | Geometry and background associations can be dataset-specific. Do not assume in-domain performance demonstrates domain robustness |
| Later ARFTR corrections mostly failed promotion despite some positive point estimates | [Companion evidence graph](https://github.com/abdullahuseyinli-dot/arftr/blob/main/results/arftr_development/knowledge_graph.json) | Methodological warning against promoting tiny selected gains. ARFTR's video score is not a POLAR baseline |

**Inference:** the best-supported transfer of engineering knowledge is to establish
strong frozen representations, nonlinear heads, and measured complementary fusion
before expensive adaptation. None of the evidence predicts a particular nine-class
score, and adding five labels need not increase macro-F1.

## 2. What we are comparing against

### Existing directly comparable four-class results

The retained ensemble and its five components share the same 3,329 test rows.
Macro-F1 is 93.9883% for the ensemble, 92.7354% for multilayer RBF, 92.5753% for
multilayer logistic, 92.5183% for adapted DINOv2-B, 91.3082% for adapted DINOv2-S,
and 89.1450% for adapted ConvNeXt-S. These are our verified implementations, not an
external leaderboard. The original blend's components also differ in input view
and adaptation scope; its ranking alone is not a pure backbone comparison.

### External evidence checked on 2026-09-20

| Source | Verified scope | Comparator status |
| --- | --- | --- |
| [Original POLAR dataset, Mendeley v1](https://data.mendeley.com/datasets/hvnsh7rwz7/1) | Authors' released dataset: 35,324 images, nine posture categories | Primary dataset authority; landing page contains no numerical method leaderboard |
| [Ma and Liang, ICSAI 2019](https://doi.org/10.1109/ICSAI48974.2019.9010160) | Original paper identified; IEEE page inaccessible behind browser verification during this audit | **Original numerical results, metric definition, preprocessing and precise evaluation protocol remain unverified** |
| [CAPA-AI accepted manuscript](https://uhra.herts.ac.uk/id/eprint/26039/7/ROMAN_2025_Novelity_.pdf), sections IV-C and V | Uses POLAR among datasets for novelty detection/adaptation; reports 89% simulation novelty/adaptation accuracy and 80% real-world adaptation accuracy | Not closed-set nine-class POLAR macro-F1; neither a score to beat nor evidence that our method wins |
| [MMT-Bench paper](https://openreview.net/attachment?id=R4Ng8zYaiz&name=pdf) | Searchable primary-paper record references POLAR in a broader multimodal benchmark; full text was not retrievable in this audit | No verified, same-split, same-metric POLAR competitor result extracted; do not substitute an aggregate multimodal score |

The bounded recovery attempt checked the DOI/IEEE endpoint, exact-title searches,
local PDF inventories in both project workspaces and Downloads, and the original
`HAC_Research_Continuation_20260906.zip` entry list. The continuation archive has
48 entries and no matching original-paper PDF. No numeric original-POLAR baseline
was recovered. This is an **external-comparability limitation, not a blocker to
running our own locked nine-class benchmark**. Do not label that benchmark an
exact reproduction of the original paper until its protocol is verified.

If the original paper becomes available, record its exact table/page, task unit,
splits, person-box access, metric, training sources and method settings. Compare
only matching quantities. Report original-split and audited-split results in
separate tables where their memberships differ.

## 3. Bounded challenger panel and nine-class progression

| Priority | Candidate/test | Evidence-based purpose | Required control |
| --- | --- | --- | --- |
| 1 | Frozen DINOv2-B two-view multilayer, linear and RBF heads | Reconstruct the proven strong reference for nine classes | Train-only transforms and calibration; exact feature provenance |
| 2 | DINOv3-B, SigLIP2-B and ConvNeXt-V2-B frozen features | Test modern representation, supervision and convolutional alternatives | Same source rows, person-box access and declared view budget; publish representation dimensions and pretraining differences |
| 3 | Best development-selected individual from each family, plus a simple fixed or development-selected probability blend | Test whether an advantage comes from complementary errors rather than a larger sweep | Base predictions must be out of fold for learned stacking; validation-only blend selection |
| 4 | Bounded partial/full adaptation of genuinely competitive families | Avoid claiming superiority over deliberately weak frozen competitors | Fresh foundation initialization or verified train-only checkpoint; equal declared search opportunity and three stochastic seeds |
| 5 | Optional DINOv3-L frozen capacity challenge | Determine whether a larger off-the-shelf representation erases the claimed engineering benefit | Separate unrestricted-capacity table, resource measurement and verified weight access; no fabricated fallback model |

Official model sources: [DINOv2](https://github.com/facebookresearch/dinov2),
[DINOv3](https://github.com/facebookresearch/dinov3),
[SigLIP2 paper](https://arxiv.org/abs/2502.14786), and
[ConvNeXt V2 implementation](https://github.com/facebookresearch/ConvNeXt-V2).
Their general benchmark claims are not evidence of POLAR scores. Pin exact
checkpoint revisions and record licenses/access before launching each candidate.

**DINOv3-B access recovery:** the feature specialist located the transferred,
previously authorized foundation snapshot at revision
`5931719e67bbdb9737e363e781fb0c67687896bc` in the local `hac_pretrained` cache and
verified the 342,662,192-byte weight file SHA256
`9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b` and config SHA256
`3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec`.
This resolves the base model's weight availability without substituting a model.
The transferred archive did **not** contain the original processor JSON. Its
declared recovery uses the pinned
[Transformers 5.5.3 DINOv3 processor defaults](https://github.com/huggingface/transformers/blob/v5.5.3/src/transformers/models/dinov3_vit/image_processing_dinov3_vit.py)
(224-square, bilinear, ImageNet normalization), with a separate restoration receipt
and provenance. This is **not original processor-byte parity** and is not the
256-pixel example transform in Meta's general README. Weight verification does
not constitute a training result or establish availability of DINOv3-L.

Run the same development procedure separately for four and nine labels. Feature
extraction from the same frozen foundation checkpoint can be shared by image/view
identity; labels, heads, selection locks and metric tables cannot be merged.
Prioritize the nine-class reference and challengers rather than spending the
whole budget replaying already known four-class experiments.

On one GPU, concurrent independent heavy GPU jobs usually compete for memory and
throughput. Use a single explicit CUDA queue with resumable per-model caches;
parallelize source auditing and bounded CPU head fits when resources permit. A
model access failure must be recorded, not silently replaced with a different
backbone. Calibration and scikit-learn head fitting may run on CPU; report that
truthfully even when encoder extraction and neural adaptation use CUDA.

## 4. Protocol and safety requirements before trusting a ranking

### Annotation and split validity

- **Verified on 2026-09-20:** the restored original annotation archive contains
  35,324 JSON records. An exhaustive independent read found exactly one person
  and exactly one active action in every record. The source action counts are
  `bendover=3867`, `jump=3562`, `lying=3581`, `run=3685`, `sit=5101`, `squat=4253`,
  `stand=4656`, `stretch=3322`, and `walk=3297`. Display names are bending, jumping,
  lying, running, sitting, squatting, standing, stretching, and walking.
  Evidence: local restored
  `.runs/benchmark_20260920_2250/data/source/archives/annotations.zip`, SHA256
  `78d18e643b119cb081506c811a0042ba212eabb67fa18eacdc161d8ed5fe26e2`.
  This resolves the single-target/one-hot schema question. The subsequently
  completed image audit recorded zero decode failures, dimension mismatches or
  invalid/clipped boxes; it does not establish identity independence or
  original-paper comparability.
- The inherited `polar.py` parser recognizes only four action keys. The new
  parser must continue to enforce the verified nine-action schema, rather than
  silently taking the first person or first positive class if a source changes.
- The released split loader expects train/val/test counts 21,194 / 7,065 / 7,065.
  Preserve source membership as provenance even when an audit excludes rows.
- **All-nine-class source audit completed before new fitting:** 35,007 images
  remain: **21,057 train / 6,966 validation / 6,984 test**. There are 317
  quarantined images in 154 components: the historical 125 exclusions plus 192
  additional images from the five newly included classes. The audit records 68
  exact/canonical source-identity edges, 313 cross-split pHash candidates and 163
  correlation-confirmed pairs. Evidence: local
  `.runs/benchmark_20260920_2250/data/polar9_data_audit.json`; clean-manifest SHA256
  `5327726cf1c8aabefd08e14e3161dc3e3fa38aaa5173f614a529e8dbc5e3c4a5`.
- **The retained four-class cohort is unchanged:** 9,958 train / 3,327 validation /
  3,329 test, with zero additional exclusion flags. Its historical metrics remain
  intact. Future additional exclusions, if ever established, would require a
  separately named cohort and matched baseline reevaluation rather than silently
  changing the existing comparison.
- The completed audit covers exact bytes, canonical source names and cross-split
  perceptual hashes with correlation confirmation, and inherits earlier
  embedding-confirmed four-class exclusions. **A fresh all-nine-class embedding
  retrieval audit has not run.** Same-split non-identical near duplicates are not
  exhaustively grouped. Detected source components are not identity/session IDs.
- Hash data manifests, class order, exclusions and source groups. Subject/session
  independence remains unknown without corresponding identifiers. No source-audit
  threshold establishes that foundation pretraining did not contain these images.

### Exposure and initialization

- **The historical four-class test has already been inspected.** New fixed
  comparisons on it are useful, but it is not a fresh hidden test.
- The original nine-class test includes old four-class test images. Its aggregate
  is therefore **partially historically exposed**, even before a nine-class
  classifier is fitted. Never claim the whole nine-class test is newly unseen.
- New-class-only evaluation is a secondary subset diagnostic, not a replacement
  nine-class headline. It must not be used to adaptively choose the winner.
- The historical final adapted POLAR checkpoints were trained on train plus
  validation. Reusing those weights for a nine-class development run while
  selecting on official validation would contaminate selection. Start from
  foundation weights or prove that a reused checkpoint saw train only.
- Frozen foundation feature caches are reusable only if row identity, checkpoint,
  layer aggregation, view, resolution and numerical representation match. Cached
  features from an adapted encoder inherit that encoder's training exposure.

### Model selection, calibration and metric calculation

- Standardization, PCA, feature selection, calibration and class weighting are
  fitted using the applicable training portion only. Inner folds respect source
  groups. Do not fit a scaler on the whole dataset because it is unsupervised.
- Learned fusion sees cross-fitted base predictions, not in-sample predictions.
  No router, blend weight, threshold, checkpoint or candidate is selected using
  test labels or historical test error identities.
- Lock candidates, maximum search budget, tie rule and metrics before new test
  evaluation. Refit train plus validation only after development selection ends.
  Any missing/failed candidate and any numerical exception remains visible.
- Report macro-F1, accuracy, per-class precision/recall/F1, confusion, NLL, summed
  multiclass Brier, ECE and cost. Fix the complete label set when computing F1;
  do not silently drop an unpredicted class.
- Use paired source-group-aware uncertainty where grouping is available. Keep a
  secondary row-stratified interval for comparability with the old study, clearly
  identified. Handle bootstrap resamples missing a class explicitly and record
  rejected draws. Do not present variation across deterministic repeated fits as
  independent training-seed uncertainty.
- Record every prespecified pairwise comparison and adjust for multiplicity,
  rather than highlighting the best unadjusted p-value after inspection.

### Gates and interpretation

1. **Integrity gate:** all source/schema checks, disjoint membership assertions,
   cache alignment, finite probabilities and artifact hashes pass. Failed gates
   stop the affected run; no silent CPU/random-weight fallback.
2. **Development gate:** a complex candidate must improve over the strongest
   prespecified reference on validation with a declared practical margin or
   justify a declared cost/calibration tradeoff. Selection is not itself evidence
   of generalization.
3. **Confirmation gate:** stronger-than-reference language requires a positive
   paired, multiplicity-aware interval on a fixed evaluation, consistent seed
   behavior, and disclosure of previous exposure. An interval crossing zero means
   competitive/uncertain, not proven improvement.
4. **SOTA gate:** verify strongest credible external competitors under the same
   task, split, inputs and metrics. Winning this finite panel supports that panel's
   scope; it does not prove global superiority over every untested method.
5. **Stop gate:** do not add another sweep after opening new test results. Any
   architecture development motivated by those results needs separate independent
   confirmation before a fresh generalization claim.

## 5. Open claims and decisive evidence

| Unknown | What resolves it |
| --- | --- |
| Original nine-class paper's numerical baseline and full protocol | Obtain legal full text/author artifact and transcribe the exact comparison with page references |
| Nine-class single-target, single-label schema | **Resolved:** exhaustive restored-archive read of all 35,324 records; enforce this invariant in the new parser |
| Full nine-class source overlap | Initial all-class byte/name/pHash audit completed; all-class embedding retrieval and exhaustive same-split near-duplicate grouping remain additional checks, not established guarantees |
| Whether DINOv3/SigLIP2/ConvNeXt-V2 beat our four-class system | Prespecified same-row challenger evaluation with defensible adaptation budget |
| Whether prior fusion gains transfer to nine classes | Development-selected fusion versus strongest individual, followed by fixed comparison |
| Whether extra classes need learned posture hierarchy | First inspect nine-class development confusion; only then predeclare a hierarchical control and an independent confirmation strategy |
| Whether a gain is view, architecture, head or capacity driven | Matched view/head controls and a separate unrestricted-resource comparison |
| True identity-disjoint and pretraining-disjoint generalization | Appropriate identity/session metadata and pretraining corpus audit, or an independently sourced confirmation dataset |
| Whether nine-class F1 will exceed four-class F1 | No valid prediction from current evidence; measure, but do not rank unlike class taxonomies |

No new performance result is claimed in this document. Preserve the existing
four-class release and its hashes throughout the expansion.

## 6. Initial implementation review

The independent review inspected the new data, feature, head and adaptation
modules plus the development screen and training entry points before a new
benchmark score was produced. The following are implementation findings, not
performance findings:

- Development loaders reject test rows and require every declared class in both
  train and validation. Task manifests supply labels independently of the shared
  feature pool. The nine-class mapping preserves the original four label indices.
- The feature loader uses pinned foundation checkpoints, CUDA-only extraction,
  atomic hashed chunks and source-image verification. Review identified that
  the data audit used `sha256` while the feature loader initially expected
  `image_sha256`; the feature owner added a validated alias and a conflict test.
- The planned logistic screen was expanded to include the historically justified
  `C=0.001` and `C=0.01`, alongside `0.1` and `1.0`. This avoids judging new
  representations only under much weaker regularization than the retained probe.
- The new RBF reference calculates its kernel on CUDA, with CPU libsvm and
  sigmoid calibration. The scaler lives inside the group-cross-validated
  estimator, preventing calibration-fold standardization leakage.
- New DINOv2 adaptation starts from original foundation weights, not historical
  train-plus-validation adapted checkpoints. Validation selects checkpoints;
  these runs are development experiments, not confirmation results.
- **Cache-integrity review fixes verified in source:** the core now enforces
  completed-cache status, contract hash, CUDA device type, full development-only
  scope, requested model/view, declared row/feature hashes and matching audited
  source-image hashes. A CPU smoke cache cannot enter the production ranking.
- **Queue and nested-artifact review fixes verified in source:** the supervisor
  verifies audit-bound artifact bytes before every job; completed screen summaries
  bind every child head and prediction artifact, and the analysis verifies the
  selected child before loading its predictions. These close the previously
  identified opportunity for modified files to be accepted merely by hashing
  their current bytes.
- **Selected-checkpoint replay implemented:** resumed training checks selected
  artifact sidecars; each completed training run reloads its best checkpoint and
  requires exact validation labels/predictions with maximum probability
  difference at most `1e-5`. A completed benchmark run must contain its own
  passing replay receipt; implementation is not a claim that all training runs
  have already passed.
- Independent prelaunch tests in the new POLAR environment passed **65 tests**,
  including CUDA RBF/reference parity and nine-class grouped calibration. Runtime
  was PyTorch `2.11.0+cu128`, CUDA `12.8`, with CUDA available. These are engineering
  checks, not new POLAR performance results.
- A frozen challenger screen plus a DINOv2 adaptation reference does not complete
  a fair adapted-challenger comparison. Historical ConvNeXt-S improved from
  80.59% validation F1 with a head-only fit to 89.20% with full adaptation. A weak
  frozen ConvNeXt-V2 result alone would not justify eliminating that family from
  the later serious competitor panel.

The current queue should stop at development evidence. Final locks, simple fusion,
bounded adapted challengers and fixed comparative evaluation remain explicit
later stages; they must not be advertised as already completed.
