# POLAR competitive benchmark: execution plan

Status: new development phase, not a new achieved result. The retained four-class
93.99% macro-F1 / 94.56% accuracy remains unchanged. Work is isolated on
`research/polar-benchmark-20260920`; private outputs live in
`.runs/benchmark_20260920_2250/`. No test predictions are opened by the current queue.

## Evidence and scope

The [evidence review](EVIDENCE_AND_COMPARATORS.md), [mind map](mind_map.md) and
[machine-readable graph](knowledge_graph.json) connect the previous gains and
failures to this phase. Frozen multilayer DINOv2 plus a nonlinear classifier was
the strongest standalone POLAR component; complementary probabilities improved
the ensemble. Later V-COCO work motivates testing SigLIP complementarity, not
assuming a newer backbone wins. These are different datasets, not shared scores.

The [machine-readable protocol](../../../experiments/polar_benchmark_protocol_20260920.json)
declares candidate budgets and evidence boundaries before new model scores.

| Cohort | Train | Validation | Test | Role |
| --- | ---: | ---: | ---: | --- |
| Historical audited four classes | 9,958 | 3,327 | 3,329 | Unchanged comparison population |
| Released original nine classes | 21,194 | 7,065 | 7,065 | Original membership/provenance; not silently mixed with cleaned results |
| New source-audited nine classes | 21,057 | 6,966 | 6,984 | Primary new development/evaluation population |

All 35,324 annotations/images were validated. Each has one person and one active
action; image decoding, dimensions and boxes pass. The source audit quarantined
317 images in 154 components, including all 125 historical exclusions. No new
exclusions affect the historical four-class cohort. Source-group auditing is not
proof of subject/session independence; no full nine-class embedding retrieval
audit has yet run. The nine-class test includes the previously inspected four-class
subset and is **not wholly unseen**.

## B–E, with an explicit completion boundary

| Phase | Work | Gate |
| --- | --- | --- |
| B1: challenger screen | DINOv2 multilayer reference, DINOv3-B, SigLIP2-B, ConvNeXtV2-B; optional DINOv3-L; identical label/split access | Pinned weights and image/cached-feature integrity; no substitution of unavailable models |
| B2: adaptation | Nine-class DINOv2 top-four-block reference versus person-preserving view recipe; seed 42 then locked recipe seeds 52/62 | CUDA smoke, finite gradients, selected-checkpoint replay |
| B3: serious adapted challengers | Bounded modern-backbone adaptation after B1; include a credible adapted convolutional comparison | Frozen-only scores cannot establish a complete modern ranking |
| C: fairness | Crop/full-frame/fused controls, declared pretraining and resolution, training-only fitting, exact cohorts | No implicit extra input privileges; compute-matched and unrestricted comparisons separated |
| D: fixed evaluation | Final selection/refit lock before batch test evaluation | No test-based threshold, fusion or model selection; old-test exposure disclosed |
| E: confidence | Paired differences, corrected comparisons, seed variance, calibration, per-class failures, runtime and source-group sensitivity | Positive lower paired interval against the strongest prespecified comparable challenger; otherwise inconclusive |

The **currently launched queue performs B1, the initial B2, and preparation for C**.
It ends with `DEVELOPMENT_COMPLETE_REVIEW_REQUIRED`, not a SOTA verdict. B3 and D–E
require the development evidence and a separate final-selection lock. This prevents
an unattended process from turning test feedback into more model selection.

## Fixed first-stage choices

- Shared CUDA extraction over 28,259 official train/validation images; supervised
  heads use only their audited task-specific rows and labels.
- Two views per representation: full frame and person plus 10% context. Their
  concatenation tests complementarity without a learned annotation-only router.
- DINOv2 keeps the historical resize/center-crop control. Other representations
  use recorded reference processors. Any reconstructed processor is explicitly
  labelled as recovered configuration, never original provider bytes.
- Balanced CUDA logistic heads: `C={.001,.01,.1,1}`; the historical strong-regularization
  settings are included. One fixed fused-view RBF (`C=10`, gamma `1/d`) per family,
  with five source-grouped training-only calibration folds.
- RBF distance computation uses CUDA; the SVM solver/calibration uses CPU. We do
  not call this a fully GPU-native SVM. Synthetic CUDA/CPU numerical parity tests pass.
- Nine-class adaptation starts from the original foundation checkpoint. Using old
  final models would contaminate validation because those models saw train+validation.
- Two seed-42 view recipes: historical mild crop versus a padded person-preserving
  recipe. Hypothesis: preserving complete body shape helps bending, lying, stretching
  and jumping. It is tested, not assumed. The winning validation recipe is locked
  before the other two seeds. The alternate recipe remains in the ledger.
- At most two simple probability fusions are screened: uniform all-family winners
  and uniform top-two development candidates. Their validation scores are selected
  estimates, not confirmatory performance.

## CUDA and unattended operation

The previous workspace interpreter was CPU-only. A separate `.venv` now uses
PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 and Transformers 5.5.3. A real CUDA
operation passed on the NVIDIA RTX PRO 3000 Blackwell laptop GPU (12 GB).
The transferred DINOv3-B weights are reused only after matching their recorded
historical SHA256. DINOv3-L is optional, not silently replaced if unavailable.

CPU source restoration, auditing and checkpoint preparation run concurrently.
Heavy GPU jobs are serialized; two independent training jobs do not compete for
12 GB VRAM. Extraction is chunk-resumable. Training resumes at epoch boundaries
with request, optimizer and selected-checkpoint integrity checks. Failures stop
the queue and record evidence; no automatic best-of-retries selection occurs.

```powershell
# The active plan includes one documented pre-training cache-check fix.
.venv\Scripts\python.exe experiments/run_polar_benchmark_queue.py `
  --run-dir .runs/benchmark_20260920_2250 --plan-name queue_plan_02.json `
  --supersedes queue_plan.json --write-plan

# Start/restart under a hidden detached supervisor; see LAUNCH_STATUS.md.
.venv\Scripts\python.exe experiments/run_polar_benchmark_queue.py `
  --run-dir .runs/benchmark_20260920_2250 --plan-name queue_plan_02.json
```

Monitor on request, not by keeping an assistant polling:

- `queue_status.json`: current command, PID, stage, elapsed time and log.
- `queue_receipts.json`: completed/skipped jobs and integrity hashes.
- `logs/`: per-job logs.
- `polar9/adaptation/*/progress.json`: batches, epoch throughput and ETA.
- `features/*/*/progress.json`: completed extraction chunks.
- `development_summary.json`: written only when the development queue finishes.

The original plan and failed startup log are preserved. The sole amendment narrowed
the local Hugging Face completeness check to the configuration and safe weight file
actually consumed; it had incorrectly required an unused alternate pickle file and
repository documentation. No training or performance score preceded that correction,
and no model-selection or safety gate changed. See `engineering_amendment_01.json`.

The run is expected to take hours rather than minutes; a reliable ETA requires
real throughput from extraction and the first full training epoch. Maximum
per-job deadlines are watchdog limits, not runtime forecasts. Long jobs run
unattended. A `STOP_AFTER_CURRENT_JOB` file requests a graceful queue stop.

## What would change our conclusion?

A stronger modern challenger means the old ensemble is no longer best in the
expanded panel. A tie means competitive, not superior. A higher nine-class score
on a different cleaned population does not beat an original-paper score by itself.
No numeric original-paper SOTA threshold has yet been verified. The nine-class
work can establish a strong reproducible benchmark before that comparison is resolved.

The final evaluation must preserve all prespecified competitors and report paired
95% macro-F1 intervals, Holm-controlled comparison tests, class-level metrics,
accuracy, NLL/Brier/ECE, rescue/harm transitions, seed dispersion and cost. If new
development is driven by inspected test outcomes, genuinely untouched confirmation
data is needed; reshuffling already used images does not restore independence.
