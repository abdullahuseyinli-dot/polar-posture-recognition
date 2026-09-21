# One bounded adaptation phase

Date: 21 September 2026. This is a development protocol, not a new achieved result.
The previous queue is complete; this phase must finish before the final evaluation
and comparison lock. No test evaluation is part of its commands.

## Verified starting point

The parent run is `.runs/benchmark_20260920_2250/`. Its 21 completed job-marker
hashes, 13 candidate probability arrays, row/label alignment, metrics, declared
fusion arithmetic and three adapted-checkpoint replay records were rechecked.
The new parent lock binds 104 artifacts, including unchanged audited manifests.

| Task | Incumbent validation macro-F1 | Accuracy | Errors | Components |
| --- | ---: | ---: | ---: | --- |
| Nine classes | 94.00454% | 94.14298% | 408 / 6,966 | Equal adapted DINOv2 three-seed mean and frozen SigLIP2 |
| Four classes | 95.15074% | 95.61166% | 146 / 3,327 | Equal frozen DINOv2 and SigLIP2 |

These are validation incumbents, not replacements for the historical four-class
93.98833% test result. See the [evidence review](EVIDENCE_REVIEW.md),
[mind map](mind_map.md) and [graph](knowledge_graph.json).

## Fixed budget and mechanism

Twelve production fits, no sweep: two models, two tasks, three seeds (42/52/62).
Four small engineering smokes precede them, followed by one finite comparison job:
**17 supervisor jobs in total**. Nine-class fits run first.

| Model | Adaptation scope | Nine-class trainable parameters | Purpose |
| --- | --- | ---: | --- |
| SigLIP2-B | Last four vision blocks, final normalization, pretrained attention pooler, new classifier | 35,447,049 | Transfer the successful person-preserving recipe while testing whether complementary information survives |
| ConvNeXt V2-B | Last encoder stage, final normalization, new classifier | 27,475,977 | A matched-budget convolutional adaptation control, not a compulsory ensemble ingredient |

Both start from their exact hash-verified original foundation weights. Historical
train-plus-validation checkpoints cannot initialize these development fits.

All fits use the same provided person box plus 25% context, aspect-preserving
square padding, 224-pixel input and fixed person-safe mild augmentation. Padding
and normalization use each backbone's official channel statistics. Geometry is
the declared new person-preserving recipe, not claimed official-processor parity.

Fixed optimization: unweighted cross-entropy; AdamW; new-classifier LR 0.001;
backbone LR 0.000005; weight decay 0.0001; dropout 0.1; gradient clip 1; batch 16
with accumulation 4; warmup 10% then cosine schedule; at most 20 epochs; minimum
three; patience four. Selection uses validation macro-F1, then NLL. There is no
smoothing, mixup, seed selection, extra resolution, or test-time adaptation.

Frozen prefixes stay in evaluation mode and receive no gradients. Pretrained
SigLIP attention-pooler weights receive the backbone LR, not the classifier LR.
CUDA is mandatory; BF16 is used when supported, otherwise FP16 with scaling.
Epoch budget is matched; FLOPs, pretraining data and parameter counts are not.

## Five candidates per task, including no change

Let `I` be the unchanged incumbent and `A` its DINOv2 component. For nine classes
`A` is the prior adapted three-seed ensemble; for four classes it is frozen DINOv2.
Let `S` and `C` be the new equal-probability three-seed SigLIP and ConvNeXt models.

1. Retain `I` unchanged.
2. `S` alone.
3. `C` alone.
4. Replacement fusion: `0.5 A + 0.5 S`.
5. Conservative fusion: `0.5 I + 0.5 replacement`.

The conservative fusion therefore keeps DINOv2 at 50%, frozen SigLIP2 at 25%,
and adapted SigLIP2 at 25%. Only these two new fusion recipes are tested. There
is no grid of weights, learned router, confidence threshold or ConvNeXt fusion.

## Promotion and stop rules, fixed before new production scores

A candidate may be nominated over the incumbent only if all these hold:

- At least **+0.20 percentage points macro-F1** and positive net corrections.
- Every seed-specific counterpart improves macro-F1 and net corrections while
  keeping the old anchor/incumbent fixed; all three seeds must be retained.
- No individual class loses more than 1 percentage point F1.
- NLL increases by no more than 0.02; summed Brier increases by no more than 0.01.
- All image/data/source integrity checks and checkpoint replays pass.

Report every attempted candidate, every failed gate, seed variance, per-class
metrics, confusion, NLL/Brier/ECE and rescue/harm transitions. The 1,000-draw paired
row-stratified and source-group bootstrap intervals are descriptive because the
validation set has already influenced development; they do not establish significance
after selection or SOTA. Probability-quality deltas remain visible even within the
allowed margins.

Among eligible candidates choose macro-F1, then NLL, then lexical name. If none
passes, **retain the incumbent and stop development**. No automatic extra trial or
favorable-seed retry follows a disappointing result. The convolutional candidate
is still reported as a comparator even when it fails promotion.

## Execution and safety

Branch: `research/polar-bounded-adaptation-20260921`.
Run: `.runs/bounded_adaptation_20260921_0406/`.
Executable [protocol](../../../experiments/polar_bounded_protocol_20260921.json).

```powershell
.venv\Scripts\python.exe experiments/run_polar_bounded_queue.py `
  --parent-run .runs/benchmark_20260920_2250 `
  --run-dir .runs/bounded_adaptation_20260921_0406 --write-plan
```

The actual launch is hidden/detached; see [launch status](LAUNCH_STATUS.md) for the
verified receipt. For a foreground restart only when no supervisor is running:

```powershell
.venv\Scripts\python.exe experiments/run_polar_bounded_queue.py `
  --run-dir .runs/bounded_adaptation_20260921_0406
```

The proven single-GPU supervisor is reused unchanged. New run outputs cannot be
nested within or overwrite the parent run. Source/protocol/parent artifact hashes
are checked before every job; altered requests cannot silently resume. A process
lock prevents duplicate launches to this run. All fits are mandatory; unavailable
weights or any failed job stop the queue, preserving logs. A three-hour per-fit
watchdog is a safety ceiling, not a forecast. Smokes have 15-minute ceilings.

Keep the machine powered and ventilated. Expect several hours; the planning
estimate is roughly 6-9 hours using the previous DINOv2 throughput, not a measured
forecast for these new backbones. Early stopping can reduce this. GPU fits are
serialized to avoid contention. Long training is left unattended, not actively
polled by the assistant. Status/logs are durable for the next user-requested check.

`STOP_AFTER_CURRENT_JOB` inside the run directory requests a graceful stop.
Do not delete process locks without confirming the recorded process has exited.
Never change locked training source while this queue is active.

## Boundary after completion

The queue writes `bounded_summary.json` and stops at development review. It does
not refit on validation or open test predictions. Then review and lock exact final
recipes, refit epoch budgets, complete comparator list, paired uncertainty and
multiplicity controls before the final batch test evaluation. The original paper's
numerical/protocol comparability remains unresolved. Nine-class test partly
contains previously inspected four-class rows; neither a wholly fresh confirmation
claim nor general state-of-the-art claim follows automatically.
