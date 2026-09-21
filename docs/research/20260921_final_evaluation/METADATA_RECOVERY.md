# Processor-metadata incident and continuation

All **23 production fits** finished and passed the original all-fit barrier.
The test-access gate opened at **21:07 Europe/London on 21 September 2026**.
At **21:09**, extraction stopped at the first SigLIP2 view with:

```text
RuntimeError: Held-out feature extraction does not match development model/preprocessing
```

The two DINOv2 feature views were already complete. Candidate prediction and
statistical comparison had not run; no new final performance scores were
inspected to design this repair. This was not a training or numerical failure.

## Cause and bounded repair

The processor returns `image_mean` and `image_std` as Python tuples. Development
contracts, saved as JSON, necessarily read those arrays back as lists. Direct
Python dictionary comparison rejects tuples versus lists even when every number
is identical. A diagnostic reload confirmed identical checkpoint records and
identical serialized preprocessing; only these in-memory sequence types differed.

The user authorized continuation: **"complete and make sure all is running
correctly"**. A separate, hash-bound runtime adapter converts only
`processor.image_mean` and `processor.image_std` from tuples to lists in the
returned metadata dictionary. It verifies that serialized JSON is unchanged.

It does **not** modify the processor, image tensors, model objects, weights,
checkpoint records, numerical tolerances, training, fusion weights, candidate
panel, statistical tests or promotion gates. The original strict comparison
still executes. Real differences still stop extraction. No checksum check is
disabled and no original locked source, selection lock or queue plan is edited.

The adapter runs only in the original feature-extraction job. The supervisor can
launch only the three remaining original evaluation jobs, never a training or
gate-opening job. It also reuses the previously audited finite status-I/O retry;
that adapter and its manifest are unchanged. This runtime correction is recorded
explicitly rather than described as an entirely unchanged executable pipeline.

## Validation

- Regression suite: **360 passed, one optional GPU test skipped**.
- Repository-wide Ruff and whitespace checks passed.
- Regression tests reproduce the original failure through the original
  extraction function, verify cached-output replay, and reject changed model
  evidence or preprocessing values.
- A separate real-CUDA preflight covered all four frozen backbones and both
  development view contracts per backbone. It used only a synthetic image;
  no POLAR images, labels or performance scores were read.
- All eight model/preprocessing contracts matched. Model, transform and
  checkpoint objects remained identical before/after wrapping. Synthetic pixel
  and feature maximum differences were **zero** for every backbone.
- The same tuple/list issue also affected ConvNeXt V2 and DINOv3 metadata;
  the preflight caught these before resuming the queue. DINOv2 needed no change.
- All 23 completed-fit checks and protected historical file hashes passed again
  before preparing the recovery manifest.
- The final inference loader was exercised on one retained seed-42 checkpoint
  for each of the five task/backbone combinations. All foundation, preprocessing,
  parameter and checkpoint contracts passed, without reading images or labels.
  The unchanged 18-comparison statistical contract also validated.

This small synthetic check establishes engineering parity for the repair, not
additional statistical evidence about benchmark performance.

## Evidence and operation

New recovery evidence directory:

`.runs/final_evaluation_20260921_1322/recovery/processor_metadata_20260921_2305/`

It preserves the failed status and logs, original receipts, verified-fit barrier,
test gate and completed DINOv2 cache metadata. `manifest.json` binds the adapter,
tests, preflight, parent recovery, original lock, plan, barrier and gate hashes.
Per-process `runtime_*.jsonl` files record actual commands and converted fields.
No prior evidence was deleted and no production fit is repeated.

The earlier `processor_metadata_20260921_2302/` directory is an incomplete
evidence-copy preparation, not an evaluation attempt: relative path handling
stopped preparation before manifest creation or launch. The path handling was
fixed, regression-tested, and the CUDA preflight repeated against the final
adapter source. The partial evidence copy remains preserved.

The hidden recovery supervisor was launched at **23:05 Europe/London on
21 September 2026**. Launch command, **only if no supervisor is already running**:

```powershell
.venv\Scripts\python.exe tools\polar_metadata_recovery.py `
  --manifest .runs\final_evaluation_20260921_1322\recovery\processor_metadata_20260921_2305\manifest.json
```

Safe read-only status command:

```powershell
.venv\Scripts\python.exe tools\polar_locked_status.py `
  --run-dir .runs\final_evaluation_20260921_1322
```

The remaining stages are feature extraction, fixed candidate prediction and the
predeclared statistical comparisons. Only root `completion.json` with status
`LOCKED_FINAL_EVALUATION_COMPLETE` establishes completion. A completed model or
feature cache alone does not. Historical test exposure remains disclosed in the
original protocol; this repair does not make the cohort newly blind.
