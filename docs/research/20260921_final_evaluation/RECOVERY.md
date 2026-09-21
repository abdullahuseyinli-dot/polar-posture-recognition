# Status-I/O incident and authorized continuation

On 21 September 2026 at **18:35 Europe/London**, the supervisor stopped during
nine-class DINOv2 seed 52, epoch 14. The exception was Windows `WinError 5`
while atomically replacing `progress.json`. It did not occur in a model forward
pass, optimizer update, or checkpoint replay. The process holding the conflicting
handle was not recorded; the preceding status read is a plausible contributor,
not a proven attribution.

At interruption, 18 of 23 production fits were complete. The failed fit had an
intact epoch-13 checkpoint, including model, optimizer, scheduler, scaler and
history. Source/request/checkpoint/history hashes passed verification. No test
gate had opened and no final test performance had been examined.

## Tested repair and limits

The Windows regression test reproduces the original error using an open reader
and the original unmodified atomic JSON writer. On this machine, even a
delete-sharing reader can temporarily block replacement of the destination.
Therefore the recovery uses both a short-lived status reader and a bounded
writer retry. It does not claim that sharing flags alone solve the problem.

Microsoft documents the [sharing flags and rename/delete access](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew).
The concrete overwrite behavior above is established by the native Windows
tests, not inferred solely from that documentation.

The separately recorded runtime adapter retries Windows errors **5, 32 or 33**
for at most **five seconds**, with 10–250 ms backoff, only when:

- The destination is `progress.json`, `queue_status.json`, `queue_receipts.json`
  or `inference_progress.json` inside this run.
- The source is the same directory's exact PID-scoped temporary JSON filename.
- The original writer attempts the same replacement with the same bytes.

Permanent failures still propagate. The adapter does **not** retry checkpoint,
request, summary or model-file writes, does not retry training jobs, and does not
catch CUDA, numerical, data-integrity or evaluation-gate errors. It invokes only
the exact original job commands registered in the unchanged queue plan. Actual
wrapped launches and any file-operation retries are recorded in per-process
audit logs. No global Python installation or system setting is modified.

The original scientific source files, selection lock, queue plan, recipes,
seeds, budgets and inference/statistical rules remain byte-identical. Runtime
behavior is not represented as completely unchanged: this explicitly disclosed
adapter changes status-file error handling only. Its own code and tests are bound
by the separate recovery manifest. No checksum validation was disabled.

## Preserved evidence and restart

User authorization: **“fix and run as needed.”** The failed status, receipts,
logs, interrupted progress/temp file, request, epoch-13 history/checkpoint and
integrity sidecar were copied into a new recovery evidence directory and their
hashes verified before continuation. No old incident evidence was deleted.

Recovery directory:

`.runs/final_evaluation_20260921_1322/recovery/status_io_20260921_1919/`

Its `manifest.json` binds the original selection lock/plan, the adapter/reader
and regression-test sources, the exact retry policy, and the preserved evidence.
The hidden supervisor was relaunched at **19:20 Europe/London**. It uses the
existing completed-fit receipts and resumes DINOv2 seed 52 from **epoch 14**, not
from scratch. Epoch-level RNG reset and optimizer/scheduler state are restored
by the existing locked trainer; no epoch or checkpoint is selected by performance.

Validation before restart: **341 tests passed, one optional GPU test skipped**;
Ruff passed. The added tests include the actual Windows collision, successful
recovery after reader release, identical retry payloads, finite failure deadlines,
non-status failures propagating unchanged, and preservation of original job args.

## Safe status command

Use the read-only observer for future polling instead of opening live progress
files with `Get-Content` or a file viewer that may hold them open:

```powershell
.venv\Scripts\python.exe tools\polar_locked_status.py `
  --run-dir .runs\final_evaluation_20260921_1322
```

The recovery supervisor command is already running; do not launch a duplicate:

```powershell
.venv\Scripts\python.exe tools\polar_status_io_recovery.py `
  --manifest .runs\final_evaluation_20260921_1322\recovery\status_io_20260921_1919\manifest.json
```

Final evaluation still requires all 23 verified production fits. This note is
an operational incident record, not a new benchmark score or a claim that the
final phase has completed.
