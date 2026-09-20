# Launch record — 20 September 2026

The development queue was launched in a hidden, detached process at **23:24 London
time**. This is a launch record, not a claim that training or the competitive audit
has finished. Read private `queue_status.json` for live status.

| Check | Result |
| --- | --- |
| Tests | 182 passed, zero failures/skips after the startup regression fix |
| Lint/style | Pass; no new findings |
| Retained evidence | All 330 protected historical files preserved; exported confusion arithmetic passes |
| Original POLAR data | Provider hashes verified; all 35,324 images/annotations validated |
| Original four-class cohort | Exact IDs, labels, boxes and image hashes retained; no new exclusions |
| Nine-class cohort | 21,057 train / 6,966 validation / 6,984 test |
| CUDA | PyTorch 2.11.0+cu128; RTX PRO 3000 Blackwell laptop GPU, 12 GB |
| Pretrained smoke tests | DINOv2-B, DINOv3-B, SigLIP2-B and ConvNeXtV2-B passed |
| Nine-class optimizer/checkpoint smoke | Passed; saved-prediction replay maximum difference 0.0 and identical labels |
| Current phase when recorded | Nine-class DINOv2 classifier screen (job 4/25); both DINOv2 feature caches complete |
| Observed extraction throughput | Approximately 320 images/s at 99% GPU utilization; not a forecast for training |
| Optional DINOv3-L | Not available locally; explicitly skipped, no substitute |
| New test evaluation | Not performed; current queue forbids it |

The DINOv3-B checkpoint was found in the transferred `C:\Users\DELL\hac_pretrained`
folder. Its original weights/configuration match historical hashes exactly. The
original processor file was absent; its recorded replacement uses explicitly
identified reference-library defaults. This is not original-processor byte parity.

## Recheck or resume

Run directory: `C:\Users\DELL\polar-posture-recognition\.runs\benchmark_20260920_2250`

Active plan: `queue_plan_02.json`. The original `queue_plan.json` and its failed
pre-training startup log are retained. The cache-completeness correction and its
regression test are recorded in `engineering_amendment_01.json`; no selection,
performance or safety threshold changed.

```powershell
Get-Content .runs/benchmark_20260920_2250/queue_status.json
Get-Content .runs/benchmark_20260920_2250/queue_receipts.json
```

Only if no supervisor is running, resume with:

```powershell
.venv\Scripts\python.exe experiments/run_polar_benchmark_queue.py `
  --run-dir .runs/benchmark_20260920_2250 --plan-name queue_plan_02.json
```

The actual supervisor writes its PID to `queue_process.lock`/`queue_status.json`;
the Windows venv launcher PID is different. Never start a second queue against the
same outputs. After a crash, verify the recorded process has exited before handling
any stale lock; do not delete locks blindly.

The current Windows power scheme already disables idle sleep on AC and battery;
no global power settings were changed. Keep the machine powered, ventilated and
connected. A graceful stop is available by creating `STOP_AFTER_CURRENT_JOB` inside
the run directory. Jobs have finite watchdog deadlines and fail closed on errors.

## What remains after the unattended queue

Review development results and available modern-backbone adaptation, then lock
final recipes and fit/evaluate once. Bounded adapted challengers, the final paired
four-class comparison, nine-class test results and confidence gates remain pending.
The current queue cannot certify state of the art. See [the full plan](PLAN.md).
