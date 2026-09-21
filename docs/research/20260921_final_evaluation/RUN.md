# Final evaluation run

Launched 21 September 2026 at 13:23 Europe/London. Final scores are pending;
this document records the launch, not completion or an improvement claim.

- Run directory: `.runs/final_evaluation_20260921_1322/`.
- Protocol: [PROTOCOL.md](PROTOCOL.md).
- Comparison boundaries: [COMPARISON_SCOPE.md](COMPARISON_SCOPE.md).
- Regression checks before launch: **324 passed**, one optional GPU test skipped
  in the normal suite. The optional synthetic CUDA head test was run separately
  and passed: five calibrated fits converged, reload difference zero, GPU/CPU
  kernel maximum difference approximately `3.28e-7`.
- All 15 neural-fit contracts, eight head-fit contracts and the complete
  18-comparison family validated against the sealed final selection lock.
- The original historical four-class test prediction artifact was recovered
  with its exact published evidence checksum; no historical artifact was replaced.

The queue runs 32 fixed jobs and stops on failure. All five real-image engineering
smokes precede production fits. The held-out gate remains closed until all 23
production fits and checkpoint replays have passed. Test extraction, predictions,
comparisons, figures and reporting then run without another model-choice step.

From the repository root, the supervisor command is:

```powershell
.venv\Scripts\python.exe experiments\run_polar_locked_queue.py `
  --selection-lock .runs\final_evaluation_20260921_1322\final_selection_lock.json
```

It was launched as a hidden process; do not launch a second supervisor.
Use `tools/polar_locked_status.py --run-dir .runs/final_evaluation_20260921_1322`
to read status with short-lived Windows-safe handles. A
`STOP_AFTER_CURRENT_JOB` file in the run directory asks it to stop safely after
the active job. There is no automatic model retry or test-driven search.

Expected final report: `comparisons/report.md`. Final numeric evidence:
`comparisons/summary.json`, `metrics.csv`, `comparisons.csv`,
`analysis_integrity.json`, and the generated plots. Only
`completion.json` with status `LOCKED_FINAL_EVALUATION_COMPLETE` means the entire
phase completed. A comparison gate failure is an outcome to report, not a reason
to change the nominated recipe or retune on the test set.

The current public four-class result and the separate ARFTR repository are
unchanged. No GitHub push or public result replacement is part of this launch.

Recovery update, 21 September at 19:20 Europe/London: a Windows status-file
replacement failure interrupted DINOv2 seed 52. See [RECOVERY.md](RECOVERY.md)
for the preserved incident evidence, tested telemetry-only retry adapter,
authorized epoch-14 continuation, and the safe status-polling command. The
original launcher above is retained as the launch record; the recovery launcher
managed this same locked queue.

Post-training recovery, 21 September: all 23 fits passed verification, but frozen
SigLIP2 feature extraction stopped on an in-memory tuple versus JSON-list
comparison. See [METADATA_RECOVERY.md](METADATA_RECOVERY.md) for the narrowly
scoped repair, all-backbone synthetic CUDA parity checks and the current recovery
command. Training is complete; final evaluation completion is tracked by the
root completion marker described above, not by this launch note.
