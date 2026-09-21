# Legacy comparison-archive recovery

The metadata repair succeeded: all eight held-out feature caches completed and
passed integrity checks. At **23:15 Europe/London on 21 September 2026**, the queue
then stopped in the four-class historical comparator. Ten new four-class
component prediction files had already been saved and verified. The nine-class
prediction panel and statistical comparisons had not started.

## Failure and repair

The historical archive's `image_ids` member is an object-dtype NumPy vector,
although all 3,329 elements are strings. The original evaluator correctly uses
`allow_pickle=False`, so NumPy rejected this legacy representation. The archive
checksum still matched its original locked SHA256:

`0f96ecb7411abf8b3385004380a5fa001965113f0855015db4337fb76078c5f9`

The recovery does **not** enable pickle loading. A bounded, passive decoder
reads the observed NumPy envelope and literal string tokens using `pickletools`;
it never imports or executes the functions named in that envelope. Unknown
instructions, malformed envelopes, non-string elements, duplicate IDs,
unexpected shapes, excessive sizes and trailing data fail closed.

A scoped adapter supplies these same IDs as a Unicode array only when the
original evaluator opens this exact hash-bound historical archive. Other
archives retain their existing safe loader. Numerical arrays, model predictions,
original source files, selection lock, queue plan and all scientific settings
remain unchanged. The original historical comparator still enforces cohort,
label, class-order and published-score parity.

User authorization: **"complete and make sure all is running correctly"**.
No new-model score was inspected to design or select this repair.

## Checks and continuation

- **378 tests passed**, one optional GPU test skipped; repository-wide Ruff and
  whitespace checks passed.
- Tests include an executable malicious-object payload that is rejected without
  execution, checksum failures, altered labels/classes/IDs, numerical-array
  equality, scope isolation and the original comparison function.
- The actual 3,329-row archive passed the original historical replay gate against
  the current audited four-class manifest: expected macro-F1
  `0.9398833285904803`, unchanged. This is the old reference result, not a new gain.
- The unchanged complete 18-comparison statistical contract validated.
- Original completed-fit verification, completed queue receipts and protected
  historical-file checks passed before restart.

The recovery supervisor restarted at **23:35 Europe/London on 21 September**.
Only the two remaining original jobs may execute: fixed-candidate prediction
and statistical comparison. Training, test-gate opening and feature extraction
cannot be relaunched through this adapter. The ten saved four-class component
predictions are retained and reused after their normal checksum/request checks.

Evidence directory:

`.runs/final_evaluation_20260921_1322/recovery/legacy_archive_20260921_2334/`

Its manifest binds the new adapter/tests and the previous recovery chain. It
preserves the failed logs/status, receipts and all existing prediction files.
`preflight.json` records the actual historical replay; runtime logs record the
original and executed commands. Previous evidence is not deleted or overwritten.

Current command, **only when no supervisor is already running**:

```powershell
.venv\Scripts\python.exe tools\polar_archive_recovery.py `
  --manifest .runs\final_evaluation_20260921_1322\recovery\legacy_archive_20260921_2334\manifest.json
```

Continue using `tools/polar_locked_status.py` for read-only status checks.
Only root `completion.json` with status `LOCKED_FINAL_EVALUATION_COMPLETE`
establishes completion of predictions and comparisons. Historical test exposure
and the prohibition on test-driven reselection remain unchanged.
