# Release validation

Project 1.1.0, checked 22 September 2026. These checks validate the public release
and saved predictions, not a new fit or independent checkpoint replication.

| Check | Result and scope |
| --- | --- |
| Full pytest suite | **405 passed, one optional real-CUDA test skipped** |
| Ruff regression check | **Zero findings** against the pinned zero-finding baseline |
| Source compilation | Passed for source, experiments and tools |
| Historical preservation | All **330 imported files** match recorded hashes |
| Historical metric arithmetic | Eight systems verified from confusion counts |
| Current metric replay | All **20 candidates**, six seed pairs and 18 paired transitions match |
| Public probability integrity | **32 prediction sets** in two compressed, pickle-free NPZ archives |
| Cohort consistency | 35,007 retained / 317 quarantined rows; predictions match test IDs, labels and groups |
| Statistical replay | All 18 paired bootstrap/randomization records and the complete Holm family match |
| Evidence failure tests | Missing artifacts, candidates, seeds, comparisons and arrays; altered class order, cohort sizes and provenance bindings rejected |
| Presentation integrity | Current local links, six historical-overview files, ten current PNG/SVG files and report bindings verified |
| PDF review | Seven-page report checked for metadata, tables, figures, print readability and portable web links |
| Wheel build | `polar_posture_recognition-1.1.0-py3-none-any.whl` built successfully |

The suite emits 14 scikit-learn deprecation warnings about the existing SVC
probability argument. They do not change these results; the preserved fitting
recipes have not been rewritten to suppress them.

The first clean Linux run exposed a newer-Transformers SigLIP interface mismatch
in three model-contract tests. The dependency is now pinned to the actual fitting
version, **5.5.3**, rather than changing the recorded model implementation.
See the workflow history for the failed run and its corrected successor.

## Repeat the checks

```bash
python tools/check_style.py
python -m compileall -q src experiments tools
python -m pytest
python tools/check_project.py
python tools/verify_benchmark_predictions.py
python tools/verify_benchmark_predictions.py --resample
git diff --check
```

The [quality-gates workflow](https://github.com/abdullahuseyinli-dot/polar-posture-recognition/actions/workflows/ci.yml?query=branch%3Amain)
records Linux/CPU validation for Python 3.11 and 3.12 per published commit. It includes fast public
probability replay; full 5,000-draw bootstrap / 10,000-draw randomization replay
is also available locally through `--resample`.

The evidence verifier rejects changed prediction bytes, path escapes, metric
mismatches and incomplete inventories. It checks fixed fusion arithmetic, cohort
alignment, seed-reference gains and marginal interval consistency. The
standard-library checker also protects the original summary hash, selection-lock
and prediction-manifest bindings, and prior-retention decisions. Reported check
counts are derived from the records actually verified.

## Boundaries

No training, image retrieval, foundation-model download or model-promotion change
is part of release preparation. Numerical replay uses the recorded statistical
implementation, not an independently implemented estimator. Full model replay
requires local data and weights. The real-CUDA metadata preflight from the
completed evaluation is documented in its [completion record](research/20260921_final_evaluation/RESULTS.md);
it is distinct from the optional test skipped in release validation.

[Prediction package](../results/polar_20260921/README.md) ·
[Reproduction guide](REPRODUCIBILITY.md) · [Project history](PROJECT_HISTORY.md).
