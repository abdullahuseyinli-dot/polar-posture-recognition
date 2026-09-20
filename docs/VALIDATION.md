# Standalone project validation

Validation for project 1.0.0 covers the public package, not a new training run.

| Check | Scope |
| --- | --- |
| `python tools/check_project.py` | Version consistency, current links, 330 preserved historical files, eight confusion-derived systems, fixed test gates and six new figure artifacts |
| `python -m pytest` | Synthetic/model-contract/evidence tests; no dataset or model download |
| `python tools/check_style.py` | Pinned Ruff; no new findings against a zero-finding extraction baseline |
| `python -m compileall -q src experiments tools` | Source compilation |
| Wheel build | Independently named package metadata and source inclusion |
| CPU GitHub Actions | Clean Linux checkout without workstation assets |

Local verification on 20 September 2026:

- **102 tests passed, four CUDA-only tests skipped**; no training or model download.
- **330 historical files** verified against the recorded original Git revisions.
- **Eight systems** recomputed from confusion counts: six POLAR, two V-COCO.
- **132 current local documentation links** and **six new figure artifacts** verified.
- **Zero Ruff findings**; compilation passed.
- `polar_posture_recognition-1.0.0-py3-none-any.whl` built successfully.
- Current PNG figures visually inspected; no clipped labels or source photographs.

The [quality-gates workflow](https://github.com/abdullahuseyinli-dot/polar-posture-recognition/actions/workflows/ci.yml?query=branch%3Amain)
records the clean Linux/CPU checkout result for each published commit.

NLL, Brier, calibration and confidence intervals are hash-preserved exports;
their row-level computation is not replayed from aggregate confusion matrices.
Full model replay, gated DINOv3 access and any GPU deployment benchmark remain
outside this packaging validation.
