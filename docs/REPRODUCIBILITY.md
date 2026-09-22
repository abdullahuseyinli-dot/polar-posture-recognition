# Reproducibility

## Public checks without a dataset or GPU

Python 3.11 or 3.12 is required.

```bash
git clone https://github.com/abdullahuseyinli-dot/polar-posture-recognition.git
cd polar-posture-recognition
python tools/check_project.py
```

This standard-library check verifies preserved history, metadata, current links,
figure/PDF hashes, audited membership and confusion-derived metrics.
It performs no fitting, image access or model download.

## Installation and prediction replay

Use a dedicated environment: this project and ARFTR share the historical `hac`
namespace and must not be installed in the same environment.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook,report]"
python tools/verify_benchmark_predictions.py
python tools/verify_benchmark_predictions.py --resample
```

Install the appropriate PyTorch wheel first if a CPU-only installation is desired.
The replay uses NumPy, SciPy and scikit-learn, not CUDA. It loads the
[public NPZs](../results/polar_20260921/README.md) with pickle disabled.

The fast check recalculates all 20 models' metrics, six seed pairs, fusion
arithmetic, all 18 rescue/harm transitions and the Holm family. The optional pass
repeats paired bootstrap and randomization draws. This verifies saved predictions;
it is not an independent replication with newly trained models.

## Code checks

```bash
python -m pytest
python tools/check_style.py
python -m compileall -q src experiments tools
python tools/check_project.py
```

CI uses CPU PyTorch without model downloads. The optional real-CUDA test is
separately opt-in; skipping it is not GPU validation. Ruff is pinned and the
allowed finding baseline remains zero.

## Figures and report

```bash
python tools/render_benchmark_release.py
python tools/build_study_papers.py docs/POLAR_BENCHMARK_REPORT.md -o output/pdf/polar_benchmark_report_v1.1.0.pdf
python tools/seal_benchmark_report.py
python tools/check_project.py
```

Only the new charts and report are rebuilt. The seal binds report source, builder,
figures and PDF; refresh it only after reviewing intentional report changes.
Library/font versions may change rendering bytes without changing numerical
results. The separate `render_project_figures.py` builds historical overview plots.

## Full training and checkpoint replay

Public source covers preparation, CUDA features, adaptation, locked refits and
statistics. Full replay also needs locally obtained POLAR images, permitted
backbone weights, matching preprocessing and reconstructed run configuration.
Caches and trained classifiers are not distributed.

Start with the [experiment map](../experiments/README.md),
[locked protocol](research/20260921_final_evaluation/PROTOCOL.md) and
[recipe](ARCHITECTURE.md). Dated documents retain private run paths as provenance,
not public turnkey commands. Keep the
[DINOv3 processor caveat](REPRESENTATIONS.md).

Final fitting used PyTorch 2.11.0 + CUDA 12.8, torchvision 0.26.0 and Transformers
5.5.3 on Windows. Transformers is pinned to that recorded version: newer releases
changed SigLIP's internal module layout and failed the clean-runner model-contract
tests. Neural fits used bfloat16; frozen kernels used FP32 with TF32 disabled and
CPU libsvm fitting. Other dependency ranges are not a bitwise environment lock.
Recorded session times can omit resumed work or feature extraction; they
are not deployment latency.

A fresh replication must define its data and selection boundary before fitting.
Do not use published test labels to train routers, select thresholds or choose
a different diagnostic winner.

## Historical tools

The [original POLAR](POLAR_PUBLIC_REPORT.md) and [V-COCO](VCOCO_V2_EXTERNAL_TRANSFER.md)
reports remain unchanged. The executed notebook covers those studies, not the
new nine-class benchmark. Legacy README and release builders target original
Git trees; do not use them to overwrite current guides.
[Provenance](PROJECT_HISTORY.md) binds 330 imported files.
