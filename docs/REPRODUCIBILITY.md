# Reproducibility

## 1. Public evidence check — no packages, GPU or dataset

```bash
git clone https://github.com/abdullahuseyinli-dot/polar-posture-recognition.git
cd polar-posture-recognition
python tools/check_project.py
```

Python 3.11 or 3.12 is required. The check verifies preserved source/result hashes,
confusion-derived macro-F1 and accuracy, ensemble-delta arithmetic, test-gate
metadata, figure/source hashes, versions and current links. It does not fit a model,
download weights, access labels, or recompute NLL/Brier/bootstrap intervals from
unavailable per-example predictions.

## 2. Installation and code-level checks

Use a dedicated environment. This project and ARFTR retain the historical `hac`
import namespace; **do not install both in the same virtual environment**.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook,report]"
python -m pytest
python tools/check_style.py
python -m compileall -q src experiments tools
python tools/check_project.py
```

Install the appropriate PyTorch wheel first for CPU or GPU use. CI uses CPU PyTorch
and performs no model downloads. CUDA-specific tests skip on CPU; their skip is
not a successful CUDA replay. Dependency ranges are current installation constraints,
not a claim of bitwise replay of the historical environment.

The imported research code is hash-preserved. Style checks pin Ruff and reject
new findings; the extraction's pre-edit baseline contained zero findings.

## 3. Rebuild current figures

```bash
python tools/render_project_figures.py
python tools/check_project.py
```

This reads only public aggregates and exported uncertainty. Outputs include PNG,
SVG and a source-hash manifest. The preserved historical figures and report PDFs
are not overwritten. Library versions can affect rendering bytes without changing
statistical content.

## 4. Full training/checkpoint replay

Full replay needs the original datasets, local manifests, preprocessing, model
revisions, fitted classifiers, checkpoints, selection locks and saved feature caches.
These are not in the public repository. The historical
[POLAR protocol](POLAR_SCALE_STUDY_PROTOCOL.md),
[report](POLAR_PUBLIC_REPORT.md), [V-COCO report](VCOCO_V2_EXTERNAL_TRANSFER.md),
and [experiment map](../experiments/README.md) identify the recipes.

Original runners often point to `.runs/polar_final`, `.runs/polar_v2` and
`.runs/vcoco_v3`; those are provenance locations, not promised public inputs.
No full replay is claimed from this packaging change. DINOv3 additionally needs
approved access to its revision-pinned model. Never bypass a test gate, retrain a
router using test labels, or select new thresholds against the published test scores.

## Historical tooling

`tools/validate_repository.py` and old release builders target their original Git
trees, including old metadata and qualitative-media inventories. They are preserved,
not the standalone project's acceptance check. Use `tools/check_project.py` here.
Likewise, historical README/notebook builders must not replace current guides.

The executed notebook covers POLAR and the V-COCO follow-up, not the later DINOv3
screen. That screen has its own [guide](REPRESENTATIONS.md). The
[origin manifest](../results/project_origin.json) binds 330 imported historical files;
new presentation files are checked separately.
