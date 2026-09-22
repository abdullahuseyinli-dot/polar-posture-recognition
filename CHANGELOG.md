# Changelog

## Standalone project 1.1.0 — 2026-09-22

- Completed the audited nine-class POLAR benchmark and fixed four-class comparison
  panel: DINOv2, DINOv3, SigLIP2 and ConvNeXt V2, with frozen and adapted controls.
- Reported development-nominated fusion at 95.21% four-class and 94.58% nine-class
  macro-F1; kept prior-model retention decisions unchanged.
- Added 32 portable prediction sets, complete audited cohort/quarantine metadata,
  metric replay and full paired-resampling verification without source images.
- Updated the main page, system diagram, results, error analysis, model card and
  reproduction guide; added a current technical report and PDF.
- Preserved all 330 imported files, original study reports and byte-exact final
  evaluation artifacts. No post-test model selection or additional training.
- Updated citation/archive metadata; no Zenodo deposit or peer-review claim.

## Standalone project 1.0.0 — 2026-09-20

- Established `polar-posture-recognition` as a standalone benchmark, linked to ARFTR.
- Preserved the locked 93.99% four-class POLAR result and its 330 imported
  source/evidence artifacts, with original Git revisions and integrity hashes.
- Added dedicated system, model, results, reproduction and provenance guides.
- Included the 96.11% secondary three-class result, 86.63% held-out V-COCO follow-up,
  86.97% nested-development fusion, and matched DINOv2/DINOv3/SigLIP2 controls.
- Added confidence-interval, transfer and representation figures in PNG and SVG.
- Added standalone metadata, CPU CI, synthetic/evidence tests and aggregate checks.
- Excluded historical qualitative photo composites; no dataset or checkpoint is distributed.
- No new model fitting, revised scores or Zenodo deposit. Original study versions below
  remain historical identities, not earlier releases of this separately named repository.

This file records versioned public study artifacts. The Python package has its own
version in `pyproject.toml`.

## Study Release 2.0.0 - 2026-08-24

- Added a split-preserving, person-level V-COCO study with training-split candidate
  fitting, validation-only selection, a locked train-plus-validation refit, and one
  official test-label open.
- Improved official-test macro-F1 from 0.7071 for the historical source-only DINO
  baseline to 0.8663 for the selected scale-conditioned DINO stack. The paired gain is
  +0.1592 with a 95% image-cluster bootstrap interval of [+0.1454, +0.1735].
- Added controlled screens for person crops, context width, aspect-ratio handling,
  DINOv2, ConvNeXt, SigLIP2, geometry, factorized targets, LP-FT, person-safe
  augmentation, AugMix, background intervention, and a pose diagnostic.
- Added official-test calibration, selective-prediction, per-class, confusion,
  person-scale, boundary, and scene-occupancy evidence.
- Added the v2 technical report, publication figures, executed notebook update,
  release notes, citation metadata, and versioned SHA-256 manifests.
- Retained the v1.0.0 POLAR report, manifest, checksum file, and Git tag as the
  historical source benchmark.

## POLAR Study Report 1.0.0 - 2026-08-23

- Added the complete independent technical report and its reproducible PDF builder.
- Added deterministic post-lock analyses covering error topology, model disagreement,
  person scale, annotation semantics, mixed-person scenes, selective prediction,
  attribution geometry, regularization tradeoffs, and class-conditioned fault response.
- Added 17 portable exploratory tables, a strict JSON summary with source hashes, and
  seven publication figure families in PNG and SVG formats.
- Added explicit separation between the primary locked result, predeclared auxiliary
  diagnostics, and post-lock hypothesis-generating analyses.
- Added GitHub release notes, Zenodo metadata, citation metadata, and a checksummed
  release manifest.
- Consolidated report navigation around the versioned v1.0.0 artifact and removed the
  superseded report duplicate.
- Removed a machine-local path from the portable failure ledger and made future exports
  sanitize repository-local failure messages.

The report version is 1.0.0. The installable Python package is version 2.0.0; the
historical report tag is `polar-study-v1.0.0`.
