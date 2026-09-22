# POLAR final-evaluation evidence

Portable numerical evidence for the evaluation completed on 21 September 2026.
The export preserves prediction values, row order and audited membership; it does
not contain images, model parameters, credentials or workstation paths.

| File | Contents |
| --- | --- |
| `polar4_predictions.npz` | 3,329 rows; ten comparison candidates and six matching-seed prediction sets |
| `polar9_predictions.npz` | 6,984 rows; ten comparison candidates and six matching-seed prediction sets |
| `cohort.csv` | All 35,007 audited nine-class rows: split, label, target box, source group and image/annotation hashes |
| `quarantine.csv` | 317 excluded rows in the same portable schema |
| `data_audit.json` | Original audit with only its machine-dependent runtime field omitted |
| `manifest.json` | Public-file hashes, original prediction hashes, class order and bindings to the locked analysis |

Each NPZ contains `image_ids`, `labels`, `source_groups`, `class_names` and
one probability matrix per candidate. Load with `numpy.load(..., allow_pickle=False)`.
Rows are in exactly the original prediction order; class names define column order.
The four-class prior is deterministic, so its three seed references are identical.
The nine-class seed pairs use their corresponding adapted-DINOv2 seeds.

From the repository root, with the project dependencies installed:

```bash
python tools/verify_benchmark_predictions.py
python tools/verify_benchmark_predictions.py --resample
```

The first command recalculates all candidate metrics, seed transitions, fixed
fusion arithmetic, paired rescue/harm counts and the Holm correction. The second
also repeats the paired bootstrap and randomization calculations. Neither fits
models nor reads source images. Hash verification is also available using only
the standard library through `python tools/check_project.py`.

The [analysis summary](../../docs/research/20260921_final_evaluation/results/summary.json)
and [metric table](../../docs/research/20260921_final_evaluation/results/metrics.csv)
remain the original generated files. [Results and interpretation](../../docs/RESULTS.md)
distinguish the evaluated nominee from the retained prior. This package enables
prediction-level verification, not a fresh checkpoint or dataset replay.

The CSVs contain derived annotation metadata, not the POLAR image archive. Credit
the [POLAR dataset](https://doi.org/10.17632/hvnsh7rwz7.1); upstream data rights are
not replaced by this repository's code licence. See [third-party notices](../../THIRD_PARTY_NOTICES.md).
