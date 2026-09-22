# Data boundary

No source images, feature caches or trained weights are distributed.
Obtain POLAR from its [publisher](https://doi.org/10.17632/hvnsh7rwz7.1) and
assess its annotation and upstream image terms independently.

| Audited cohort | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Nine classes | 21,057 | 6,966 | 6,984 |
| Four-class subset | 9,958 | 3,327 | 3,329 |

The nine-class audit excludes 317 images across 154 cross-split source components,
including the 125-image historical quarantine. Original split membership is
preserved. Four-class membership is unchanged by the additional audit.

The public [cohort and quarantine CSVs](../results/polar_20260921/README.md)
contain image IDs, splits, labels, target boxes, source groups and checksums,
without image paths or media. The audit cannot prove subject/session independence
or absence of every near duplicate. The four-class test was historically inspected
and is part of the nine-class test.

The tracked `data/manifest.csv` belongs to the **earlier 285-image COCO pilot**,
not POLAR. Do not use it to train the current benchmark.
[Reproduction requirements](../docs/REPRODUCIBILITY.md) ·
[Model card](../docs/MODEL_CARD.md).

V-COCO uses official image memberships with a custom three-class person-level
posture mapping; it is not the standard agent/role-AP task.
