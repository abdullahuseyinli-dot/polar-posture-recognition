# Contributing

Keep results traceable to a protocol and evidence artifact. Separate held-out,
development, exploratory and oracle quantities. Do not call custom POLAR/V-COCO
posture tasks standard leaderboard results.

Before proposing a change:

```bash
python -m pytest
python tools/check_style.py
python tools/check_project.py
python tools/verify_benchmark_predictions.py
git diff --check
```

Imported source and evidence are frozen by `results/project_origin.json`.
For new behavior, add a versioned implementation rather than overwriting the
historical recipe or updating its hash to hide a change. Current presentation
and additional tests can evolve with checks. Keep source images, checkpoints,
credentials, feature caches and local `.runs/` files out of Git.

The public `results/polar_20260921/` probability package is an intentional,
checksummed exception to private run outputs. Preserve exact bytes and row order.
Do not select models or thresholds from its test labels. Current report and
figure manifests may change only with reviewed presentation changes; never
refresh historical hashes to hide a numerical or implementation change.

Changes to the temporal architecture belong in the
[ARFTR project](https://github.com/abdullahuseyinli-dot/arftr).
