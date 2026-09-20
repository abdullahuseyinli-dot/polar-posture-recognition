# Contributing

Keep results traceable to a protocol and evidence artifact. Separate held-out,
development, exploratory and oracle quantities. Do not call custom POLAR/V-COCO
posture tasks standard leaderboard results.

Before proposing a change:

```bash
python -m pytest
python tools/check_style.py
python tools/check_project.py
git diff --check
```

Imported source and evidence are frozen by `results/project_origin.json`.
For new behavior, add a versioned implementation rather than overwriting the
historical recipe or updating its hash to hide a change. Current presentation
and additional tests can evolve with checks. Keep source images, checkpoints,
credentials, feature caches and local `.runs/` files out of Git.

Changes to the temporal architecture belong in the
[ARFTR project](https://github.com/abdullahuseyinli-dot/arftr).
