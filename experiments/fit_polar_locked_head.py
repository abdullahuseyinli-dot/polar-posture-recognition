"""Refit one preselected calibrated RBF head on all locked development rows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.polar_locked_features import fit_locked_head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--task", required=True, choices=("polar4", "polar9"))
    parser.add_argument("--model-kind", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = fit_locked_head(
        args.selection_lock.resolve(), args.task, args.model_kind, args.output_dir.resolve()
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "task", "model_kind", "training_rows", "fit_seconds")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
