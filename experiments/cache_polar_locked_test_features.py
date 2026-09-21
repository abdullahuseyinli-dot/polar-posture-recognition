"""Extract the fixed label-free held-out feature panel after every final fit is verified."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.polar_locked_features import cache_locked_test_features


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-lock", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            cache_locked_test_features(args.selection_lock.resolve(), args.run_dir.resolve())
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
