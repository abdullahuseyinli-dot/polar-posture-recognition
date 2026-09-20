"""Restore hash-pinned POLAR v1 and audit nine classes without changing old evidence."""

from __future__ import annotations

import argparse
from pathlib import Path

from hac.polar_benchmark_data import prepare_benchmark_data, restore_sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--restore-source", action="store_true")
    parser.add_argument("--source-only", action="store_true")
    parser.add_argument("--annotations-dir", type=Path)
    parser.add_argument("--images-dir", type=Path)
    parser.add_argument("--image-sets-dir", type=Path)
    parser.add_argument("--legacy-data-root", type=Path)
    parser.add_argument(
        "--legacy-data-lock", type=Path, default=Path("results/polar_data_lock.json")
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--download-workers", type=int, default=3)
    parser.add_argument("--seven-zip", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.download_workers < 1:
        parser.error("worker counts must be positive")
    if args.source_only and not args.restore_source:
        parser.error("--source-only requires --restore-source")
    source = args.output_dir.resolve() / "source"
    paths = {
        "annotations": source / "Annotations",
        "images": source / "images" / "JPEGImages",
        "splits": source / "ImageSets",
    }
    if args.restore_source:
        paths = restore_sources(source, workers=args.download_workers, seven_zip=args.seven_zip)
    if args.source_only:
        return
    if args.legacy_data_root is None:
        parser.error("--legacy-data-root is required for the immutable four-class parity audit")
    prepare_benchmark_data(
        args.output_dir.resolve(),
        args.annotations_dir or paths["annotations"],
        args.images_dir or paths["images"],
        args.image_sets_dir or paths["splits"],
        args.legacy_data_root,
        args.legacy_data_lock,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
