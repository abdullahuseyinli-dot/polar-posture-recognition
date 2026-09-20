"""Cache pinned CUDA representations for development-only four/nine-class POLAR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hac.polar_benchmark_features import (
    MODEL_SPECS,
    PREPROCESS_POLICIES,
    VIEWS,
    FeatureRequest,
    cache_features,
    download_model_checkpoints,
    probe_model_availability,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-kind", choices=tuple(MODEL_SPECS))
    parser.add_argument("--view", choices=VIEWS)
    parser.add_argument("--preprocess", choices=PREPROCESS_POLICIES, default="official_processor")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunk-rows", type=int, default=1024)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Download this exact pinned checkpoint if needed",
    )
    parser.add_argument("--autocast-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument(
        "--max-images",
        type=int,
        help="Deterministic first-N development smoke cache; not a full cache",
    )
    parser.add_argument(
        "--probe-models",
        action="store_true",
        help="Read-only local availability probe; no CUDA or downloads",
    )
    parser.add_argument(
        "--download-models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        help="Prepare exact model snapshots without CUDA; requires --allow-download",
    )
    parser.add_argument("--download-report", type=Path)
    parser.add_argument(
        "--public-range-download",
        action="store_true",
        help="Eight bounded HTTPS workers for public pinned safetensors; gated models excluded",
    )
    args = parser.parse_args(argv)
    if args.download_models and (not args.allow_download or args.download_report is None):
        parser.error("--download-models requires --allow-download and --download-report")
    if args.public_range_download and not args.download_models:
        parser.error("--public-range-download requires --download-models")
    if (
        not args.probe_models
        and not args.download_models
        and any(
            getattr(args, name) is None for name in ("manifest", "output_dir", "model_kind", "view")
        )
    ):
        parser.error("Extraction requires --manifest, --output-dir, --model-kind, and --view")
    return args


def main() -> None:
    args = parse_args()
    if args.probe_models:
        print(json.dumps(probe_model_availability(args.cache_dir), indent=2, sort_keys=True))
        return
    if args.download_models:
        download_model_checkpoints(
            args.download_models,
            cache_dir=args.cache_dir,
            report_path=args.download_report,
            public_range_download=args.public_range_download,
        )
        return
    request = FeatureRequest(
        **{
            key: value
            for key, value in vars(args).items()
            if key
            not in {"probe_models", "download_models", "download_report", "public_range_download"}
        }
    )
    result = cache_features(request)
    print(
        json.dumps(
            {
                "status": result["status"],
                "model_kind": result["model_kind"],
                "view": result["view"],
                "rows": result["rows"],
                "feature_shape": result["feature_shape"],
                "runtime_seconds": result["runtime_seconds"],
                "output_dir": str(request.output_dir.resolve()),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
