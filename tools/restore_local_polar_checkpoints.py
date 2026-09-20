"""Restore the user's hash-verified transferred DINOv3-B checkpoint without network access.

The original transfer omitted the image processor. Its replacement is explicitly
identified as a new reference configuration from Transformers 5.5.3 defaults, not
as recovered bytes from the original Hugging Face model revision.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import shutil
from pathlib import Path

from hac.polar import sha256_file
from hac.polar_benchmark_features import MODEL_SPECS, checkpoint_evidence

EXPECTED_SOURCE = {
    "config.json": {
        "bytes": 744,
        "sha256": "3c9cc418f4622fd6d5587fd142b6f3cba0ba6a69f67ced907d8b7f26118451ec",
    },
    "model.safetensors": {
        "bytes": 342662192,
        "sha256": "9a21ac3df0c63839d62612dda6f454d816c25611cc7a52966ed5a5a94921dc8b",
    },
}
PROCESSOR_SOURCE = "https://github.com/huggingface/transformers/blob/v5.5.3/src/transformers/models/dinov3_vit/image_processing_dinov3_vit.py"
META_SOURCE = "https://github.com/facebookresearch/dinov3#image-transforms"


def write_new_or_verify(path: Path, encoded: bytes) -> None:
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Existing recovery artifact differs; not overwritten: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)


def copy_verified_file(source: Path, destination: Path, expected: dict) -> None:
    if source.stat().st_size != expected["bytes"] or sha256_file(source) != expected["sha256"]:
        raise RuntimeError(f"Transferred checkpoint source hash or length mismatch: {source.name}")
    if destination.exists():
        if (
            destination.stat().st_size != expected["bytes"]
            or sha256_file(destination) != expected["sha256"]
        ):
            raise RuntimeError(
                f"Existing checkpoint destination differs; not overwritten: {destination.name}"
            )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".local-restore.tmp")
    # Independent copy preserves the original even if a future cache consumer modifies its copy.
    shutil.copyfile(source, temporary)
    if (
        temporary.stat().st_size != expected["bytes"]
        or sha256_file(temporary) != expected["sha256"]
    ):
        raise RuntimeError("Restored checkpoint copy failed verification")
    if destination.exists():
        raise RuntimeError("A competing writer created the destination during restoration")
    temporary.rename(destination)


def declared_processor_configuration() -> tuple[bytes, dict]:
    import transformers
    from transformers import DINOv3ViTImageProcessor

    if transformers.__version__ != "5.5.3":
        raise RuntimeError("The declared processor recovery requires exactly Transformers 5.5.3")
    source_file = Path(inspect.getfile(DINOv3ViTImageProcessor))
    recovery = {
        "origin": "declared_transformers_5_5_3_reference_defaults",
        "original_hf_processor_bytes_recovered": False,
        "original_model_weights_and_config_bytes_verified": True,
        "processor_class": "DINOv3ViTImageProcessor",
        "transformers_version": transformers.__version__,
        "implementation_sha256": sha256_file(source_file),
        "reference_source": PROCESSOR_SOURCE,
        "meta_reference_source": META_SOURCE,
        "reference_scope": "HF library's 224-square bilinear defaults; Meta README's generic example uses 256, not claimed identical",
        "selection_used_benchmark_labels": False,
    }
    processor = DINOv3ViTImageProcessor(benchmark_processor_origin=recovery)
    configuration = processor.to_dict()
    if (
        configuration["size"] != {"height": 224, "width": 224}
        or int(configuration["resample"]) != 2
    ):
        raise RuntimeError(
            "Official library defaults differ from the declared 224 bilinear configuration"
        )
    return processor.to_json_string().encode("utf-8"), recovery


def restore_local_dinov3(source_snapshot: Path, cache_dir: Path, report_path: Path) -> dict:
    specification = MODEL_SPECS["dinov3_base"]
    source_snapshot = source_snapshot.resolve(strict=True)
    # Validate both originals before creating a destination or reconstructing any configuration.
    for filename, expected in EXPECTED_SOURCE.items():
        path = source_snapshot / filename
        if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
            raise RuntimeError(
                f"Transferred DINOv3 source failed historical evidence check: {filename}"
            )
    encoded_processor, recovery = declared_processor_configuration()
    snapshot = (
        cache_dir
        / ("models--" + specification["model_id"].replace("/", "--"))
        / "snapshots"
        / specification["revision"]
    )
    for filename, expected in EXPECTED_SOURCE.items():
        copy_verified_file(source_snapshot / filename, snapshot / filename, expected)
    write_new_or_verify(snapshot / "preprocessor_config.json", encoded_processor)
    recovery["preprocessor_config_sha256"] = hashlib.sha256(encoded_processor).hexdigest()
    write_new_or_verify(
        snapshot / "processor_recovery.json",
        (json.dumps(recovery, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(
        snapshot, local_files_only=True, trust_remote_code=False, use_fast=False
    )
    if (
        processor.to_dict()
        .get("benchmark_processor_origin", {})
        .get("original_hf_processor_bytes_recovered")
        is not False
    ):
        raise RuntimeError(
            "Recovered-processor disclosure was lost during production-compatible loading"
        )
    report = {
        "status": "LOCAL_DINOV3_BASE_RESTORED_AND_VERIFIED",
        "model_kind": "dinov3_base",
        "source_snapshot": str(source_snapshot),
        "destination_snapshot": str(snapshot.resolve()),
        "source_files": EXPECTED_SOURCE,
        "checkpoint": checkpoint_evidence(snapshot, "dinov3_base"),
        "processor_recovery": recovery,
        "production_processor_class": type(processor).__name__,
        "network_access_required": False,
        "original_sources_modified": False,
        "gated_access_bypassed": False,
        "restorer_source_sha256": sha256_file(Path(__file__)),
    }
    write_new_or_verify(
        report_path, (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return report


def merge_completed_preparation_report(path: Path, restoration: dict, receipt_path: Path) -> None:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "CHECKPOINT_PREPARATION_COMPLETE":
        raise RuntimeError("The public checkpoint worker is still active; refuse a report race")
    entry = {
        "model_kind": "dinov3_base",
        "status": "PINNED_CHECKPOINT_READY",
        "checkpoint": restoration["checkpoint"],
        "availability_source": "verified_user_transferred_checkpoint",
        "local_restore_receipt": str(receipt_path.resolve()),
        "local_restore_receipt_sha256": sha256_file(receipt_path),
        "processor_recovery": restoration["processor_recovery"],
    }
    previous = next(
        (item for item in report["models"] if item["model_kind"] == "dinov3_base"), None
    )
    if previous == entry:
        return
    if previous is not None:
        report.setdefault("availability_history", []).append(previous)
    report["models"] = [
        item for item in report["models"] if item["model_kind"] != "dinov3_base"
    ] + [entry]
    temporary = path.with_suffix(".local-restore.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    from huggingface_hub.constants import HF_HUB_CACHE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path(HF_HUB_CACHE))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--completed-preparation-report", type=Path)
    args = parser.parse_args()
    report = restore_local_dinov3(args.source_snapshot, args.cache_dir, args.report)
    if args.completed_preparation_report:
        merge_completed_preparation_report(args.completed_preparation_report, report, args.report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "model_kind": "dinov3_base",
                "receipt": str(args.report.resolve()),
                "processor_original_bytes_recovered": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
