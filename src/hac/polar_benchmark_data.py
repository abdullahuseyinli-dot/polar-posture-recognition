"""Fail-closed, label-blind source audit for the nine-class POLAR benchmark.

This is deliberately separate from the immutable four-class study parser.  The
first four class indices are preserved, official split membership is never
changed, and the original four-class cohort is relocated rather than relabelled.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from hac.polar import (
    enrich_near_pairs,
    inspect_images,
    near_phash_cross_split_pairs,
    sha256_file,
    source_related_pairs,
)

SOURCE_TO_LABEL = {
    "sit": "sitting",
    "stand": "standing",
    "walk": "walking",
    "run": "running",
    "bendover": "bending",
    "jump": "jumping",
    "lying": "lying",
    "squat": "squatting",
    "stretch": "stretching",
}
LABEL_TO_INDEX = {label: index for index, label in enumerate(SOURCE_TO_LABEL.values())}
EXPECTED_SPLIT_COUNTS = {"train": 21194, "val": 7065, "test": 7065}
PROVIDER_URL = "https://data.mendeley.com/datasets/hvnsh7rwz7/1"
SOURCE_FILES = {
    "annotations.zip": (
        "dd8db6ea-ef6e-426d-881c-f1a37607c487",
        10378149,
        "78d18e643b119cb081506c811a0042ba212eabb67fa18eacdc161d8ed5fe26e2",
    ),
    "train.txt": (
        "0c721660-fd81-4635-a6cb-1f22224d1c90",
        211940,
        "e5371ed641ee176b524e7e674afd26826208194ed294783dee8c94a5a4f0692e",
    ),
    "val.txt": (
        "ec118b80-5983-4c9b-8264-935e9c5f47e6",
        70650,
        "59b7dc11f29dbbafd7c0b12a6e6b69d42ed9a4c321297b171ce7ae8ecbe87864",
    ),
    "test.txt": (
        "ab4cb3ee-22ea-485b-bf63-02cf9bd42c09",
        70650,
        "ff6a5e2b043293c02b2831545beafb22126468694234f7c8b00f7a0972cc98c8",
    ),
    "trainval.txt": (
        "15451cc8-9c6c-4781-a33c-0f0ef39f507c",
        282590,
        "c954394196259d17ae2e7efe9dfcf89eed0ed81f40d84012334594982e282bde",
    ),
    "JPEGImages.z01": (
        "b4676fce-e404-4f05-9f31-2c272f567b84",
        734003200,
        "2c70432788d221627ac5a7eeb67a044acf7e39e3f51323cd253e2fa3f946831d",
    ),
    "JPEGImages.z02": (
        "44c09785-edd0-4a74-a766-6bbedc5d285f",
        734003200,
        "ae23efcbd37dc96b3bc052ff703a51ec594c52a0101e8bdb356c382dd1bc3865",
    ),
    "JPEGImages.z03": (
        "97c695c1-557c-4071-b2e8-775f2045fd52",
        734003200,
        "ce622c898f7898e288619632c3e84b490cd40683cb771e0f185fcfb545d68b6a",
    ),
    "JPEGImages.z04": (
        "882f8895-b1f9-4e44-b491-b74cbaacf250",
        734003200,
        "3e7ef5f09fcf6bb7ef9160b43f4a23ee7036308231eba553e61535b07d4860a0",
    ),
    "JPEGImages.zip": (
        "afec6899-5fa9-496e-9e92-c7faa68a6c37",
        153773946,
        "a911de85a38f5d428408f2f377f839caa568d023f6c4b7c11cae46e2e4de4846",
    ),
}


def write_json_new(path: Path, payload: dict) -> None:
    """Never replace an existing audit receipt or previous evidence."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def source_file_spec(name: str, source_root: Path) -> dict:
    identifier, size, digest = SOURCE_FILES[name]
    subdir = "ImageSets" if name.endswith(".txt") else "archives"
    return {
        "filename": name,
        "url": f"https://data.mendeley.com/public-files/datasets/hvnsh7rwz7/files/{identifier}/file_downloaded",
        "bytes": size,
        "sha256": digest,
        "path": str((source_root / subdir / name).resolve()),
    }


def verify_source_file(path: Path, spec: dict) -> None:
    if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
        raise ValueError(f"Provider size/hash mismatch: {path}")


def _download_source(spec: dict) -> dict:
    path = Path(spec["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        verify_source_file(path, spec)
        print(f"Verified existing {path.name}", flush=True)
        return spec
    partial = path.with_suffix(path.suffix + ".partial")
    print(f"Downloading {path.name} ({spec['bytes']:,} bytes)", flush=True)
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > spec["bytes"]:
            raise ValueError(f"Partial is larger than the pinned source: {partial}")
        if offset == spec["bytes"]:
            break
        headers = {"User-Agent": "POLAR-research-restoration/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(spec["url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                if offset and (
                    response.status != 206
                    or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                ):
                    raise ValueError("Provider did not honor the exact byte-range resume request")
                with partial.open("ab" if partial.exists() else "xb") as handle:
                    shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            break
        except OSError:
            if attempt == 2:
                raise
            print(f"Resuming interrupted {path.name}, attempt {attempt + 2}/3", flush=True)
    verify_source_file(partial, spec)
    partial.rename(path)
    print(f"Verified downloaded {path.name}", flush=True)
    return spec


def validate_archive_member(name: str, destination: Path) -> None:
    normalized = name.replace("\\", "/")
    if re.match(r"^[A-Za-z]:", normalized) or normalized.startswith("/"):
        raise ValueError(f"Absolute archive member: {name}")
    target = (destination / normalized).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"Archive member escapes destination: {name}")


def restore_sources(source_root: Path, *, workers: int = 3, seven_zip: Path | None = None) -> dict:
    """Restore only the exact version-one provider bytes with pinned hashes."""
    source_root = source_root.resolve()
    source_root.mkdir(parents=True, exist_ok=True)
    specs = [source_file_spec(name, source_root) for name in SOURCE_FILES]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        downloaded = list(executor.map(_download_source, specs))
    receipt = source_root / "provider_download_receipt.json"
    if not receipt.exists():
        write_json_new(
            receipt,
            {
                "dataset_doi": "10.17632/hvnsh7rwz7.1",
                "provider_url": PROVIDER_URL,
                "file_metadata_verified_utc": "2026-09-20",
                "files": downloaded,
                "hash_provenance": "2026-09-20 official public API; images and annotations also match historical transfer lock",
            },
        )
    annotations = source_root / "Annotations"
    if not annotations.exists():
        annotations.mkdir()
        with zipfile.ZipFile(source_root / "archives" / "annotations.zip") as archive:
            for name in archive.namelist():
                validate_archive_member(name, annotations)
            archive.extractall(annotations)
    image_destination = source_root / "images"
    images = image_destination / "JPEGImages"
    if not images.exists():
        executable = seven_zip or Path(r"C:\Program Files\7-Zip\7z.exe")
        if not executable.is_file():
            raise FileNotFoundError("A multipart-ZIP-capable 7-Zip executable is required")
        image_destination.mkdir(exist_ok=True)
        archive = source_root / "archives" / "JPEGImages.zip"
        listing = subprocess.run(
            [str(executable), "l", "-slt", str(archive)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for section in listing.split("----------", 1)[-1].splitlines():
            if section.startswith("Path = "):
                validate_archive_member(section.removeprefix("Path = "), image_destination)
        print("Extracting hash-verified multipart image archive", flush=True)
        subprocess.run(
            [str(executable), "x", str(archive), f"-o{image_destination}", "-aos", "-bsp0"],
            check=True,
        )
    return {"annotations": annotations, "images": images, "splits": source_root / "ImageSets"}


def load_official_splits(root: Path, *, expected: dict[str, int] | None = None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    by_split: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        ids = [
            line.strip()
            for line in (root / f"{split}.txt").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate identifier within official {split}")
        if expected is not None and len(ids) != expected[split]:
            raise ValueError(f"Unexpected official {split} count: {len(ids)}")
        if set(ids) & lookup.keys():
            raise ValueError(f"Official split overlap in {split}")
        by_split[split] = set(ids)
        lookup.update(dict.fromkeys(ids, split))
    trainval_path = root / "trainval.txt"
    if trainval_path.exists():
        trainval = [
            line.strip()
            for line in trainval_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if (
            len(trainval) != len(set(trainval))
            or set(trainval) != by_split["train"] | by_split["val"]
        ):
            raise ValueError("Official trainval is not the unique union of train and val")
    return lookup


def canonical_source_name(original_name: str) -> str:
    """Getty asset identity, not photographer, person identity or scene identity.

    Search category and result page are stripped, while asset suffixes such as
    -001 versus -002 are preserved.  No labels or class agreement enter grouping.
    """
    name = re.sub(r"^\[[^\]]+\]_?", "", original_name.strip()).casefold()
    name = re.sub(r"^getty_p\d+_", "getty_asset_", name)
    return name


def parse_nine_annotation(path: Path, images_root: Path, split_by_id: dict[str, str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    filename = str(payload["filename"])
    image_id = Path(filename).stem
    if Path(filename).name != filename or image_id != path.stem or image_id not in split_by_id:
        raise ValueError(f"Annotation filename/identity/split mismatch: {path}")
    persons = payload.get("persons", [])
    if len(persons) != 1:
        raise ValueError(f"Expected exactly one annotated person: {image_id}")
    actions = persons[0].get("actions", {})
    if set(actions) != set(SOURCE_TO_LABEL) or any(
        value not in (0, 1) for value in actions.values()
    ):
        raise ValueError(f"Unknown/missing/nonbinary action schema: {image_id}")
    enabled = [source for source, value in actions.items() if value == 1]
    if len(enabled) != 1:
        raise ValueError(f"Expected exactly one active action: {image_id}")
    label = SOURCE_TO_LABEL[enabled[0]]
    box = persons[0]["bndbox"]
    row = {
        "image_id": image_id,
        "image_path": str((images_root / filename).resolve()),
        "annotation_path": str(path.resolve()),
        "annotation_sha256": sha256_file(path),
        "split": split_by_id[image_id],
        "source_label": enabled[0],
        "label": label,
        "label_index": LABEL_TO_INDEX[label],
        "original_name": str(payload.get("originalname", "")),
        "annotated_width": int(payload["width"]),
        "annotated_height": int(payload["height"]),
        "person_count": len(persons),
        "active_action_count": len(enabled),
    }
    for coordinate in ("xmin", "ymin", "xmax", "ymax"):
        value = box[coordinate]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Nonintegral bounding box: {image_id}")
        row[f"bbox_{coordinate}"] = value
    row["source_name_key"] = canonical_source_name(row["original_name"])
    if not row["source_name_key"]:
        raise ValueError(f"Missing source filename: {image_id}")
    return row


def build_official_manifest(
    annotations_root: Path,
    images_root: Path,
    splits_root: Path,
    *,
    workers: int = 8,
    expected: dict[str, int] | None = EXPECTED_SPLIT_COUNTS,
) -> pd.DataFrame:
    split_by_id = load_official_splits(splits_root, expected=expected)
    paths = sorted(annotations_root.glob("*.json"))
    if {path.stem for path in paths} != set(split_by_id):
        raise ValueError("Annotation identities do not exactly cover official splits")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        rows = list(
            executor.map(lambda path: parse_nine_annotation(path, images_root, split_by_id), paths)
        )
    frame = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if frame["image_id"].duplicated().any():
        raise ValueError("Duplicate image identity")
    if {path.stem for path in images_root.glob("*.jpg")} != set(split_by_id):
        raise ValueError("Image identities do not exactly cover official splits")
    print(f"Hashing/decoding {len(frame):,} original POLAR images", flush=True)
    inspections = inspect_images(
        [Path(path) for path in frame["image_path"]], workers=workers, show_progress=True
    )
    frame = pd.concat([frame, pd.DataFrame(inspections)], axis=1)
    frame["image_sha256"] = frame["sha256"]
    frame["dimension_match"] = (frame["actual_width"] == frame["annotated_width"]) & (
        frame["actual_height"] == frame["annotated_height"]
    )
    frame["bbox_valid"] = (
        (frame["bbox_xmin"] >= 0)
        & (frame["bbox_ymin"] >= 0)
        & (frame["bbox_xmax"] <= frame["actual_width"])
        & (frame["bbox_ymax"] <= frame["actual_height"])
        & (frame["bbox_xmax"] > frame["bbox_xmin"])
        & (frame["bbox_ymax"] > frame["bbox_ymin"])
    )
    invalid = ~frame["decode_ok"] | ~frame["dimension_match"] | ~frame["bbox_valid"]
    if invalid.any():
        raise ValueError(
            f"Image/box/decode audit failed, no rows silently discarded: {frame.loc[invalid, 'image_id'].tolist()[:20]}"
        )
    frame["bbox_area_fraction"] = (
        (frame["bbox_xmax"] - frame["bbox_xmin"])
        * (frame["bbox_ymax"] - frame["bbox_ymin"])
        / (frame["actual_width"] * frame["actual_height"])
    )
    return frame


class SourceComponents:
    def __init__(self, identifiers: list[str]) -> None:
        self.parent = dict(zip(identifiers, identifiers, strict=True))

    def find(self, identifier: str) -> str:
        if identifier not in self.parent:
            raise ValueError(f"Source pair refers to absent image: {identifier}")
        while self.parent[identifier] != identifier:
            self.parent[identifier] = self.parent[self.parent[identifier]]
            identifier = self.parent[identifier]
        return identifier

    def union(self, left: str, right: str) -> None:
        first, second = sorted((self.find(left), self.find(right)))
        self.parent[second] = first


def identity_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field, rule in (
        ("sha256", "exact_sha256"),
        ("source_name_key", "canonical_getty_asset_or_source_name"),
    ):
        for _, group in frame.groupby(field, sort=True):
            ids = sorted(group["image_id"].astype(str))
            rows.extend(
                {"left_image_id": ids[0], "right_image_id": other, "audit_source": rule}
                for other in ids[1:]
            )
    return pd.DataFrame(rows, columns=["left_image_id", "right_image_id", "audit_source"])


def audit_source_groups(
    frame: pd.DataFrame, pairs: pd.DataFrame, legacy_quarantine: pd.DataFrame
) -> pd.DataFrame:
    components = SourceComponents(frame["image_id"].astype(str).tolist())
    for pair in pairs.itertuples(index=False):
        components.union(str(pair.left_image_id), str(pair.right_image_id))
    legacy_ids = set(legacy_quarantine["image_id"].astype(str))
    for _, group in legacy_quarantine.groupby("quarantine_group"):
        ids = sorted(group["image_id"].astype(str))
        for other in ids[1:]:
            components.union(ids[0], other)
    if not legacy_ids <= set(frame["image_id"]):
        raise ValueError("Historical quarantine refers to absent official image")
    result = frame.copy()
    result["source_group"] = [
        f"source_{components.find(identifier)}" for identifier in result["image_id"]
    ]
    split_counts = result.groupby("source_group")["split"].nunique()
    crossing = set(split_counts[split_counts > 1].index)
    legacy_groups = set(result.loc[result["image_id"].isin(legacy_ids), "source_group"])
    result["legacy_quarantined"] = result["image_id"].isin(legacy_ids)
    result["quarantined"] = result["source_group"].isin(crossing | legacy_groups)
    result["eligible"] = ~result["quarantined"]
    result["exclusion_reason"] = ""
    result.loc[result["quarantined"], "exclusion_reason"] = (
        "cross_split_source_related_or_inherited_quarantine"
    )
    clean = result[result["eligible"]]
    if clean.groupby("source_group")["split"].nunique().gt(1).any():
        raise AssertionError("Source component survived across clean split boundaries")
    return result


def validate_legacy_cohort(frame: pd.DataFrame, legacy: pd.DataFrame) -> None:
    lookup = frame.set_index("image_id")
    if not lookup.index.is_unique:
        raise ValueError("Current manifest contains duplicate identifiers")
    if legacy["image_id"].duplicated().any() or not set(legacy["image_id"]) <= set(lookup.index):
        raise ValueError("Historical cohort has duplicate/unknown identifiers")
    current = lookup.loc[legacy["image_id"]].reset_index(drop=True)
    old = legacy.reset_index(drop=True)
    for old_column, new_column in [
        ("label_4", "label"),
        ("split", "split"),
        ("sha256", "sha256"),
        *[(f"bbox_{key}", f"bbox_{key}") for key in ("xmin", "ymin", "xmax", "ymax")],
    ]:
        if not old[old_column].astype(str).equals(current[new_column].astype(str)):
            raise ValueError(f"Historical cohort changed: {old_column}")


def relocate_legacy_cohort(frame: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
    validate_legacy_cohort(frame, legacy)
    result = legacy.copy()
    lookup = frame.set_index("image_id")
    for column in ("image_path", "annotation_path", "source_group", "label", "label_index"):
        result[column] = result["image_id"].map(lookup[column])
    result["image_sha256"] = result["sha256"]
    result["additional_nine_class_quarantined"] = result["image_id"].map(lookup["quarantined"])
    return result


def split_label_counts(frame: pd.DataFrame) -> dict:
    return {
        split: {
            label: int(((frame["split"] == split) & (frame["label"] == label)).sum())
            for label in LABEL_TO_INDEX
        }
        for split in ("train", "val", "test")
    }


def prepare_benchmark_data(
    output_dir: Path,
    annotations_root: Path,
    images_root: Path,
    splits_root: Path,
    legacy_data_root: Path,
    data_lock_path: Path,
    *,
    workers: int = 8,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "polar9_data_audit.json").exists() or (
        output_dir / "polar9_official_manifest.csv"
    ).exists():
        raise FileExistsError("Audit artifacts already exist; use a new output directory")
    started = time.monotonic()
    lock = json.loads(data_lock_path.read_text(encoding="utf-8"))
    legacy_clean_path = legacy_data_root / "polar_clean_manifest.csv"
    if sha256_file(legacy_clean_path) != lock["clean_manifest_sha256"]:
        raise ValueError("Legacy clean manifest does not match its locked SHA256")
    legacy_full = pd.read_csv(legacy_data_root / "polar_target_manifest.csv", keep_default_na=False)
    if sha256_file(legacy_data_root / "polar_target_manifest.csv") != lock["full_manifest_sha256"]:
        raise ValueError("Legacy full manifest does not match its locked SHA256")
    frame = build_official_manifest(annotations_root, images_root, splits_root, workers=workers)
    validate_legacy_cohort(frame, legacy_full)
    print("Finding label-blind all-nine-class cross-split pHash candidates", flush=True)
    # Legacy helper column naming is adapted only for candidate-table metadata;
    # neither retrieval nor the confirmation rule consumes labels.
    audit_view = frame.rename(columns={"label": "label_4"})
    near = near_phash_cross_split_pairs(audit_view, max_distance=6)
    near = enrich_near_pairs(near, workers=workers)
    confirmed = source_related_pairs(near, minimum_correlation=0.90)
    confirmed["audit_source"] = "phash_le6_and_grayscale_correlation_ge0.90"
    identity = identity_pairs(frame)
    pairs = pd.concat(
        [identity, confirmed[["left_image_id", "right_image_id", "audit_source"]]],
        ignore_index=True,
    )
    quarantine_path = legacy_data_root / "quarantine.csv"
    historical_quarantine = pd.read_csv(quarantine_path, keep_default_na=False)
    frame = audit_source_groups(frame, pairs, historical_quarantine)
    clean = frame[frame["eligible"]].copy()
    legacy_clean = pd.read_csv(legacy_clean_path, keep_default_na=False)
    legacy = relocate_legacy_cohort(frame, legacy_clean)
    audited_four = legacy[~legacy["additional_nine_class_quarantined"]].copy()
    if len(legacy) != int(lock["clean_rows"]):
        raise ValueError("Historical clean row count changed")
    tables = {
        "polar9_official_manifest.csv": frame,
        "polar9_official_development_manifest.csv": frame[frame["split"] != "test"],
        "polar9_official_test_manifest.csv": frame[frame["split"] == "test"],
        "feature_pool_development.csv": frame[frame["split"] != "test"],
        "polar9_clean_manifest.csv": clean,
        "polar9_development_manifest.csv": clean[clean["split"] != "test"],
        "polar9_train_manifest.csv": clean[clean["split"] == "train"],
        "polar9_val_manifest.csv": clean[clean["split"] == "val"],
        "polar9_test_manifest.csv": clean[clean["split"] == "test"],
        "polar9_quarantine.csv": frame[frame["quarantined"]],
        "polar9_near_cross_split_candidates.csv": near,
        "polar9_confirmed_source_pairs.csv": pairs,
        "polar4_legacy_clean_manifest.csv": legacy,
        "polar4_legacy_development_manifest.csv": legacy[legacy["split"] != "test"],
        "polar4_legacy_test_manifest.csv": legacy[legacy["split"] == "test"],
        "polar4_source_audited_clean_manifest.csv": audited_four,
        "polar4_source_audited_development_manifest.csv": audited_four[
            audited_four["split"] != "test"
        ],
        "polar4_source_audited_test_manifest.csv": audited_four[audited_four["split"] == "test"],
    }
    artifacts = {}
    for filename, table in tables.items():
        path = output_dir / filename
        if path.exists():
            raise FileExistsError(path)
        table.to_csv(path, index=False)
        artifacts[filename] = {"sha256": sha256_file(path), "rows": len(table)}
    audit = {
        "status": "LOCKED_BEFORE_NEW_BENCHMARK_FITTING",
        "dataset_doi": "10.17632/hvnsh7rwz7.1",
        "class_to_index": LABEL_TO_INDEX,
        "source_to_label": SOURCE_TO_LABEL,
        "official_rows": len(frame),
        "official_counts": split_label_counts(frame),
        "source_audited_rows": len(clean),
        "source_audited_counts": split_label_counts(clean),
        "quarantine_rows": int(frame["quarantined"].sum()),
        "quarantine_components": int(frame.loc[frame["quarantined"], "source_group"].nunique()),
        "historical_quarantine_rows_inherited": len(historical_quarantine),
        "historical_quarantine_sha256": sha256_file(quarantine_path),
        "exact_or_canonical_source_identity_edges": len(identity),
        "phash_candidates": len(near),
        "phash_confirmed_pairs": len(confirmed),
        "decode_failures": 0,
        "dimension_mismatches": 0,
        "invalid_or_clipped_boxes": 0,
        "ambiguous_person_or_action_records": 0,
        "legacy_four_class_rows": len(legacy),
        "legacy_four_class_additional_nine_class_quarantine_flags": int(
            legacy["additional_nine_class_quarantined"].sum()
        ),
        "legacy_four_class_membership_changed": False,
        "additional_source_audited_four_class_rows": len(audited_four),
        "additional_source_audited_four_class_counts": split_label_counts(audited_four),
        "legacy_data_lock_sha256": sha256_file(data_lock_path),
        "preparation_module_sha256": sha256_file(Path(__file__)),
        "official_split_file_sha256": {
            split: sha256_file(splits_root / f"{split}.txt")
            for split in ("train", "val", "test", "trainval")
        },
        "official_split_membership_moved": False,
        "new_test_predictions_read": False,
        "test_labels_used_for_selection": False,
        "test_exposure": "four-class subset previously evaluated; nine-class test is not wholly unseen",
        "limitations": [
            "Known annotated target boxes: classification, not person detection or action detection AP.",
            "No verified subject, photographer-session or scene-disjoint identifiers.",
            "Exact/source-name plus cross-split pHash radius6/correlation0.90 audit is label-blind but cannot prove absence of all near duplicates.",
            "Historical four-class embedding-confirmed quarantine is inherited; no fresh all-nine-class embedding retrieval audit has yet run.",
            "Non-identical same-split near duplicates are not exhaustively grouped; source_group is not a person/scene identifier.",
            "Nine-class audited cohort differs from original official population; report official and audited protocols separately.",
        ],
        "artifacts": artifacts,
        "runtime_seconds": time.monotonic() - started,
    }
    write_json_new(output_dir / "polar9_data_audit.json", audit)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return audit
