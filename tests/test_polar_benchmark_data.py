import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from hac.polar_benchmark_data import (
    LABEL_TO_INDEX,
    SOURCE_TO_LABEL,
    _download_source,
    audit_source_groups,
    build_official_manifest,
    canonical_source_name,
    identity_pairs,
    load_official_splits,
    parse_nine_annotation,
    relocate_legacy_cohort,
    validate_archive_member,
    verify_source_file,
    write_json_new,
)


def annotation(path: Path, *, action: str = "bendover", persons: int = 1) -> dict:
    payload = {
        "filename": f"{path.stem}.jpg",
        "originalname": f"[{action}]_getty_p10_123-001.jpg",
        "width": 16,
        "height": 20,
        "persons": [
            {
                "actions": {key: int(key == action) for key in SOURCE_TO_LABEL},
                "bndbox": {"xmin": 1, "ymin": 1, "xmax": 15, "ymax": 19},
            }
            for _ in range(persons)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def test_nine_class_indices_preserve_legacy_four():
    assert list(LABEL_TO_INDEX.items())[:4] == [
        ("sitting", 0),
        ("standing", 1),
        ("walking", 2),
        ("running", 3),
    ]
    assert len(LABEL_TO_INDEX) == 9
    assert SOURCE_TO_LABEL["bendover"] == "bending"


def test_nine_annotation_preserves_verified_source_action(tmp_path):
    path = tmp_path / "p1_00001.json"
    annotation(path)
    row = parse_nine_annotation(path, tmp_path, {"p1_00001": "val"})
    assert row["label"] == "bending" and row["label_index"] == 4
    assert row["split"] == "val"
    assert row["source_name_key"] == "getty_asset_123-001.jpg"


@pytest.mark.parametrize("persons", [0, 2])
def test_ambiguous_person_count_fails_closed(tmp_path, persons):
    path = tmp_path / "p1_00001.json"
    annotation(path, persons=persons)
    with pytest.raises(ValueError, match="exactly one annotated person"):
        parse_nine_annotation(path, tmp_path, {"p1_00001": "train"})


@pytest.mark.parametrize("mutation", ["multiple", "missing", "nonbinary", "unknown"])
def test_ambiguous_action_schema_fails_closed(tmp_path, mutation):
    path = tmp_path / "p1_00001.json"
    payload = annotation(path)
    actions = payload["persons"][0]["actions"]
    if mutation == "multiple":
        actions["sit"] = 1
    elif mutation == "missing":
        actions.pop("sit")
    elif mutation == "nonbinary":
        actions["sit"] = 0.5
    else:
        actions["newclass"] = 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        parse_nine_annotation(path, tmp_path, {"p1_00001": "train"})


def test_source_identity_ignores_class_and_search_page_not_asset_suffix():
    assert canonical_source_name("[sit]_getty_p8_123-001.jpg") == canonical_source_name(
        "[stand]_getty_p72_123-001.jpg"
    )
    assert canonical_source_name("[sit]_getty_p8_123-001.jpg") != canonical_source_name(
        "[sit]_getty_p8_123-002.jpg"
    )


@pytest.mark.parametrize("case", ["within", "between", "trainval"])
def test_split_integrity_fails_closed(tmp_path, case):
    values = {"train": ["a"], "val": ["b"], "test": ["c"], "trainval": ["a", "b"]}
    if case == "within":
        values["train"] = ["a", "a"]
    elif case == "between":
        values["test"] = ["a"]
    else:
        values["trainval"] = ["a", "c"]
    for split, ids in values.items():
        (tmp_path / f"{split}.txt").write_text("\n".join(ids), encoding="utf-8")
    with pytest.raises(ValueError):
        load_official_splits(tmp_path)


def test_source_quarantine_propagates_entire_component_without_moving_splits():
    frame = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d", "e"],
            "split": ["train", "train", "test", "val", "train"],
            "label": ["sitting", "standing", "jumping", "lying", "running"],
        }
    )
    pairs = pd.DataFrame({"left_image_id": ["a", "b"], "right_image_id": ["b", "c"]})
    legacy = pd.DataFrame({"image_id": ["d"], "quarantine_group": ["old"]})
    result = audit_source_groups(frame, pairs, legacy)
    assert result["split"].tolist() == frame["split"].tolist()
    assert result.loc[result["eligible"], "image_id"].tolist() == ["e"]
    assert result.loc[:2, "source_group"].nunique() == 1


def test_identity_grouping_is_label_blind():
    frame = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "sha256": ["sha1", "sha2", "sha1"],
            "source_name_key": ["asset1", "asset1", "asset3"],
            "label": ["sitting", "standing", "jumping"],
        }
    )
    first = identity_pairs(frame)
    frame["label"] = ["jumping", "jumping", "jumping"]
    pd.testing.assert_frame_equal(first, identity_pairs(frame))
    assert set(zip(first["left_image_id"], first["right_image_id"], strict=True)) == {
        ("a", "b"),
        ("a", "c"),
    }


def test_missing_source_pair_is_not_silently_discarded():
    frame = pd.DataFrame({"image_id": ["a"], "split": ["train"]})
    pairs = pd.DataFrame({"left_image_id": ["a"], "right_image_id": ["absent"]})
    with pytest.raises(ValueError, match="absent image"):
        audit_source_groups(frame, pairs, pd.DataFrame(columns=["image_id", "quarantine_group"]))


def test_relocation_preserves_legacy_population_even_if_new_nine_class_audit_flags_it():
    old = pd.DataFrame(
        {
            "image_id": ["a"],
            "split": ["val"],
            "label_4": ["sitting"],
            "sha256": ["known"],
            "bbox_xmin": [1],
            "bbox_ymin": [2],
            "bbox_xmax": [3],
            "bbox_ymax": [4],
            "image_path": ["old/a.jpg"],
            "annotation_path": ["old/a.json"],
        }
    )
    current = old.rename(columns={"label_4": "label"}).assign(
        image_path="new/a.jpg",
        annotation_path="new/a.json",
        label_index=0,
        source_group="source_a",
        quarantined=True,
    )
    result = relocate_legacy_cohort(current, old)
    assert len(result) == 1 and result.loc[0, "split"] == "val"
    assert result.loc[0, "image_path"] == "new/a.jpg"
    assert result.loc[0, "additional_nine_class_quarantined"]
    assert old.loc[0, "image_path"] == "old/a.jpg"
    current.loc[0, "sha256"] = "changed"
    with pytest.raises(ValueError, match="Historical cohort changed"):
        relocate_legacy_cohort(current, old)


def test_official_manifest_checks_pixels_boxes_and_identity(tmp_path):
    annotations = tmp_path / "annotations"
    images = tmp_path / "images"
    splits = tmp_path / "splits"
    for root in (annotations, images, splits):
        root.mkdir()
    for split, identifier in (("train", "a"), ("val", "b"), ("test", "c")):
        annotation(annotations / f"{identifier}.json")
        Image.new("RGB", (16, 20), "red").save(images / f"{identifier}.jpg")
        (splits / f"{split}.txt").write_text(identifier, encoding="utf-8")
    (splits / "trainval.txt").write_text("a\nb", encoding="utf-8")
    frame = build_official_manifest(annotations, images, splits, workers=1, expected=None)
    assert len(frame) == 3 and frame["decode_ok"].all()
    Image.new("RGB", (16, 19), "red").save(images / "a.jpg")
    with pytest.raises(ValueError, match="no rows silently discarded"):
        build_official_manifest(annotations, images, splits, workers=1, expected=None)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "C:/outside", "sub/../../escape", r"..\escape"]
)
def test_unsafe_archive_members_rejected(tmp_path, name):
    with pytest.raises(ValueError):
        validate_archive_member(name, tmp_path)


def test_receipts_are_immutable(tmp_path):
    path = tmp_path / "receipt.json"
    write_json_new(path, {"locked": True})
    with pytest.raises(FileExistsError):
        write_json_new(path, {"locked": False})
    assert json.loads(path.read_text())["locked"]


def test_download_size_and_hash_both_required(tmp_path):
    path = tmp_path / "source.zip"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="size/hash mismatch"):
        verify_source_file(path, {"bytes": 3, "sha256": "incorrect"})


def test_interrupted_provider_download_resumes_verified_range(tmp_path, monkeypatch):
    path = tmp_path / "source.zip"
    partial = tmp_path / "source.zip.partial"
    partial.write_bytes(b"abc")
    spec = {
        "path": str(path),
        "url": "https://example.test/source",
        "bytes": 6,
        "sha256": hashlib.sha256(b"abcdef").hexdigest(),
    }

    def response(request, timeout):
        assert request.get_header("Range") == "bytes=3-"
        result = io.BytesIO(b"def")
        result.status = 206
        result.headers = {"Content-Range": "bytes 3-5/6"}
        return result

    monkeypatch.setattr("hac.polar_benchmark_data.urllib.request.urlopen", response)
    _download_source(spec)
    assert path.read_bytes() == b"abcdef"
    assert not partial.exists()


def test_download_refuses_server_that_ignores_resume_range(tmp_path, monkeypatch):
    path = tmp_path / "source.zip"
    partial = tmp_path / "source.zip.partial"
    partial.write_bytes(b"abc")
    spec = {
        "path": str(path),
        "url": "https://example.test/source",
        "bytes": 6,
        "sha256": hashlib.sha256(b"abcdef").hexdigest(),
    }

    def response(request, timeout):
        result = io.BytesIO(b"abcdef")
        result.status = 200
        result.headers = {}
        return result

    monkeypatch.setattr("hac.polar_benchmark_data.urllib.request.urlopen", response)
    with pytest.raises(ValueError, match="exact byte-range"):
        _download_source(spec)
    assert partial.read_bytes() == b"abc"
    assert not path.exists()
