"""Record immutable upstream provenance for the separately named benchmark project."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "040d2fdc3a0e5e91f06a550ff1ddfc2d11ba4da8"
SUPPLEMENT = "3c3b2226041f4cf49485da5727cc1b7a2f54e99d"
MUTABLE = {
    "README.md", "CITATION.cff", ".zenodo.json", "CHANGELOG.md", "pyproject.toml",
    ".gitignore", ".gitattributes", ".github/workflows/ci.yml", "THIRD_PARTY_NOTICES.md",
    "docs/PORTFOLIO_ARTICLE.md", "assets/README.md", "data/README.md",
    "experiments/README.md", "output/pdf/README.md", "results/README.md",
    "results/quality/ruff_baseline.json",
}
EXCLUDED = {
    "assets/champion_error_gallery.png", "assets/convnext_small_faithfulness_gallery.jpg",
    "assets/dinov2_small_faithfulness_gallery.jpg",
    "assets/probability_blend_faithfulness_gallery.jpg",
}
TEXT = {".md", ".py", ".ipynb", ".json", ".csv", ".svg", ".txt", ".toml", ".cff", ".yml"}


def normalized(name: str, data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n") if Path(name).suffix in TEXT else data


def git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(source), *args])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    target = ROOT / "results/project_origin.json"
    if target.exists():
        raise FileExistsError("Origin record already exists; it must not be refreshed")
    base_files = git(args.source, "ls-tree", "-r", "--name-only", BASE).decode().splitlines()
    supplement_files = git(args.source, "ls-tree", "-r", "--name-only", SUPPLEMENT).decode().splitlines()
    records = {}
    for revision, names in ((BASE, base_files), (SUPPLEMENT, supplement_files)):
        for name in names:
            path = ROOT / name
            if name in records or name in MUTABLE or name in EXCLUDED or not path.is_file():
                continue
            if revision == SUPPLEMENT and name in base_files:
                continue
            original = normalized(name, git(args.source, "show", f"{revision}:{name}"))
            actual = normalized(name, path.read_bytes())
            if actual != original:
                raise ValueError(f"Imported historical file differs: {name}")
            records[name] = {
                "source_revision": revision,
                "sha256": hashlib.sha256(actual).hexdigest(),
                "normalized_lf": Path(name).suffix in TEXT,
            }
    payload = {
        "schema_version": 1,
        "project": "polar-posture-recognition",
        "project_version": "1.0.0",
        "source_repository": "https://github.com/abdullahuseyinli-dot/arftr",
        "original_repository_name": "human-activity-classification",
        "base_study_tag": "polar-study-v2.0.0",
        "base_revision": BASE,
        "supplement_revision": SUPPLEMENT,
        "scope": "Still-image POLAR/V-COCO source, tests and evidence, plus the later matched representation and fusion screens; no new fits",
        "excluded_third_party_media": sorted(EXCLUDED),
        "mutable_presentation_paths": sorted(MUTABLE),
        "frozen_files": dict(sorted(records.items())),
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"frozen_imported_files": len(records), "base_revision": BASE}))


if __name__ == "__main__":
    main()
