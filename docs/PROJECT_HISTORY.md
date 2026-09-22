# Project history and provenance

**POLAR Posture Recognition 1.0.0** is the first standalone project package.
It was separated from the human-activity-classification research repository on
20 September 2026. That repository is now
[ARFTR](https://github.com/abdullahuseyinli-dot/arftr), the companion temporal
architecture project and complete historical Git record.

| Imported work | Original identity | Source revision |
| --- | --- | --- |
| POLAR and person-level V-COCO code, reports and evidence | `polar-study-v2.0.0`, preserving study v1 and v2 | `040d2fdc3a0e5e91f06a550ff1ddfc2d11ba4da8` |
| DINOv3/SigLIP2 and nested source-tag screens | Later research continuation | `3c3b2226041f4cf49485da5727cc1b7a2f54e99d` |

[Machine-readable origin and hashes](../results/project_origin.json).
The extraction preserves imported numerical evidence and implementation bytes
(text hashes normalize LF). Four COCO/Flickr qualitative composites were excluded
from the new repository; aggregate diagrams and charts are included.

## Version namespaces

Project **1.1.0**, dated 22 September 2026, adds the completed four-/nine-class
comparison, public prediction evidence, current technical report and refreshed
presentation. It does not replace the original report versions or change the
330 imported-file hashes. All final-test promotion decisions remain unchanged.
[Release notes](releases/PROJECT_1.1.0.md).

Project `1.0.0` is not a rewrite of study v1, v2 or v3. Original report PDFs, result
manifests and study release notes keep their original dates and identifiers.
The companion ARFTR also begins at independent project version 1.0.0. Neither new
version means a new fit or changed result. No existing tag is changed and no
Zenodo DOI is asserted.

## Ownership and maintenance

This repository owns the maintained still-image benchmark presentation. ARFTR owns
the temporal architecture and its continuation experiments. Shared historical `hac`
modules are preserved snapshots, not automatically synchronized dependencies.
Use separate virtual environments. Any future behavioral change needs its own
version and tests rather than silently refreshing a historical origin hash.

Legacy documents may refer to the old GitHub name or local execution directories.
Those references remain provenance; current guides link the two named projects.
The [origin builder](../tools/build_project_origin.py) is an extraction utility and
refuses to overwrite its existing record. It is not a routine release-refresh step.
