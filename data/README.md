# Data boundary

No POLAR or COCO source images, feature caches or trained weights are distributed.

The main benchmark uses four POLAR classes: sitting, standing, walking and running.
The pre-fit source audit excluded 125 images in 61 cross-split components, leaving
9,958 training, 3,327 validation and 3,329 test images.
[Audit](../results/polar_data_audit.json) · [Protocol](../docs/POLAR_SCALE_STUDY_PROTOCOL.md).

Obtain POLAR from its [publisher](https://doi.org/10.17632/hvnsh7rwz7.1), then follow
the historical preparation recipe in the [reproduction guide](../docs/REPRODUCIBILITY.md).
Dataset and upstream image licences must be assessed independently.

The tracked `manifest.csv` belongs to the **earlier 285-image COCO pilot**, not
the 16,614-image clean POLAR population. Do not train the POLAR experiment from it.
POLAR manifests containing workstation-local image paths remain non-distributed.

V-COCO follows its official image memberships but maps actions to three exclusive
posture classes. This is not the standard agent/role-AP benchmark.
