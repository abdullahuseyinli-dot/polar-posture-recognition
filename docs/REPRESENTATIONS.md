# DINOv2, DINOv3 and SigLIP2

There are two different representation questions in this project:

1. The POLAR four-class study compares adapted DINOv2/ConvNeXt and frozen DINOv2
   classifiers, culminating in the locked 93.99% test ensemble.
2. A later V-COCO development study compares **DINOv2-B, DINOv3-B and SigLIP2-B**
   with matched views, folds, classifier families and selection budgets.

DINOv3 was not a component of the POLAR ensemble and was not tested on its locked
test set as a new challenger. The later matched screen gives 83.95%, 83.67% and
83.58% macro-F1 respectively. DINOv3's small locomotion-F1 increase does not pass
the promotion gate. DINOv2 stays the reference backbone.

![Matched representations and later fusion](../assets/representation_results.png)

A separate nested-fusion screen finds a **DINO + SigLIP factorized reliability
stack at 86.97% macro-F1**. That positive result illustrates why a representation
can add complementary information without being the strongest standalone model.
It remains nested-development evidence, not a new test score.

## Source and replay map

| Evidence / component | Location |
| --- | --- |
| All nested/spatial/representation metrics | [Source table](../results/vcoco_v3/source_tag_development_metrics.csv) |
| Promotion decisions and adjusted p-values | [Decision record](../results/vcoco_v3/source_tag_promotion_decisions.json) |
| Historical protocol lineage | [Lineage](../results/vcoco_v3/protocol_lineage.json) |
| Revision-pinned feature models | [Implementation](../src/hac/vcoco_v3_representations.py) |
| Representation selection grid | [Grid](../experiments/vcoco_v3_representation_grid.json) |
| Matched evaluator | [Evaluator](../experiments/evaluate_vcoco_v3_representations.py) |
| Nested fusion evaluator | [Evaluator](../experiments/evaluate_vcoco_v3_nested_stacks.py) |
| Gated DINOv3 checkpoint | [Access and recorded hashes](DINOV3_ACCESS.md) |

The source-tag subset of the historical v3 study is imported here. Temporal
confirmation and its other implementation modules remain in ARFTR; the original
lineage file describes the full historical study and therefore mentions that work.
The [origin manifest](../results/project_origin.json) records exactly which files
were imported and their original revisions. Full historical replay requires
non-distributed feature caches and execution locks; this extraction does not
manufacture a replacement lock or download gated weights.
