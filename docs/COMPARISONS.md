# External comparisons and claim scope

Reviewed 22 September 2026. These are related studies, not a numerical leaderboard.
Different labels, inputs, split memberships and metrics prevent a direct ranking.

| Work | Reported result | Task and comparability |
| --- | --- | --- |
| This repository, conservative nominee | 95.21% four-class / 94.58% nine-class macro-F1 | Audited POLAR, given-person-box classification; 3,329 / 6,984 test images; prior retention unchanged |
| Habibi et al., ICHI 2026 | Author slides report POLAR sitting / standing / lying accuracy of 91% / 89% / 88% | Three-class transfer study; the separate 98% F1 headline belongs to a custom video dataset, not nine-class POLAR |
| Ghamati et al., CAPA-AI, RO-MAN 2025 | 89% simulated / 80% real-world novelty adaptation accuracy | Continual novelty detection and adaptation across activities; not full nine-class POLAR macro-F1 |
| Mandita and Rokhman, 2026 | SVM macro-F1 0.76, micro-F1 0.79 | 879 retained POLAR + curated images; multilabel FHP / PK / RSP / normal posture, 80:20 split; different clinical-posture targets |
| Ma and Liang, original POLAR study | No numerical baseline asserted here | Matching original model/protocol result has not been verified sufficiently for a superiority claim |

Primary sources: [Habibi author presentation](https://habibi6010.github.io/assets/paper_presentation/ICHI_2026.pdf),
[CAPA-AI accepted manuscript](https://uhra.herts.ac.uk/id/eprint/26039/7/ROMAN_2025_Novelity_.pdf),
[Mandita and Rokhman article](https://publikasi.mercubuana.ac.id/index.php/format/article/view/39260),
[original POLAR paper](https://doi.org/10.1109/ICSAI48974.2019.9010160),
[POLAR dataset record](https://doi.org/10.17632/hvnsh7rwz7.1).

The Habibi source is an author presentation, not a reproduced baseline. The
Mandita study's multilabel F1 and CAPA-AI's adaptation accuracy are not equivalent
to the single-label macro-F1 measured here. A numerically higher percentage does
not establish that one system is better.

## Claims supported by this repository

- A source-overlap-audited, four-/nine-class known-box POLAR benchmark with public
  probability evidence and a fixed DINOv2/DINOv3/SigLIP2/ConvNeXt V2 panel.
- Measured nominee scores of 95.211019% and 94.583075% macro-F1 under the stated
  protocol, without test-driven model promotion.
- A paired +1.222686-point gain over the historical four-class ensemble and a
  +0.678874-point gain over the nine-class adapted-DINOv2 reference, under the
  specified bootstrap/randomization analysis and complete Holm family.

These are within-repository comparisons, not claims against the external papers.
[Exact evidence and uncertainty](RESULTS.md).

## What is not established

No directly matched external nine-class comparison, official-population rank,
subject-disjoint generalization, pretraining-data independence, clinical validity,
end-to-end person-detection performance or measured deployment advantage is
established. The four-class test is historically exposed and nested in the
nine-class test. Statistical intervals do not undo that history.

An external superiority claim would require matching labels, image membership,
bounding-box access, training data and metric definitions, or clearly separating
a reproduced common-protocol comparison from each paper's original result.
It does not require another author's permission to report this independent
experiment; it requires evidence before claiming to outperform their method.

The release is an independent technical report and software artifact. It does not
assert journal acceptance, peer review, a Zenodo deposit or a DOI for this project.
