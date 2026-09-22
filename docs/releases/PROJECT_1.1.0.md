# POLAR Posture Recognition 1.1.0

22 September 2026. This release presents the completed four- and nine-class
benchmark, with current results, report, diagrams and public prediction evidence.

| Task | Evaluated nominee F1 | Retained prior F1 | Test rows |
| --- | ---: | ---: | ---: |
| Four classes | 95.211019% | 94.746153% | 3,329 |
| Nine classes | 94.583075% | 94.425294% | 6,984 |

The supported gains over the original four-class ensemble and nine-class
adapted-DINOv2 reference are +1.222686 and +0.678874 percentage points.
The final increments against immediate priors do not pass every locked promotion
condition; both priors remain retained. No diagnostic test winner is substituted.

## Included

- Fixed DINOv2, DINOv3, SigLIP2 and ConvNeXt V2 comparisons, including adapted controls.
- 32 public prediction sets, audited membership and hashes, without images or weights.
- Recalculation of candidate metrics, seed transitions, paired tests and Holm correction.
- Current technical report/PDF, system diagram, result intervals and error matrix.
- Updated citation/archive metadata and CPU validation.

All 330 imported historical artifacts and byte-exact final numerical reports
remain intact. The four-class test is historically exposed and nested in the
nine-class test; no external state-of-the-art, peer-review or Zenodo-deposit claim
is made. This release packaging performs no new training.

[Report](../POLAR_BENCHMARK_REPORT.md) · [Results](../RESULTS.md) ·
[Validation](../VALIDATION.md) · [Provenance](../PROJECT_HISTORY.md).
