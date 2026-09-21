# Locked POLAR evaluation

The development-nominated model remains `conservative_fusion` for both tasks. This analysis does not choose a new winner, tune a threshold, or modify historical results.

The four-class test was previously inspected; the nine-class test includes that exposed subset. These results are not fresh independent confirmation or a verified state-of-the-art claim. Detected source groups are not verified subject identities.

## Predeclared promotion checks

| Task | Nominee macro-F1 | Accuracy | Errors | All checks vs prior incumbent |
| --- | ---: | ---: | ---: | --- |
| polar4 | 95.211% | 95.764% | 141 | FAIL |
| polar9 | 94.583% | 94.716% | 369 | FAIL |

Each failed check is reported explicitly; no alternate model is selected.

- polar4: global_holm_p.
- polar9: source_group_delta_ci_lower, global_holm_p.

## Fixed comparison family

All 18 two-sided source-group randomization tests share one Holm correction (10,000 draws, +1 correction). Intervals are paired percentile 95% intervals from 5,000 source-group draws; they are not simultaneous or selection-adjusted. Row-stratified intervals are also recorded in `summary.json`. All seeds are included.

| Task | Reference | Primary | F1 gain (pp) | Group 95% interval (pp) | Rescue / harm | Holm p |
| --- | --- | --- | ---: | --- | ---: | ---: |
| polar9 | prior_incumbent | yes | +0.158 | [-0.091, +0.408] | 44 / 33 | 0.90831 |
| polar9 | replacement_fusion | no | +0.056 | [-0.186, +0.297] | 38 / 35 | 1.00000 |
| polar9 | frozen_dinov2_base | no | +3.691 | [+3.048, +4.327] | 392 / 132 | 0.00180 |
| polar9 | frozen_siglip2_base | no | +2.667 | [+2.067, +3.262] | 330 / 133 | 0.00180 |
| polar9 | frozen_dinov3_base | no | +3.389 | [+2.746, +4.045] | 387 / 148 | 0.00180 |
| polar9 | frozen_convnextv2_base | no | +9.710 | [+8.893, +10.537] | 812 / 126 | 0.00180 |
| polar9 | adapted_siglip2 | no | +1.730 | [+1.170, +2.278] | 246 / 123 | 0.00180 |
| polar9 | adapted_convnextv2 | no | +9.067 | [+8.234, +9.881] | 749 / 114 | 0.00180 |
| polar9 | adapted_dinov2 | yes | +0.679 | [+0.381, +0.987] | 80 / 33 | 0.00180 |
| polar4 | prior_incumbent | yes | +0.465 | [+0.064, +0.890] | 27 / 13 | 0.19858 |
| polar4 | replacement_fusion | no | -0.480 | [-0.965, -0.015] | 20 / 34 | 0.25797 |
| polar4 | frozen_dinov2_base | no | +2.122 | [+1.465, +2.785] | 88 / 20 | 0.00180 |
| polar4 | frozen_siglip2_base | yes | +0.020 | [-0.539, +0.597] | 40 / 38 | 1.00000 |
| polar4 | frozen_dinov3_base | no | +2.156 | [+1.357, +2.945] | 111 / 44 | 0.00180 |
| polar4 | frozen_convnextv2_base | no | +7.462 | [+6.435, +8.541] | 271 / 32 | 0.00180 |
| polar4 | adapted_siglip2 | no | -0.016 | [-0.660, +0.618] | 48 / 48 | 1.00000 |
| polar4 | adapted_convnextv2 | no | +5.818 | [+4.835, +6.832] | 226 / 44 | 0.00180 |
| polar4 | historical_ensemble | yes | +1.223 | [+0.573, +1.940] | 75 / 35 | 0.00420 |

![Fixed comparison deltas](comparison_f1.png)

![Nominated-model confusion matrices](nominee_confusion.png)

Complete classwise precision/recall/F1, probability metrics, fixed-coverage risks, seed sensitivity, runtime records (when supplied), hashes, and all checks are in `summary.json`. No inference-latency benchmark is inferred from training runtime.
