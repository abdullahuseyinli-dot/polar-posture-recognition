# Model card

| Property | Scope |
| --- | --- |
| System | Development-locked five-component probability ensemble |
| Task | Four-class still-image posture classification |
| Labels | Sitting, standing, walking, running |
| Inputs | RGB image and provided target-person bounding box |
| Primary data | Source-overlap-audited subset of POLAR v1 |
| Clean split sizes | Train 9,958; validation 3,327; test 3,329 |
| Test metrics | Macro-F1 93.99%; accuracy 94.56%; NLL 0.1564; Brier 0.0838 |
| Macro-F1 uncertainty | Class-stratified 95% bootstrap interval 93.12–94.81% |
| Distributed artifacts | Code, tests, protocols, aggregate results, reports and charts |
| Not distributed | Source media, model weights, fitted classifiers and feature caches |

## Intended use

Research and engineering comparison of transfer learning, input views, calibration
and heterogeneous classifiers. The package is an evidence-backed benchmark, not
a turnkey, validated deployment product or a newly pretrained foundation model.

## Evaluation controls

125 images in 61 cross-split source-related components were quarantined before
supervised fitting. Selection, weights and test access were locked. This audit
reduces detected content overlap but does not prove subject- or session-disjoint
generalization; those identities are unavailable.

## Limitations

- Four of POLAR's nine original classes are used, with audited exclusions. A
  best-on-the-internet or standard-leaderboard claim is not established.
- Person boxes are provided. Detection/tracking errors and raw-camera deployment
  are outside the primary evaluation.
- Candidate views and adaptation scopes differ; backbone-only causality is not
  established by their score ordering.
- Strong POLAR performance does not establish cross-domain reliability. The
  source-only V-COCO stress test regressed substantially; the 86.63% follow-up
  includes target-domain training under its own locked protocol.
- V-COCO posture labels are a forced mapping from non-exclusive action tags,
  not the official agent/role-AP task.
- DINOv3 and SigLIP2 screens are development comparisons, not POLAR test updates.
- Subject identity, demographic fairness, privacy impact, surveillance suitability
  and safety-critical reliability were not established. Do not use the reported
  score as authorization for decisions about people.

[Full results](RESULTS.md) · [Original limitations](POLAR_PUBLIC_REPORT.md#12-limitations-and-threats-to-validity)
· [Third-party terms](../THIRD_PARTY_NOTICES.md).
