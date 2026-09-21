# Completed evidence to bounded next move

Date: 2026-09-21. See [evidence review](EVIDENCE_REVIEW.md),
[machine-readable graph](knowledge_graph.json), and the
[parent graph](../20260920_polar_benchmark/knowledge_graph.json). Scores below are
selected **validation** results; no new test score is claimed.

```mermaid
flowchart TB
    P[Prior four-class study and all-nine-class source audit]
    P --> R[All eight frozen winners: two-view RBF]
    P --> D[Person-preserving DINOv2 recipe: +2.505 pp at seed 42]
    D --> A[Three-seed adapted DINOv2: 93.471% nine-class F1]
    R --> S[Frozen SigLIP2: strongest frozen family]
    A --> I9[Nine-class incumbent fusion: 94.005%; 408 errors]
    S --> I9
    S --> I4[Four-class incumbent fusion: 95.151%; 146 errors]
    R --> I4
    U[Uniform-all blends lose 47 / 16 net corrections]
    U --> B[Keep finite fusion family; retain incumbent action]
    I9 --> C[Better NLL/Brier does not imply better ECE or selective risk]
    I4 --> C

    D -.-> H[Hypothesis: person-preserving SigLIP adaptation helps]
    S -.-> H
    H --> N[Six SigLIP2 fits: two tasks x three seeds]
    P --> V[Six ConvNeXtV2 last-stage control fits]
    N --> F[Only replacement and conservative new fusions]
    V --> G[Standalone control; no automatic ConvNeXt fusion]
    B --> F
    F --> T[Locked gain, seed, class and probability-quality gates]
    G --> T
    C --> T
    T --> O[Nominate eligible finalist or retain incumbent]
    O --> Z[Stop after twelve fits; review and lock final evaluation]
    L[Adaptive validation; partly exposed test; original paper comparator unknown]
    L --> Z
```

The conservative fusion keeps the old DINO anchor at 50%, frozen SigLIP2 at 25%
and adapted SigLIP2 at 25%. It tests adaptation without discarding all the frozen
information that created the current nine-class gain. If no candidate passes the
predeclared gates, the correct outcome is to retain the incumbent and finish the
fixed comparison, not start another validation sweep.
