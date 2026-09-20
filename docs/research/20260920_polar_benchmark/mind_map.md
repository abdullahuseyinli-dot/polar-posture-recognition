# Evidence-to-experiment map

Date: 2026-09-20. Solid relationships below are backed by the linked artifacts;
proposed runs and cross-domain extrapolations are marked as hypotheses. See
[evidence and comparators](EVIDENCE_AND_COMPARATORS.md) for numbers, qualifications
and sources, and [machine-readable graph](knowledge_graph.json) for typed edges.

```mermaid
flowchart TB
    P[Original POLAR: nine classes, 35,324 images]
    P --> Q[Historical audited four-class subset]
    Q --> D[9,958 train / 3,327 validation / 3,329 test]
    D --> R[Frozen DINOv2 multilayer + nonlinear head]
    D --> A[Adapted ConvNeXt and DINOv2]
    R --> C[Complementary development errors]
    A --> C
    C --> E[Locked ensemble: 93.9883% macro-F1]
    R --> B[Best component: 92.7354% macro-F1]
    B -->|paired gain +1.2529 pp| E
    E --> X[Four-class test already exposed]

    V[Separate V-COCO experiments]
    V --> N[Newer backbone alone did not win]
    V --> F[DINO plus SigLIP fusion helped nested development]
    V --> S[Domain shift: scale, annotation semantics and context]
    N -.->|hypothesis: compare, do not assume| T[Matched modern challenger panel]
    F -.->|hypothesis: complementarity transfers| G[Simple fusion versus best individual]
    S --> I[Same rows, views, boxes and declared adaptation budgets]

    P --> J[Completed all-class audit: 35,007 clean / 317 quarantined]
    J --> JA[21,057 train / 6,966 validation / 6,984 test]
    J -->|zero additional exclusions| Q
    X --> W[Nine-class test is partly historically exposed]
    JA --> K[Nine-class foundation-weight reference]
    K --> T
    I --> T
    T --> G
    G --> L[Development selection and hash lock]
    L --> M[Fixed evaluation and paired uncertainty]
    W --> M
    M --> O[Measured ranking within the declared scope]

    U[Original paper numerical baseline still unverified]
    U --> Z[External SOTA comparison remains unresolved]
    O --> Z
    H[ARFTR failed corrections: separate video protocol]
    H -->|methodological lesson only| Y[No selected tiny gains, no test-driven retries]
    Y --> L
    EE[65 CUDA-environment tests passed; cache and artifact controls reviewed]
    EE --> L
    DR[DINOv3-B transferred weights hash-verified; processor recovery disclosed]
    DR --> T
```

The immediate move is **audited nine-class reference plus a bounded modern
challenger panel**, with reusable CUDA features and separate four-class heads.
This tests whether the proven head/fusion engineering remains competitive and
builds a credible nine-class baseline. It does not assume that nine labels are
easier than four, or that a new model name guarantees an improvement.

The nine-class audit found no additional exclusions in the retained four-class
cohort. Its known source groups do not establish subject/session independence;
all-nine-class embedding retrieval remains unperformed. Every completed training
run must produce a passing selected-checkpoint replay receipt. These engineering
checks are not new model scores, and the original paper's numerical baseline
remains unverified.

DINOv3-B's verified local foundation weights remove its access blocker. The
missing original processor file is recovered through a declared pinned 224-pixel
processor recipe, not represented as original processor-byte parity; this does
not establish DINOv3-L availability.
