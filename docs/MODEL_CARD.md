# Model card

## Intended use and inputs

POLAR Posture Recognition 1.1.0 is a research benchmark for still-image posture
classification given an RGB image and an annotated target-person box. It is not
a validated clinical, safety-critical or surveillance system.

Output is a probability vector and argmax class. Four-class order: sitting,
standing, walking, running. Nine-class order appends bending, jumping, lying,
squatting, stretching. No action label, source-group identity, human-review
category or track is an inference input. Predicted-box and end-to-end detection
performance have not been measured.

## Models

Frozen DINOv2-B, DINOv3-B, SigLIP2-B and ConvNeXt V2-B provide two-view features
for calibrated RBF heads. Partially adapted DINOv2, SigLIP2 and ConvNeXt V2 are
separate controls. The nominated fusion and retained prior combine DINOv2 and
SigLIP2 probabilities using development-fixed weights. DINOv3 is a comparator,
not a fusion component. [Architecture and recipe](ARCHITECTURE.md).

## Data

| Population | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Original nine-class POLAR | 21,194 | 7,065 | 7,065 |
| Source-audited nine-class cohort | 21,057 | 6,966 | 6,984 |
| Source-audited four-class subset | 9,958 | 3,327 | 3,329 |

The audit quarantined 317 images across 154 cross-split source components without
reassigning split membership. Final refits use train plus validation after locking
models, hyperparameters and epoch counts. The previously inspected four-class
test is nested in the nine-class test: this is not a wholly fresh holdout.

Canonical-source and confirmed perceptual-hash auditing cannot prove subject
independence or absence of every duplicate. Pretraining overlap has not been ruled
out. [Audit and membership](../results/polar_20260921/README.md).

## Evaluation and retention

| Task | Nominee macro-F1 | Accuracy | Errors | Source-group 95% F1 interval | Retained prior F1 |
| --- | ---: | ---: | ---: | --- | ---: |
| Four classes | 95.211019% | 95.764494% | 141 | 94.4252–95.9449% | 94.746153% |
| Nine classes | 94.583075% | 94.716495% | 369 | 94.0306–95.1102% | 94.425294% |

Neither nominee passed every prespecified promotion gate against its immediate
prior; both priors remain retained. The 95.69% four-class replacement control is
archived without post-test selection. [Complete results](RESULTS.md).

## Limitations and availability

- Static images can leave posture boundaries ambiguous.
- Intervals describe this cohort, not all training, domain or historical selection uncertainty.
- No verified subject, camera, session or demographic-disjoint evaluation.
- Protected-group performance, clinical validity and deployment latency were not measured.
- Matched local fitting budgets do not equalize backbone pretraining, capacity or FLOPs.
- DINOv3's missing original processor configuration required a documented 224-pixel
  reconstruction; provider preprocessing parity is not established.
- Source code, derived membership and probabilities are public; images, caches
  and checkpoints are not redistributed. Prediction verification is not model replay.

See [reproducibility](REPRODUCIBILITY.md), [third-party terms](../THIRD_PARTY_NOTICES.md)
and [comparison scope](COMPARISONS.md). No standard leaderboard rank, peer review
or Zenodo DOI is implied.
