# Engineering a verifiable posture benchmark

POLAR Posture Recognition studies when adapting a visual backbone helps more
than fitting a strong classifier on frozen features. The project extends an
original four-class ensemble into an audited nine-class comparison with public
per-example evidence.

## Data and modeling

A source-overlap audit retained 35,007 of 35,324 images without reshuffling
original split membership. Public metadata records retained and excluded rows,
labels, boxes, source groups and image/annotation hashes. It does not assert
subject independence or rule out backbone pretraining overlap.

Frozen DINOv2, DINOv3, SigLIP2 and ConvNeXt V2 share two-view calibrated-RBF
controls. Adapted models preserve person aspect ratio and use bounded top-layer
training. Fusion combines a task-appropriate DINOv2 anchor with frozen and
adapted SigLIP2, with weights fixed before final evaluation.

The result is task-dependent. Frozen SigLIP2 reaches 95.19% four-class macro-F1,
almost the nominee's 95.21%. On nine classes fusion reaches 94.58% against adapted
DINOv2's 93.90%, rescuing 80 errors while harming 33 predictions. Adding every
available model was not the useful mechanism.

## Verifiable decisions

The final panel contains 20 systems and 18 paired comparisons. Gains over the
original four-class ensemble and nine-class adapted DINOv2 have positive paired
intervals and adjusted p-values below 0.05. Smaller gains over the immediate
priors do not pass all locked gates, so the priors remain retained.

Readers can replay 32 probability sets, metrics and paired statistics without
images or checkpoints. That verifies the reported calculations, not a fresh
model replication or an external leaderboard position.

## Release engineering

Historical reports and 330 imported artifacts remain hash-preserved. Current
guides, charts and the new technical report have separate validation. CPU CI
checks code and public evidence without private data. Original datasets and
pretrained models retain their attribution and terms.

[Architecture](ARCHITECTURE.md) · [Results](RESULTS.md) ·
[Technical report](POLAR_BENCHMARK_REPORT.md) · [Reproducibility](REPRODUCIBILITY.md).
