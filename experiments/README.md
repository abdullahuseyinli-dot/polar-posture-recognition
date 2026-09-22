# Experiment entry points

## Current four-/nine-class benchmark

| Phase | Entry points |
| --- | --- |
| Source audit and nine-class manifests | `tools/prepare_polar_benchmark_data.py` |
| CUDA frozen features and screens | `cache_polar_benchmark_features.py`, `screen_polar_benchmark.py` |
| Person-preserving adaptation | `train_polar_benchmark.py`, `confirm_polar_benchmark_recipe.py` |
| Bounded SigLIP2/ConvNeXt V2 development | `train_polar_bounded.py`, `summarize_polar_bounded.py` |
| Locked final preparation | `prepare_polar_locked.py` |
| Final neural/head refits | `train_polar_locked.py`, `fit_polar_locked_head.py` |
| Sealed test features and predictions | `cache_polar_locked_test_features.py`, `evaluate_polar_locked.py` |
| Paired final statistics | `analyze_polar_locked.py` |

The phase is complete. For read-only verification use the public
[prediction replay](../docs/REPRODUCIBILITY.md), not training or test-opening
commands. Original local configurations are documented in the
[locked protocol](../docs/research/20260921_final_evaluation/PROTOCOL.md).

## Historical studies

Historical implementations are preserved for provenance, not silently rewritten
as a new protocol. Do not run queues or test-opening scripts just to inspect results.

| Phase | Entry points |
| --- | --- |
| POLAR preparation | `tools/prepare_polar.py`, `tools/audit_polar_embeddings.py` |
| Training and confirmations | `train_polar_candidate.py`, `run_polar_confirmation_queue.py` |
| Frozen representation classifiers | `cache_polar_features.py`, `screen_polar_embedding_classifiers.py` |
| Lock, refit, evaluate | `lock_polar_final_selection.py`, `fit_polar_final_model.py`, `fit_polar_final_probe.py`, `evaluate_polar_final.py` |
| V-COCO person-centric follow-up | `cache_vcoco_v2_features.py`, `fit_vcoco_v2_final_stack.py`, `evaluate_vcoco_v2_final_test.py` |
| DINOv2/DINOv3/SigLIP2 screen | `cache_vcoco_v3_features.py`, `evaluate_vcoco_v3_representations.py` |
| Later fusion and spatial controls | `evaluate_vcoco_v3_nested_stacks.py`, `evaluate_vcoco_v3_spatial.py` |

All filenames in the table are under this directory unless prefixed with `tools/`.
The original execution locks and local caches are required for historical replay.
Use `--help` to inspect interfaces, then [read the reproduction limits](../docs/REPRODUCIBILITY.md).
No training or model downloads occur in the public aggregate check.
