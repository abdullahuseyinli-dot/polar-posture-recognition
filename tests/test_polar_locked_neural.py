"""Final refit tests use only tiny CPU models and synthetic development locks."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import train_polar_locked as training
from PIL import Image

import hac.polar_locked_neural as locked
from hac.polar import sha256_file
from hac.polar_benchmark import canonical_hash
from hac.polar_models import PinnedDinoV2Classifier


def make_manifest(tmp_path, classes=4):
    rows = []
    for split in ("train", "val"):
        for label in range(classes):
            for example in range(3):
                image_id = f"{split}-{label}-{example}"
                rows.append(
                    {
                        "image_id": image_id,
                        "image_path": str(tmp_path / f"{image_id}.png"),
                        "split": split,
                        "label": f"class_{label}",
                        "label_index": label,
                        "source_group": image_id,
                        "image_sha256": "a" * 64,
                        "bbox_xmin": 0,
                        "bbox_ymin": 0,
                        "bbox_xmax": 8,
                        "bbox_ymax": 16,
                    }
                )
    manifest = tmp_path / "development.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return {
        "classes": classes,
        "class_names": [f"class_{label}" for label in range(classes)],
        "development_manifest": str(manifest),
        "development_manifest_sha256": sha256_file(manifest),
        "development_rows": len(rows),
    }


def make_lock(tmp_path):
    root, output = tmp_path / "repo", tmp_path / "run"
    output.mkdir()
    sources = {}
    for relative in locked.required_source_paths():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Synthetic unit-test source\n", encoding="utf-8")
        sources[relative] = sha256_file(path)
    specification = make_manifest(tmp_path)
    fit = {
        "seeds": [42, 52, 62],
        "epochs": 7,
        "epoch_selection": "median_of_three_development_best_epochs",
        "schedule_horizon_epochs": 20,
        "output_dir_pattern": "polar4/neural/{model_kind}/seed{seed}",
    }
    specification["neural_fits"] = {"siglip2_base": fit}
    recipe = {
        key: locked.FINAL_FIXED[key]
        for key in (
            "batch_size",
            "accumulation",
            "head_lr",
            "backbone_lr",
            "weight_decay",
            "dropout",
            "gradient_clip",
            "warmup_fraction",
            "schedule_horizon_epochs",
        )
    }
    recipe.update(
        {
            "loss": "cross_entropy",
            "early_stopping": False,
            "precision": "cuda_bfloat16",
            "view": "person_context_25",
            "augmentation": "person_safe_mild",
            "workers": 4,
        }
    )
    value = {
        "status": "POLAR_FINAL_EVALUATION_LOCKED",
        "output_root": str(output),
        "implementation": sources,
        "tasks": {"polar4": specification},
        "neural_training": recipe,
        "test_manifest": "THIS_MUST_NEVER_BE_OPENED",
    }
    path = output / "final_selection_lock.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    arguments = {
        "task": "polar4",
        "model_kind": "siglip2_base",
        "seed": 42,
        "output_dir": output / "polar4" / "neural" / "siglip2_base" / "seed42",
        "root": root,
    }
    return path, value, arguments


def test_lock_binds_every_neural_source_and_exact_production_destination(tmp_path):
    path, value, arguments = make_lock(tmp_path)
    result, specification, fit = locked.read_fit_lock(path, **arguments)
    assert result == value and fit["epochs"] == 7 and specification["classes"] == 4
    assert "experiments/train_polar_locked.py" in locked.required_source_paths()
    assert "src/hac/polar_bounded_models.py" in locked.required_source_paths()
    assert not Path(value["test_manifest"]).exists()
    with pytest.raises(ValueError, match="immutable locked destination"):
        locked.read_fit_lock(path, **{**arguments, "output_dir": tmp_path / "other"})
    source = arguments["root"] / "experiments/train_polar_locked.py"
    source.write_text("# Changed code", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source implementation drift"):
        locked.read_fit_lock(path, **arguments)


@pytest.mark.parametrize(
    "mutation",
    [
        "unsealed",
        "epochs_zero",
        "epochs_bool",
        "wrong_seed_set",
        "compressed_schedule",
        "changed_recipe",
        "missing_source",
        "wrong_root",
        "path_escape",
    ],
)
def test_invalid_locks_are_rejected(tmp_path, mutation):
    path, value, arguments = make_lock(tmp_path)
    fit = value["tasks"]["polar4"]["neural_fits"]["siglip2_base"]
    if mutation == "unsealed":
        value["status"] = "DRAFT"
    elif mutation == "epochs_zero":
        fit["epochs"] = 0
    elif mutation == "epochs_bool":
        fit["epochs"] = True
    elif mutation == "wrong_seed_set":
        fit["seeds"] = [42, 52, 63]
    elif mutation == "compressed_schedule":
        fit["schedule_horizon_epochs"] = 7
    elif mutation == "changed_recipe":
        value["neural_training"]["head_lr"] = 0.1
    elif mutation == "missing_source":
        value["implementation"].pop("experiments/train_polar_locked.py")
    elif mutation == "wrong_root":
        value["output_root"] = str(tmp_path)
    else:
        fit["output_dir_pattern"] = "../escape/{model_kind}/seed{seed}"
    path.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)):
        locked.read_fit_lock(path, **arguments)


def test_smoke_must_be_separate_and_under_locked_output(tmp_path):
    path, value, arguments = make_lock(tmp_path)
    output = Path(value["output_root"])
    locked.read_fit_lock(
        path, **{**arguments, "output_dir": output / "smokes" / "polar4_siglip2_base"}, smoke=True
    )
    for target in (arguments["output_dir"], output / "other", tmp_path / "smoke"):
        with pytest.raises(ValueError, match="Engineering smoke"):
            locked.read_fit_lock(path, **{**arguments, "output_dir": target}, smoke=True)


def test_test_rows_rejected_before_any_label_columns_are_parsed(tmp_path, monkeypatch):
    specification = make_manifest(tmp_path)
    path = Path(specification["development_manifest"])
    frame = pd.read_csv(path)
    frame.loc[0, "split"] = "test"
    frame.to_csv(path, index=False)
    calls, original = [], pd.read_csv

    def spy(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", spy)
    with pytest.raises(ValueError, match="before parsing any labels"):
        locked.load_locked_development(specification)
    assert len(calls) == 1 and calls[0]["usecols"] == ["split"]


@pytest.mark.parametrize("classes", [4, 9])
def test_all_development_rows_pooled_and_smoke_replay_are_deterministic(tmp_path, classes):
    specification = make_manifest(tmp_path, classes)
    frame = locked.load_locked_development(specification)
    assert len(frame) == 6 * classes and set(frame.split) == {"train", "val"}
    smoke = locked.load_locked_development(specification, smoke=True)
    assert len(smoke) == 4 * classes
    assert smoke.groupby(["split", "label_index"]).size().eq(2).all()
    sample = locked.replay_sample(frame.sample(frac=1, random_state=42))
    assert sample.image_id.tolist() == locked.replay_sample(frame).image_id.tolist()
    assert len(sample) == 2 * classes
    frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="development examples only"):
        locked.replay_sample(frame)


def test_manifest_bytes_row_count_and_class_mapping_cannot_drift(tmp_path):
    specification = make_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="row count"):
        locked.load_locked_development({**specification, "development_rows": 25})
    with pytest.raises(RuntimeError, match="class mapping"):
        locked.load_locked_development(
            {**specification, "class_names": specification["class_names"][::-1]}
        )
    path = Path(specification["development_manifest"])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(RuntimeError, match="manifest bytes changed"):
        locked.load_locked_development(specification)


def test_dino_scope_and_forward_match_historical_top_four_plus_all_norms():
    from transformers import Dinov2Config, Dinov2Model

    torch.set_num_threads(2)
    backbone = Dinov2Model(
        Dinov2Config(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=6,
            num_attention_heads=2,
            image_size=16,
            patch_size=4,
        )
    )
    model = locked.LockedDinoClassifier(backbone, 4)
    assert type(model).forward is PinnedDinoV2Classifier.forward
    for name, parameter in model.named_parameters():
        expected = (
            name.startswith("classifier.")
            or any(name.startswith(f"backbone.encoder.layer.{index}.") for index in range(2, 6))
            or (name.startswith("backbone.") and "norm" in name.lower())
        )
        assert parameter.requires_grad == expected, name
    model.train()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(2, 3, 16, 16)), torch.tensor([0, 3]))
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    evidence = locked.parameter_evidence(model, "dinov2_base")
    assert evidence["scope"] == "historical_top_four_blocks_and_all_backbone_norms"
    assert (
        evidence["total_parameters"]
        == evidence["trainable_parameters"] + evidence["frozen_parameters"]
    )


def test_dino_foundation_requires_exact_original_bytes_and_local_only(tmp_path, monkeypatch):
    import huggingface_hub

    for name in locked.DINO_FOUNDATION_HASHES:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        locked,
        "DINO_FOUNDATION_HASHES",
        {name: sha256_file(tmp_path / name) for name in locked.DINO_FOUNDATION_HASHES},
    )
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    path, evidence, _ = locked.final_foundation("dinov2_base")
    assert path == tmp_path and calls[0]["local_files_only"] is True
    assert evidence["files"]["model.safetensors"]["sha256"] == sha256_file(
        tmp_path / "model.safetensors"
    )
    (tmp_path / "model.safetensors").write_bytes(b"Changed original")
    with pytest.raises(RuntimeError, match="Original DINO foundation changed"):
        locked.final_foundation("dinov2_base")


def test_dino_missing_weights_never_silently_randomly_initialized(tmp_path, monkeypatch):
    from transformers import AutoModel

    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return object(), {"missing_keys": ["encoder.layer.0.weight"]}

    monkeypatch.setattr(AutoModel, "from_pretrained", load)
    with pytest.raises(RuntimeError, match="no hidden-weight substitutions"):
        locked.build_final_model("dinov2_base", 9, tmp_path)
    assert calls[0]["local_files_only"] and calls[0]["use_safetensors"]
    assert calls[0]["trust_remote_code"] is False


@pytest.mark.parametrize(
    "kind,mean,std",
    [
        ("dinov2_base", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ("siglip2_base", [0.5] * 3, [0.5] * 3),
        ("convnextv2_base", [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ],
)
def test_final_transform_preserves_model_specific_normalization(kind, mean, std):
    from torchvision import transforms

    for training_mode in (True, False):
        transform = locked.final_transform(
            kind, {"image_mean": mean, "image_std": std}, training=training_mode
        )
        assert list(transform.transforms[-1].mean) == mean
        assert list(transform.transforms[-1].std) == std
        assert not any(
            isinstance(
                item, (transforms.RandomCrop, transforms.RandomResizedCrop, transforms.CenterCrop)
            )
            for item in transform.transforms
        )
        assert transform(Image.new("RGB", (8, 16))).shape == (3, 224, 224)


def test_dataset_refuses_test_and_hash_drift_before_decoding(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"Changed image")
    frame = pd.DataFrame(
        [{"split": "train", "image_path": str(image), "image_sha256": "0" * 64, "label_index": 0}]
    )
    dataset = locked.FinalFitImages(frame, model_kind="dinov2_base", processor={}, training=False)
    with pytest.raises(RuntimeError, match="image bytes changed"):
        dataset[0]
    frame.loc[0, "split"] = "test"
    with pytest.raises(ValueError, match="never accept test rows"):
        locked.FinalFitImages(frame, model_kind="dinov2_base", processor={}, training=False)


def test_cli_exposes_no_epoch_selection_or_hyperparameter_override(tmp_path):
    base = [
        "--lock",
        str(tmp_path / "lock.json"),
        "--task",
        "polar9",
        "--model-kind",
        "dinov2_base",
        "--seed",
        "42",
        "--output-dir",
        str(tmp_path / "run"),
    ]
    args = training.parse_args(base + ["--smoke"])
    assert args.smoke and args.workers == 4
    for switch in ("--epochs", "--head-lr", "--checkpoint", "--patience"):
        with pytest.raises(SystemExit):
            training.parse_args(base + [switch, "1"])


def test_fixed_scheduler_horizon_is_not_compressed_to_refit_epochs():
    assert training.schedule_steps(65) == 40
    assert training.schedule_steps(64) == 20
    with pytest.raises(ValueError, match="20-epoch"):
        training.schedule_steps(65, 7)


def test_cuda_and_bfloat16_fail_closed_before_manifest_reads(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="refusing CPU"):
        training.run_training(SimpleNamespace())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    with pytest.raises(RuntimeError, match="precision substitution"):
        training.run_training(SimpleNamespace())


@pytest.mark.parametrize("smoke", [True, False])
def test_request_binds_lock_sources_pooled_development_and_fixed_epoch_budget(
    tmp_path, monkeypatch, smoke
):
    path, lock, _ = make_lock(tmp_path)
    specification = lock["tasks"]["polar4"]
    fit = specification["neural_fits"]["siglip2_base"]
    frame = locked.load_locked_development(specification, smoke=smoke)
    args = SimpleNamespace(
        lock=path, task="polar4", model_kind="siglip2_base", seed=42, workers=4, smoke=smoke
    )
    monkeypatch.setattr(training, "environment_evidence", lambda: {"unit_test": True})
    model = SimpleNamespace(parameter_evidence=lambda: {"scope": "fixture"})
    request = training.final_request(
        args,
        lock,
        specification,
        fit,
        frame,
        model,
        {},
        {"image_mean": [0.5] * 3, "image_std": [0.5] * 3},
        {},
    )
    assert request["epochs"] == (1 if smoke else 7)
    assert request["schedule_horizon_epochs"] == 20
    assert request["selection_lock_sha256"] == request["lock_sha256"] == sha256_file(path)
    assert request["train_rows"] == (16 if smoke else 24)
    assert request["role"] == ("engineering_smoke" if smoke else locked.FINAL_STAGE)
    assert request["test_rows_read"] == 0 and request["test_labels_read"] is False
    assert request["validation_selection"] is False and request["early_stopping"] is False
    assert "experiments/train_polar_locked.py" in request["implementation"]


def test_resume_binds_checkpoint_history_request_and_lock(tmp_path):
    request = {"lock_sha256": "a" * 64}
    assert training.resume_integrity(tmp_path, request) is None
    (tmp_path / "last.pt").write_bytes(b"checkpoint")
    with pytest.raises(RuntimeError, match="lacks integrity"):
        training.resume_integrity(tmp_path, request)
    (tmp_path / "history.csv").write_text("epoch,train_loss\n1,0.5\n", encoding="utf-8")
    sidecar = {
        "sha256": sha256_file(tmp_path / "last.pt"),
        "request_sha256": canonical_hash(request),
        "lock_sha256": request["lock_sha256"],
        "history_sha256": sha256_file(tmp_path / "history.csv"),
    }
    (tmp_path / "last_checkpoint.json").write_text(json.dumps(sidecar), encoding="utf-8")
    assert training.resume_integrity(tmp_path, request) == sidecar
    (tmp_path / "history.csv").write_text("epoch,train_loss\n1,0.1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        training.resume_integrity(tmp_path, request)


def test_replay_requires_exact_identity_labels_decisions_and_probability_parity(tmp_path):
    saved = tmp_path / "replay.npz"
    ids, labels = np.array(["a", "b"]), np.array([0, 1])
    probabilities = np.array([[0.7, 0.3], [0.2, 0.8]])
    np.savez(saved, image_ids=ids, labels=labels, probabilities=probabilities)
    audit = locked.validate_replay(ids, labels, probabilities.copy(), saved)
    assert audit["status"] == "PASS" and audit["test_rows_read"] == 0
    assert audit["maximum_probability_difference"] == 0 and audit["predictions_identical"]
    for altered_ids, altered_labels, altered_probabilities in [
        (ids[::-1], labels, probabilities),
        (ids, labels[::-1], probabilities),
        (ids, labels, probabilities[:, ::-1]),
        (ids, labels, probabilities + 2e-5),
        (ids, labels, probabilities * np.nan),
    ]:
        with pytest.raises(RuntimeError, match="replay"):
            locked.validate_replay(altered_ids, altered_labels, altered_probabilities, saved)


def complete_fixture(tmp_path, monkeypatch):
    request = {
        "role": locked.FINAL_STAGE,
        "stage": locked.FINAL_STAGE,
        "smoke": False,
        "lock_sha256": "a" * 64,
        "selection_lock_sha256": "a" * 64,
        "test_rows_read": 0,
        "test_labels_read": False,
        "selection": "none_fixed_locked_epoch_budget",
        "implementation": {},
        "epochs": 7,
        "model_kind": "dinov2_base",
        "classes": 4,
        "task": "polar4",
        "seed": 42,
        "train_rows": 24,
        "schedule_horizon_epochs": 20,
        "foundation": {},
        "foundation_loading": {},
        "preprocessing": locked.final_preprocessing("dinov2_base", {}),
    }
    model = torch.nn.Linear(2, 4)
    request["parameters"] = locked.parameter_evidence(model, "dinov2_base")
    monkeypatch.setattr(locked, "verify_implementation", lambda *_: None)
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    checkpoint = {
        "model": model.state_dict(),
        "request_sha256": canonical_hash(request),
        "lock_sha256": request["lock_sha256"],
        "epoch": 7,
        "model_kind": "dinov2_base",
        "task": "polar4",
        "seed": 42,
    }
    torch.save(checkpoint, tmp_path / "final.pt")
    checkpoint_sha = sha256_file(tmp_path / "final.pt")
    sidecar = {
        "sha256": checkpoint_sha,
        "request_sha256": canonical_hash(request),
        "lock_sha256": request["lock_sha256"],
    }
    (tmp_path / "final_checkpoint.json").write_text(json.dumps(sidecar), encoding="utf-8")
    replay = {
        "status": "PASS",
        "predictions_identical": True,
        "test_rows_read": 0,
        "checkpoint_sha256": checkpoint_sha,
    }
    (tmp_path / "replay_audit.json").write_text(json.dumps(replay), encoding="utf-8")
    for name in ("dev_replay_predictions.npz", "history.csv", "last.pt", "last_checkpoint.json"):
        (tmp_path / name).write_bytes(b"unit-test-only integrity fixture")
    artifacts = {path.name: sha256_file(path) for path in tmp_path.iterdir() if path.is_file()}
    summary = {
        **{
            key: request[key]
            for key in (
                "role",
                "stage",
                "lock_sha256",
                "selection_lock_sha256",
                "model_kind",
                "task",
                "seed",
                "train_rows",
                "test_rows_read",
                "test_labels_read",
                "selection",
                "schedule_horizon_epochs",
            )
        },
        "status": "COMPLETE",
        "request_sha256": canonical_hash(request),
        "epochs_completed": 7,
        "artifacts": artifacts,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    return request, model


def test_inference_loader_reconstructs_exact_final_checkpoint_on_cpu_fixture(tmp_path, monkeypatch):
    request, original = complete_fixture(tmp_path, monkeypatch)
    locked.validate_final_run(tmp_path, expected_lock_sha256=request["lock_sha256"])
    monkeypatch.setattr(locked, "final_foundation", lambda _: (tmp_path, {}, {}))
    monkeypatch.setattr(locked, "build_final_model", lambda *_: (torch.nn.Linear(2, 4), {}))
    restored, transform, restored_request = locked.load_final_model(
        tmp_path, expected_lock_sha256=request["lock_sha256"], device="cpu"
    )
    assert restored_request == request and not restored.training and callable(transform)
    assert all(
        torch.equal(value, restored.state_dict()[name])
        for name, value in original.state_dict().items()
    )
    (tmp_path / "final.pt").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="artifact drift"):
        locked.load_final_model(tmp_path, expected_lock_sha256=request["lock_sha256"], device="cpu")


def test_smoke_or_wrong_lock_never_qualifies_for_final_inference(tmp_path, monkeypatch):
    request, _ = complete_fixture(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="production final refit"):
        locked.validate_final_run(tmp_path, expected_lock_sha256="b" * 64)
    request["smoke"], request["role"] = True, "engineering_smoke"
    (tmp_path / "request.json").write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(RuntimeError, match="production final refit"):
        locked.load_final_model(tmp_path, expected_lock_sha256=request["lock_sha256"], device="cpu")
