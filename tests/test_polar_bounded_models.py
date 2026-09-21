"""Tiny real architectures verify scope, preprocessing and gradient isolation."""

from __future__ import annotations

import json

import pytest
import torch
from PIL import Image
from torchvision import transforms

from hac.polar import sha256_file
from hac.polar_bounded_models import (
    BoundedClassifier,
    build_bounded_model,
    build_person_transform,
    foundation_evidence,
    validate_loading_info,
)
from hac.polar_training import optimizer_parameter_groups


def tiny_model(kind):
    if kind == "siglip2_base":
        from transformers import SiglipVisionConfig, SiglipVisionModel

        backbone = SiglipVisionModel(
            SiglipVisionConfig(
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=6,
                num_attention_heads=2,
                image_size=16,
                patch_size=4,
            )
        )
        pixels = torch.randn(2, 3, 16, 16)
    else:
        from transformers import ConvNextV2Config, ConvNextV2Model

        backbone = ConvNextV2Model(
            ConvNextV2Config(
                num_stages=4, hidden_sizes=[4, 8, 16, 32], depths=[1, 1, 1, 1], drop_path_rate=0.2
            )
        )
        pixels = torch.randn(2, 3, 32, 32)
    return BoundedClassifier(backbone, kind, 4), pixels


@pytest.mark.parametrize("kind", ["siglip2_base", "convnextv2_base"])
def test_exact_scope_frozen_eval_and_finite_gradients(kind):
    torch.set_num_threads(2)
    model, pixels = tiny_model(kind)
    evidence = model.parameter_evidence()
    assert (
        evidence["total_parameters"]
        == evidence["trainable_parameters"] + evidence["frozen_parameters"]
    )
    trainable = set(evidence["trainable_names_and_counts"])
    if kind == "siglip2_base":
        expected = tuple(
            f"backbone.vision_model.encoder.layers.{index}." for index in range(2, 6)
        ) + ("backbone.vision_model.post_layernorm.", "backbone.vision_model.head.", "classifier.")
        assert not model.backbone.vision_model.embeddings.training
        assert not model.backbone.vision_model.encoder.layers[0].training
        assert all(block.training for block in model.backbone.vision_model.encoder.layers[-4:])
        frozen_prefix = model.backbone.vision_model.encoder.layers[0]
    else:
        expected = ("backbone.encoder.stages.3.", "backbone.layernorm.", "classifier.")
        assert all(not stage.training for stage in model.backbone.encoder.stages[:-1])
        assert model.backbone.encoder.stages[-1].training
        frozen_prefix = model.backbone.encoder.stages[0]
    assert trainable == {name for name, _ in model.named_parameters() if name.startswith(expected)}
    observed = []
    handle = frozen_prefix.register_forward_hook(
        lambda _module, _args, result: observed.append(result.requires_grad)
    )
    loss = torch.nn.functional.cross_entropy(model(pixels.requires_grad_()), torch.tensor([0, 3]))
    loss.backward()
    handle.remove()
    assert observed == [False]
    assert pixels.grad is None
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
        else:
            assert parameter.grad is None, name
    model.eval()
    assert not any(module.training for module in model.modules())
    with torch.no_grad():
        _, features = model(pixels, return_features=True)
        expected_features = model.backbone(pixel_values=pixels).pooler_output
    assert torch.equal(features, expected_features)


def test_pretrained_pooler_uses_backbone_lr_not_new_classifier_lr():
    model, _ = tiny_model("siglip2_base")
    groups = optimizer_parameter_groups(model, head_lr=0.001, backbone_lr=5e-6, weight_decay=1e-4)
    rates = {id(parameter): group["lr"] for group in groups for parameter in group["params"]}
    assert all(
        rates[id(parameter)] == 5e-6 for parameter in model.backbone.vision_model.head.parameters()
    )
    assert all(rates[id(parameter)] == 0.001 for parameter in model.classifier.parameters())
    assert len(rates) == sum(parameter.requires_grad for parameter in model.parameters())


@pytest.mark.parametrize("training", [False, True])
def test_siglip_normalization_and_no_random_body_crop(training):
    processor = {"image_mean": [0.5] * 3, "image_std": [0.5] * 3}
    transform = build_person_transform(processor, training=training)
    operations = transform.transforms
    assert not any(
        isinstance(
            item, (transforms.RandomResizedCrop, transforms.RandomCrop, transforms.CenterCrop)
        )
        for item in operations
    )
    assert operations[0].fill == (128, 128, 128)
    assert tuple(operations[-1].mean) == (0.5, 0.5, 0.5)
    assert tuple(operations[-1].std) == (0.5, 0.5, 0.5)
    if not training:
        pixels = transform(Image.new("RGB", (8, 16), (128, 128, 128)))
        assert pixels.shape == (3, 224, 224)
        expected = ((torch.tensor(128, dtype=torch.float32) / 255 - 0.5) / 0.5).item()
        assert torch.equal(pixels, torch.full_like(pixels, expected))


@pytest.mark.parametrize(
    "kind,loading",
    [
        ("siglip2_base", {"missing_keys": ["vision_model.head.probe"]}),
        ("siglip2_base", {"mismatched_keys": ["vision_model.encoder.layers.0"]}),
        ("convnextv2_base", {"error_msgs": ["load failed"]}),
        ("siglip2_base", {"unexpected_keys": ["vision_model.fake.weight"]}),
        ("convnextv2_base", {"unexpected_keys": ["encoder.stages.0.fake"]}),
    ],
)
def test_missing_or_mismatched_hidden_weights_rejected(kind, loading):
    with pytest.raises(RuntimeError):
        validate_loading_info(kind, loading)


def test_unused_nonvision_weights_are_explicitly_permitted():
    validate_loading_info(
        "siglip2_base",
        {"unexpected_keys": ["text_model.embeddings.weight", "logit_scale", "logit_bias"]},
    )
    validate_loading_info(
        "convnextv2_base", {"unexpected_keys": ["classifier.weight", "classifier.bias"]}
    )


def test_foundation_hashes_checked_before_loading(tmp_path, monkeypatch):
    import huggingface_hub

    import hac.polar_bounded_models as bounded

    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"fake-test-only")
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_mean": [0.5] * 3, "image_std": [0.5] * 3}), encoding="utf-8"
    )
    expected = {path.name: sha256_file(path) for path in tmp_path.iterdir()}
    monkeypatch.setitem(bounded.FOUNDATION_HASHES, "siglip2_base", expected)
    calls = []

    def snapshot(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    path, evidence, _ = foundation_evidence("siglip2_base")
    assert path == tmp_path
    assert evidence["files"]["model.safetensors"]["sha256"] == expected["model.safetensors"]
    assert calls[0]["local_files_only"] is True
    (tmp_path / "model.safetensors").write_bytes(b"different")
    with pytest.raises(RuntimeError, match="foundation bytes differ"):
        foundation_evidence("siglip2_base")


def test_builder_passes_safe_loading_flags_and_rejects_random_missing_backbone(
    tmp_path, monkeypatch
):
    from transformers import SiglipVisionModel

    model, _ = tiny_model("siglip2_base")
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return model.backbone, {"missing_keys": ["vision_model.encoder.layers.0.weight"]}

    monkeypatch.setattr(SiglipVisionModel, "from_pretrained", load)
    with pytest.raises(RuntimeError, match="random substitution"):
        build_bounded_model("siglip2_base", 4, tmp_path)
    assert calls[0]["local_files_only"] and calls[0]["use_safetensors"]
    assert calls[0]["trust_remote_code"] is False
