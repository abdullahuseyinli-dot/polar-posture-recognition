"""Two fixed, person-preserving partial-adaptation controls for POLAR.

Foundation weights and preprocessing statistics are hash-bound. The forward path
is the official backbone forward: frozen prefixes automatically build no autograd
graph because both their inputs and parameters have requires_grad=False.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

from hac.augmentations import (
    SquarePad,
    build_aspect_preserving_eval_transform,
    build_person_train_transform,
)
from hac.polar import sha256_file
from hac.polar_benchmark_features import MODEL_SPECS

BOUNDED_MODELS = ("siglip2_base", "convnextv2_base")
FOUNDATION_HASHES = {
    "siglip2_base": {
        "config.json": "fe8b5fe6d5734360678fd71c11c21e1ea3364bd8598d34295d9206335973ffd7",
        "model.safetensors": "612923381c76ec5a9bed335d1c48827e3f2e506ac31b044b63b2031fadee6a0b",
        "preprocessor_config.json": "9b36b57ebaf20f09bf4c22100ccc21877ea6bfe5aead0c00c59f8af8ccefacfc",
    },
    "convnextv2_base": {
        "config.json": "dac818f34972086f8171c444adac7752e13466b201eee2e248768597aa114732",
        "model.safetensors": "8937dc20b1d7a63ea628752315a89b9a4fc0f58bf78b3299041e99c0ad3f93bb",
        "preprocessor_config.json": "9ef34d18dbf7909143a91f05ad23457180fcfed05034bf94843f300e32a9b968",
    },
}


def validate_loading_info(model_kind: str, loading: dict) -> None:
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(
            "Foundation backbone has missing/mismatched weights; random substitution prohibited"
        )
    unexpected = loading.get("unexpected_keys", [])
    if model_kind == "siglip2_base":
        allowed = all(
            name.startswith("text_model.") or name in {"logit_scale", "logit_bias"}
            for name in unexpected
        )
    elif model_kind == "convnextv2_base":
        allowed = set(unexpected).issubset({"classifier.weight", "classifier.bias"})
    else:
        raise ValueError("Unknown bounded model")
    if not allowed:
        raise RuntimeError("Unexpected foundation vision weights were not loaded")


def foundation_evidence(model_kind: str) -> tuple[Path, dict, dict]:
    from huggingface_hub import snapshot_download

    if model_kind not in BOUNDED_MODELS:
        raise ValueError("Only the two prespecified bounded models are supported")
    specification = MODEL_SPECS[model_kind]
    snapshot = Path(
        snapshot_download(
            repo_id=specification["model_id"],
            revision=specification["revision"],
            allow_patterns=list(FOUNDATION_HASHES[model_kind]),
            local_files_only=True,
        )
    )
    files = {}
    for filename, expected in FOUNDATION_HASHES[model_kind].items():
        path = snapshot / filename
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Original pinned foundation bytes differ: {model_kind}/{filename}")
        files[filename] = {"sha256": expected, "bytes": path.stat().st_size}
    processor = json.loads((snapshot / "preprocessor_config.json").read_text(encoding="utf-8"))
    mean, std = processor["image_mean"], processor["image_std"]
    if len(mean) != 3 or len(std) != 3 or any(not 0 < float(value) <= 1 for value in std):
        raise ValueError("Invalid official normalization statistics")
    if model_kind == "siglip2_base" and (mean != [0.5] * 3 or std != [0.5] * 3):
        raise RuntimeError("SigLIP normalization differs from its pinned original processor")
    return (
        snapshot,
        {
            "model_id": specification["model_id"],
            "revision": specification["revision"],
            "files": files,
        },
        processor,
    )


def build_person_transform(processor: dict, *, training: bool):
    """Keep shared person_safe_mild geometry, replace only normalization and fills."""
    mean, std = tuple(processor["image_mean"]), tuple(processor["image_std"])
    if len(mean) != 3 or len(std) != 3 or any(not 0 < float(value) <= 1 for value in std):
        raise ValueError("Three finite positive channel standard deviations are required")
    fill = tuple(round(255 * float(value)) for value in mean)
    if not training:
        return build_aspect_preserving_eval_transform(224, mean=mean, std=std, fill=fill)
    transform = build_person_train_transform("person_safe_mild", 224)
    for index, operation in enumerate(transform.transforms):
        if isinstance(operation, SquarePad):
            transform.transforms[index] = SquarePad(fill=fill)
        elif isinstance(operation, transforms.Normalize):
            transform.transforms[index] = transforms.Normalize(mean, std)
        elif isinstance(operation, transforms.RandomApply):
            for nested in operation.transforms:
                if isinstance(nested, transforms.RandomAffine):
                    nested.fill = fill
    return transform


def preprocessing_evidence(processor: dict) -> dict:
    return {
        "view": "person_context_25",
        "image_size": 224,
        "geometry": "square_pad_bicubic_resize_person_safe_mild",
        "random_resized_crop": False,
        "normalization_source": "hash_verified_pinned_official_preprocessor_config",
        "image_mean": processor["image_mean"],
        "image_std": processor["image_std"],
        "padding_and_affine_fill": [round(255 * float(value)) for value in processor["image_mean"]],
        "official_processor": processor,
        "official_resize_reused": False,
        "reason": "Shared person-preserving geometry; only channel statistics follow each model's official processor",
    }


class BoundedClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, model_kind: str, num_classes: int) -> None:
        super().__init__()
        if model_kind not in BOUNDED_MODELS or num_classes not in {4, 9}:
            raise ValueError(
                "Bounded adaptation supports only its two models and four/nine classes"
            )
        self.backbone, self.model_kind = backbone, model_kind
        backbone.requires_grad_(False)
        if model_kind == "siglip2_base":
            vision = backbone.vision_model
            layers = vision.encoder.layers
            if len(layers) < 4 or not getattr(vision, "use_head", True):
                raise ValueError(
                    "SigLIP requires at least four encoder layers and the original pooler"
                )
            self._adapted_modules = tuple(layers[-4:]) + (vision.post_layernorm, vision.head)
            self.scope = "last_four_vision_encoder_blocks_plus_post_layernorm_plus_attention_pooler"
            dimensions = int(backbone.config.hidden_size)
        else:
            if len(backbone.encoder.stages) != 4:
                raise ValueError("ConvNeXtV2 requires the four-stage foundation architecture")
            self._adapted_modules = (backbone.encoder.stages[-1], backbone.layernorm)
            self.scope = "last_encoder_stage_plus_final_layernorm"
            dimensions = int(backbone.config.hidden_sizes[-1])
        for module in self._adapted_modules:
            module.requires_grad_(True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(dimensions, num_classes)
        self.train()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        for module in self._adapted_modules:
            module.train(mode)
        return self

    def forward(self, pixels: torch.Tensor, *, return_features: bool = False):
        # No input-gradient objective is part of this fixed adaptation experiment.
        features = self.backbone(pixel_values=pixels.detach()).pooler_output
        if features is None or features.ndim != 2:
            raise RuntimeError("Official foundation returned no pooled vision representation")
        logits = self.classifier(self.dropout(features))
        return (logits, features) if return_features else logits

    def parameter_evidence(self) -> dict:
        trainable = {
            name: parameter.numel()
            for name, parameter in self.named_parameters()
            if parameter.requires_grad
        }
        frozen = {
            name: parameter.numel()
            for name, parameter in self.named_parameters()
            if not parameter.requires_grad
        }
        return {
            "scope": self.scope,
            "trainable_parameters": sum(trainable.values()),
            "frozen_parameters": sum(frozen.values()),
            "total_parameters": sum(trainable.values()) + sum(frozen.values()),
            "trainable_names_and_counts": trainable,
            "frozen_names_and_counts": frozen,
            "frozen_modules_eval": True,
            "frozen_prefix_autograd": "pruned_by_frozen_parameters_and_detached_input",
        }


def build_bounded_model(
    model_kind: str, num_classes: int, snapshot: Path
) -> tuple[BoundedClassifier, dict]:
    from transformers import AutoModel, SiglipVisionModel

    if model_kind not in BOUNDED_MODELS:
        raise ValueError("Unknown bounded model")
    model_class = SiglipVisionModel if model_kind == "siglip2_base" else AutoModel
    backbone, loading = model_class.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    validate_loading_info(model_kind, loading)
    model = BoundedClassifier(backbone, model_kind, num_classes)
    return model, {
        "unexpected_nonvision_keys": sorted(loading.get("unexpected_keys", [])),
        "missing_keys": [],
        "mismatched_keys": [],
    }
