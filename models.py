# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
from mmap import MAP_PRIVATE
import torch
import torch.nn as nn
from functools import partial
import pathlib

from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

import collaborate_attention


@register_model
def deit_tiny_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_tiny_colab_patch16_224(pretrained=False, all_key_dim=None, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])

    model.cuda()
    collaborate_attention.swap(model, all_key_dim)
    model.cpu()

    return model


@register_model
def deit_small_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu",
            check_hash=True,
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def deit_base_patch16_224_collab384(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    collaborate_attention.swap(model, compressed_key_dim=384, reparametrize=False)
    return model


@register_model
def deit_base_patch16_224_collab256(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    collaborate_attention.swap(model, compressed_key_dim=256, reparametrize=False)
    return model

@register_model
def deit_base3_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    assert not pretrained
    return model

# ========== REDUCED KEY DIMENSION CONCATENATE ATTENTION MODELS ========== #

@register_model
def deit_base3_patch16_224_key384(pretrained=False, **kwargs):
    import timm.models.vision_transformer
    from collaborate_attention import FlexibleKeyDimensionAttention
    timm.models.vision_transformer.Attention = partial(FlexibleKeyDimensionAttention, all_key_dim=384)

    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    assert not pretrained
    return model


@register_model
def deit_base3_patch16_224_key192(pretrained=False, **kwargs):
    import timm.models.vision_transformer
    from collaborate_attention import FlexibleKeyDimensionAttention
    timm.models.vision_transformer.Attention = partial(FlexibleKeyDimensionAttention, all_key_dim=192)

    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    assert not pretrained
    return model


@register_model
def deit_base3_patch16_224_key96(pretrained=False, **kwargs):
    import timm.models.vision_transformer
    from collaborate_attention import FlexibleKeyDimensionAttention
    timm.models.vision_transformer.Attention = partial(FlexibleKeyDimensionAttention, all_key_dim=96)

    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    model.default_cfg = _cfg()
    assert not pretrained
    return model


# ========== COLLABORATIVE ATTENTION MODELS ========== #

# ========== BASE 3 LAYERS ========== #

@register_model
def deit_base3_patch16_224_collab384(pretrained=False, models_directory=None, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    collaborate_attention.swap(model, compressed_key_dim=384, reparametrize=False)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint_path = pathlib.Path(models_directory) / "deit_base3_patch16_224_collab384.pth"
        print(f"Load model from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base3_patch16_224_collab192(pretrained=False, models_directory=None, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    collaborate_attention.swap(model, compressed_key_dim=192, reparametrize=False)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint_path = pathlib.Path(models_directory) / "deit_base3_patch16_224_collab192.pth"
        print(f"Load model from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base3_patch16_224_collab96(pretrained=False, models_directory=None, **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=768,
        depth=3,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    collaborate_attention.swap(model, compressed_key_dim=96, reparametrize=False)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint_path = pathlib.Path(models_directory) / "deit_base3_patch16_224_collab96.pth"
        print(f"Load model from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


# ========== BASE ========== #

@register_model
def deit_base_patch16_224_collab64(pretrained=False, models_directory="./models", **kwargs):
    model = deit_base_patch16_224(pretrained=False)
    collaborate_attention.swap(model, compressed_key_dim=64, reparametrize=False)
    if pretrained:
        checkpoint_path = pathlib.Path(models_directory) / "deit_base_patch16_224_collab64.pth"
        print(f"Load model from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_collab128(pretrained=False, models_directory="./models", **kwargs):
    model = deit_base_patch16_224(pretrained=False)
    collaborate_attention.swap(model, compressed_key_dim=128, reparametrize=False)
    if pretrained:
        checkpoint_path = pathlib.Path(models_directory) / "deit_base_patch16_224_collab128.pth"
        print(f"Load model from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_collab256(pretrained=False, models_directory="./models", **kwargs):
    model = deit_base_patch16_224(pretrained=False)
    collaborate_attention.swap(model, compressed_key_dim=256, reparametrize=False)
    if pretrained:
        checkpoint_path = pathlib.Path(models_directory) / "deit_base_patch16_224_collab256.pth"
        print(f"Load model from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_collab384(pretrained=False, models_directory="./models", **kwargs):
    model = deit_base_patch16_224(pretrained=False)
    collaborate_attention.swap(model, compressed_key_dim=384, reparametrize=False)
    if pretrained:
        checkpoint_path = pathlib.Path(models_directory) / "deit_base_patch16_224_collab384.pth"
        print(f"Load model from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_collab512(pretrained=False, models_directory="./models", **kwargs):
    model = deit_base_patch16_224(pretrained=False)
    collaborate_attention.swap(model, compressed_key_dim=512, reparametrize=False)
    if pretrained:
        checkpoint_path = pathlib.Path(models_directory) / "deit_base_patch16_224_collab512.pth"
        print(f"Load model from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def deit_base_patch16_224_collab768(pretrained=False, models_directory="./models", **kwargs):
    model = deit_base_patch16_224(pretrained=False)
    collaborate_attention.swap(model, compressed_key_dim=768, reparametrize=False)
    if pretrained:
        checkpoint_path = pathlib.Path(models_directory) / "deit_base_patch16_224_collab768.pth"
        print(f"Load model from '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

