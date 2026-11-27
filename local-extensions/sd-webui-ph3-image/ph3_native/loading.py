# SPDX-License-Identifier: AGPL-3.0-or-later
"""Strict P-H3 loaders using Forge operations and its existing ModelPatcher."""
from __future__ import annotations
import gc
from pathlib import Path
import torch
from .contracts import PH3Error
from .headers import normalized_key, count_layers, inspect_component
from .models import make_text_encoder_class, make_vae_class
from .provenance import native_module


def _strict_load(model, state_dict, name):
    # assign=True keeps mmap-backed storage and avoids a second complete weight copy.
    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=True)
    unexpected = [k for k in unexpected if "comfy_attention" not in k]
    if missing or unexpected:
        raise PH3Error(f"{name}の重みが実装と一致しません。\n不足: {missing[:20]}\n余剰: {unexpected[:20]}")
    if any(p.is_meta for p in model.parameters()):
        raise PH3Error(f"{name}に未読込のmetaパラメータが残っています。")
    model.eval()
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.requires_grad_(False)


def _restore_plain_fp32(model, values):
    for key, value in values.items():
        parent, _, leaf = key.rpartition(".")
        owner = model.get_submodule(parent) if parent else model
        value = value.detach().to(torch.float32)
        if leaf in owner._parameters:
            owner._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
        elif leaf in owner._buffers:
            owner._buffers[leaf] = value
        else:
            raise PH3Error(f"float32部品が見つかりません: {key}")


def load_component(path: str, kind: str, log=print):
    from backend import memory_management as mm
    from backend.operations import using_forge_operations
    from backend.state_dict import convert_quantization, detect_quantization
    from backend.utils import load_torch_file
    from backend.patcher.base import ModelPatcher
    from backend.quant_ops import QUANT_ALGOS
    from transformers.modeling_utils import no_init_weights

    base_kind = {"text_encoder_edit": "text_encoder", "vae_encoder": "vae"}.get(kind, kind)
    info = inspect_component(path, base_kind)
    for fmt in info["quant_formats"]:
        if fmt not in QUANT_ALGOS:
            raise PH3Error(f"現在のForge Neoは量子化形式 {fmt!r} に未対応です。既存環境は自動更新しません。")
    log(f"{kind}: {Path(path).name} ({info['bytes'] / 2**30:.2f} GiB)")
    sd, metadata = load_torch_file(path, safe_load=True, device=mm.cpu, return_metadata=True)
    sd, metadata = convert_quantization(sd, metadata)
    if base_kind == "text_encoder":
        sd = {normalized_key(k): v for k, v in sd.items()}
        # t2i does not consume the vision tower, an LM head, or editing inputs.
        excluded = ("lm_head.",) if kind == "text_encoder_edit" else ("visual.", "lm_head.")
        sd = {k: v for k, v in sd.items() if not k.startswith(excluded)}
        cls = make_text_encoder_class(with_vision=(kind == "text_encoder_edit"))
        args = {}
        fp32 = {k: v for k, v in sd.items() if k.startswith("text_connector.")}
        compute_dtype = torch.bfloat16
    elif base_kind == "vae":
        n = native_module("video_vae")
        if kind == "vae_encoder":
            from .editing import make_encoder_class
            cls = make_encoder_class(n)
            args = {}
            sd = {k: v for k, v in sd.items() if not k.startswith(("decoder.", "post_quant_conv."))}
        else:
            cls = make_vae_class(n)
            args = {"num_layers": count_layers(sd, "decoder.transformer_blocks.")}
            sd = {k: v for k, v in sd.items() if not k.startswith(("encoder.", "quant_conv."))}
        sd.setdefault("latents_mean", torch.tensor(n.LATENTS_MEAN))
        sd.setdefault("latents_std", torch.tensor(n.LATENTS_STD))
        fp32 = {}
        compute_dtype = torch.float32  # input/decoder compute uses fp16, not bf16.
    elif kind == "dit":
        detection = native_module("detect")
        cls = native_module("transformer").MiniMaxH3Model
        args = detection.detect(sd, metadata=metadata)
        if not args:
            raise PH3Error("MiniMax-H3形式の画像生成モデルを検出できませんでした。")
        args.pop("image_model", None)
        # A quantized matrix can have packed columns. The P-H3 connector fixes this dimension.
        args["text_dim"] = 5120
        args.update(sigma_shift_video=1.0, sigma_shift_audio=1.0)
        fp32 = native_module("islands").fp32_islands(sd, curve="adaln_t_table" in sd)
        compute_dtype = torch.bfloat16
        args.update(dtype=compute_dtype, device=mm.cpu)
    else:
        raise PH3Error(f"不明な部品: {kind}")

    quant = detect_quantization(sd, is_unet=(kind == "dit"))
    if quant:
        quant = dict(quant)
        quant.pop("layout", None)
        if base_kind == "vae":
            quant["TE"] = False
    fb = native_module("filebacked")
    storages = fb.file_storages(sd)
    with no_init_weights(), using_forge_operations(device=mm.cpu, dtype=compute_dtype,
                                                   manual_cast_enabled=True, extra_dtype=quant):
        model = cls(**args)
    try:
        _strict_load(model, sd, kind)
        _restore_plain_fp32(model, fp32)
        # This affects only these newly created modules; Forge global functions are not patched.
        backed, total = fb.keep_file_backed(model, storages)
        info["file_backed_bytes"] = backed
        info["parameter_payload_bytes"] = total
        log(f"{kind}: file-backed {backed / 2**30:.2f} / {total / 2**30:.2f} GiB")
        patcher = ModelPatcher(model, load_device=mm.get_torch_device(), offload_device=mm.cpu)
    except BaseException:
        del model
        raise
    finally:
        del sd, fp32
        gc.collect()
    return patcher, info
