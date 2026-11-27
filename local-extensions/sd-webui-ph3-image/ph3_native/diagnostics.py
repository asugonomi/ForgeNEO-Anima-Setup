# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
import importlib.metadata
import inspect
import json
from pathlib import Path
import platform
import sys
from .contracts import ROOT, GIB, PH3Error, UPSTREAM_COMMIT
from .headers import inspect_component
from .provenance import verify_vendor, native_module


def runtime_checks() -> dict:
    try:
        import torch
        import comfy_kitchen as ck
        from backend import memory_management as mm
        from backend.operations import using_forge_operations
        from backend.patcher.base import ModelPatcher
        from backend.quant_ops import QUANT_ALGOS
        from backend.nn.llm.llama import Qwen3VL_4BConfig, Llama2_, precompute_freqs_cis
        from modules.call_queue import queue_lock
    except ImportError as e:
        raise PH3Error(f"必要なForge Neo APIを読み込めません: {e}\nA1111本家・旧Forgeは対象外です。") from e
    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise PH3Error("この実験版にはNVIDIA CUDA GPUが必要です。")
    device = mm.get_torch_device()
    if device.type != "cuda":
        raise PH3Error("Forgeの推論デバイスがCUDAではありません。")
    if torch.cuda.get_device_capability(device)[0] < 8:
        raise PH3Error("この実験版はNVIDIA Ampere以降を対象にしています。")
    needed = {"memory_required", "minimum_memory_required", "force_full_load"}
    if not needed.issubset(inspect.signature(mm.load_models_gpu).parameters):
        raise PH3Error("Forgeのload_models_gpu APIが想定と異なります。")
    for name in ("unload_all_models", "soft_empty_cache", "get_free_memory", "cleanup_models"):
        if not callable(getattr(mm, name, None)):
            raise PH3Error(f"Forgeのメモリ管理APIがありません: {name}")
    if not hasattr(ck, "rms_rope_split_half_"):
        raise PH3Error("comfy-kitchenに必要なRMS/RoPEカーネルがありません。自動更新は行いません。")
    state = getattr(getattr(mm, "vram_state", None), "name", "unknown")
    if state not in {"NORMAL_VRAM", "LOW_VRAM", "NO_VRAM"}:
        raise PH3Error(f"現在のForge VRAM設定は {state} です。\n--highvram / --gpu-only等を外し、通常のオフロード可能な設定で起動してください。")
    tokenizer = ROOT / "tokenizer"
    if not (tokenizer / "tokenizer_config.json").is_file():
        raise PH3Error("トークナイザーが未設定です。setup_native.batを実行してください。")
    from .tokenizer import load_tokenizer
    # Check the real host transformers/assets boundary, not just file existence.
    tokenizer_probe = load_tokenizer(tokenizer).encode("A red teapot.", add_special_tokens=False)
    if not tokenizer_probe:
        raise PH3Error("トークナイザーの実エンコード検査が空の結果を返しました。")
    manifest = verify_vendor()
    for module_name, class_name in (("transformer", "MiniMaxH3Model"), ("video_vae", "MiniMaxH3VideoVAE")):
        try:
            implementation = native_module(module_name)
            if not callable(getattr(implementation, class_name, None)):
                raise PH3Error(f"ネイティブ実装に必要なクラスがありません: {module_name}.{class_name}")
        except ImportError as e:
            raise PH3Error(f"ネイティブ実装の依存関係を読み込めません: {e}。自動更新は行いません。") from e
    free, total = torch.cuda.mem_get_info(device)
    data = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "cuda_build": torch.version.cuda, "gpu": torch.cuda.get_device_name(device),
            "vram_total_gib": total / GIB, "vram_free_gib": free / GIB, "forge_vram_mode": state,
            "native_commit": manifest["commit"], "kitchen_backends": ck.list_backends(),
            "quant_formats_supported": sorted(QUANT_ALGOS), "tokenizer_encode_ok": True}
    for package in ("comfy-kitchen", "transformers", "safetensors", "gradio"):
        try:
            data[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            data[package] = "unknown"
    try:
        import psutil
        mem = psutil.virtual_memory()
        data.update(ram_total_gib=mem.total / GIB, ram_available_gib=mem.available / GIB)
    except ImportError:
        data["ram_available_gib"] = None
    return data


def diagnose(paths: dict[str, str]) -> dict:
    report = {"status": "diagnostic_only", "gpu_inference_tested": False,
              "note": "この診断はAPIとヘッダーの確認です。生成成功・画質・16GB動作を証明しません。",
              "errors": [], "warnings": [], "components": []}
    try:
        report["runtime"] = runtime_checks()
    except Exception as e:
        report["errors"].append(str(e))
    for kind, path in paths.items():
        if not str(path).strip():
            report["warnings"].append(f"{kind}: ファイルが未指定です。")
            continue
        try:
            report["components"].append(inspect_component(path, kind))
        except Exception as e:
            report["errors"].append(f"{kind}: {e}")
    report["api_and_headers_ok"] = not report["errors"] and len(report["components"]) == 3
    return report


def render_report(report: dict) -> str:
    return json.dumps(report, ensure_ascii=False, indent=2, default=str)
