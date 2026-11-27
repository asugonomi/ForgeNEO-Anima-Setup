# SPDX-License-Identifier: AGPL-3.0-or-later
"""Read safetensors headers without loading model weights or importing CUDA."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import struct
from .contracts import PH3Error, weight_path

MAX_HEADER = 64 * 1024 * 1024


def read_header(path: str | Path) -> tuple[dict, dict]:
    p = weight_path(str(path))
    with p.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise PH3Error(f"safetensorsのヘッダーがありません: {p.name}")
        length, = struct.unpack("<Q", raw)
        if length < 2 or length > MAX_HEADER or length + 8 > p.stat().st_size:
            raise PH3Error(f"safetensorsのヘッダーサイズが不正です: {p.name}")
        try:
            header = json.loads(f.read(length))
        except (ValueError, UnicodeDecodeError) as exc:
            raise PH3Error(f"ヘッダーを読めません: {p.name}: {exc}") from exc
    if not isinstance(header, dict):
        raise PH3Error("safetensorsヘッダーが辞書ではありません。")
    metadata = header.pop("__metadata__", {})
    body_size = p.stat().st_size - 8 - length
    for name, entry in header.items():
        if not isinstance(entry, dict) or "shape" not in entry or "data_offsets" not in entry:
            raise PH3Error(f"不正なテンソル定義: {name}")
        a, b = entry["data_offsets"]
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b <= body_size):
            raise PH3Error(f"不完全なモデルファイルです: {name}")
    return header, metadata


def normalized_key(key: str) -> str:
    # Official P-H3 Comfy encoder conversion uses these two namespaces.
    for old, new in (("model.language_model.", "model."), ("model.visual.", "visual.")):
        if key.startswith(old):
            return new + key[len(old):]
    return key


def count_layers(keys, prefix: str) -> int:
    found = set()
    for key in keys:
        if key.startswith(prefix):
            part = key[len(prefix):].split(".", 1)[0]
            if part.isdigit():
                found.add(int(part))
    if not found or found != set(range(max(found) + 1)):
        raise PH3Error(f"層の番号が欠けています: {prefix}")
    return max(found) + 1


def inspect_component(path: str | Path, kind: str) -> dict:
    header, metadata = read_header(path)
    hdr = {normalized_key(k): v for k, v in header.items()}
    if kind == "dit":
        needed = ("video_patch_proj.weight", "audio_patch_proj.weight", "condition_proj.weight", "final_layer.video_out.weight")
        for key in needed:
            if key not in hdr:
                raise PH3Error(f"P-H3画像生成モデルではありません。必要なキー: {key}")
        if hdr["final_layer.video_out.weight"]["shape"][0] != 96:
            raise PH3Error("対応していない画像出力ヘッドです。P-H3単体モデルを選択してください。")
        layers = count_layers(hdr, "blocks.")
    elif kind == "text_encoder":
        if "text_connector.proj.weight" not in hdr:
            raise PH3Error("P-H3専用TextConnectorがありません。通常のH3用32Bエンコーダーは使用できません。")
        if hdr["text_connector.proj.weight"]["shape"] != [5120, 12800]:
            raise PH3Error("P-H3専用TextConnectorのサイズが一致しません。")
        layers = count_layers(hdr, "model.layers.")
        if layers != 36:
            raise PH3Error(f"P-H3用Qwen3-VL 4Bの36層を期待しましたが、{layers}層です。")
    elif kind == "vae":
        key = "decoder.proj_out.weight"
        if key not in hdr or hdr[key]["shape"][0] != 768:
            raise PH3Error("P-H3専用の静止画VAEではありません。通常のH3動画VAEは使用できません。")
        layers = count_layers(hdr, "decoder.transformer_blocks.")
    else:
        raise PH3Error(f"不明な部品: {kind}")
    formats = _quant_formats(path, header, metadata)
    return {"kind": kind, "name": Path(path).name, "bytes": Path(path).stat().st_size,
            "layers": layers, "tensors": len(header), "quant_formats": formats,
            "dtypes": dict(Counter(v.get("dtype", "unknown") for v in hdr.values()))}


def _quant_formats(path, header, metadata) -> list[str]:
    formats = set()
    qmeta = metadata.get("_quantization_metadata") if isinstance(metadata, dict) else None
    if qmeta:
        try:
            obj = json.loads(qmeta)
            for entry in obj.get("layers", {}).values():
                if isinstance(entry, dict) and "format" in entry:
                    formats.add(str(entry["format"]))
        except (ValueError, AttributeError):
            raise PH3Error("量子化メタデータが不正です。")
    # Some versions serialize per-layer metadata as byte tensors.
    with Path(path).open("rb") as f:
        length, = struct.unpack("<Q", f.read(8))
        for key, entry in header.items():
            if not key.endswith(".comfy_quant"):
                continue
            a, b = entry["data_offsets"]
            if b - a > 65536:
                raise PH3Error("量子化レイヤー情報のサイズが不正です。")
            f.seek(8 + length + a)
            try:
                formats.add(str(json.loads(f.read(b - a))["format"]))
            except (ValueError, KeyError):
                raise PH3Error(f"量子化レイヤー情報が不正です: {key}")
    return sorted(formats)
