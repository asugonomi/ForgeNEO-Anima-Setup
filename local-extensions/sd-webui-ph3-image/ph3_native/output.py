# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations
from datetime import datetime
import json
from pathlib import Path
import re
from PIL import Image, PngImagePlugin
from . import __version__


def save_result(image: Image.Image, directory: Path, request, seed: int, details: dict) -> Path:
    directory = Path(directory) / datetime.now().strftime("%Y-%m-%d")
    directory.mkdir(parents=True, exist_ok=True)
    parameters = (f"{request.prompt}\nNegative prompt: \n"
                  f"Steps: 8, Sampler: P-H3 Euler (fixed sigmas), CFG scale: 1, Seed: {seed}, "
                  f"Size: {request.width}x{request.height}, Model: {Path(request.dit).stem}, "
                  f"P-H3 extension: {__version__}, VAE tile: {details.get('tile_size', request.tile_size)}, "
                  "Powered by MiniMaxH3")
    png = PngImagePlugin.PngInfo()
    png.add_text("parameters", parameters)
    png.add_text("Software", f"Forge Neo P-H3-Image {__version__}; Powered by MiniMaxH3")
    png.add_text("ph3_generation", json.dumps(details, ensure_ascii=False))
    # A daily, fixed-width serial sorts by generation order, independently of Seed.
    serials = [int(m[1]) for p in directory.iterdir() if (m := re.match(r"^(\d{8})-ph3-", p.name))]
    number = max(serials, default=0) + 1
    while True:
        if number >= 100_000_000:
            raise RuntimeError("出力フォルダーの連番上限に達しました。")
        path = directory / f"{number:08d}-ph3-{seed}.png"
        reservation = directory / f"{number:08d}-ph3-reserve.lock"
        try:
            with reservation.open("x"):
                pass
        except FileExistsError:
            number += 1
            continue
        try:
            handle = path.open("xb")
            break
        except FileExistsError:
            reservation.unlink()
            number += 1
        except BaseException:
            reservation.unlink()
            raise
    try:
        with handle:
            image.save(handle, format="PNG", pnginfo=png)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    finally:
        reservation.unlink(missing_ok=True)
    if getattr(request, "save_json", False):
        sidecar = path.with_suffix(".json")
        with sidecar.open("x", encoding="utf-8") as f:
            json.dump({"parameters": parameters, **details}, f, ensure_ascii=False, indent=2)
    return path
