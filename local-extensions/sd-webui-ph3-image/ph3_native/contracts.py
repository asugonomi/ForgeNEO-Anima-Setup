# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validated request format and model-independent sampling constants."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
import math
import secrets

ROOT = Path(__file__).resolve().parent.parent
UPSTREAM_REPO = "eduardoabreu81/minimax-h3-forge-neo"
UPSTREAM_COMMIT = "9387d9189273384f14143707898126abbcb7ae74"
COMFY_REFERENCE_COMMIT = "72bcda7696471e5b673c52dabb216650e7c5b71a"
MODEL_REPO = "PrunaAI/P-H3-Image-Comfy"
SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718, 0.0)
HIDDEN_STATES = (13, 19, 25, 31, 36)
FRAME_POSITION = 2.0 * (0.5 / 7.0) - 1.0
GIB = 1024 ** 3

class PH3Error(RuntimeError):
    """A user-actionable error, not a silently degraded inference path."""

class Cancelled(PH3Error):
    pass


def clean_path(value: str) -> Path:
    value = str(value).strip().strip('"')
    if not value:
        raise PH3Error("モデルのファイルパスを指定してください。")
    return Path(value).expanduser().resolve()


def weight_path(value: str) -> Path:
    p = clean_path(value)
    if not p.is_file() or p.suffix.lower() != ".safetensors":
        raise PH3Error(f"safetensors ファイルを指定してください: {p}")
    return p


@dataclass(frozen=True)
class Request:
    prompt: str
    dit: str
    text_encoder: str
    vae: str
    width: int = 2048
    height: int = 2048
    seed: int = -1
    count: int = 1
    reserve_gib: float = 6.0
    tile_size: int = 256
    keep_cpu_cache: bool = True
    retry_oom: bool = True
    max_tokens: int = 1024
    source_image: str | None = None
    image_scale: float = 1.0
    reference_images: tuple[str, ...] = ()
    mode: str = "edit"
    save_json: bool = False

    def checked(self, check_files: bool = True) -> "Request":
        if self.mode not in ("edit", "reference"):
            raise PH3Error("不明な画像処理モードです。")
        if self.mode == "reference":
            if self.source_image is not None:
                raise PH3Error("参照から作成モードにはメイン画像を指定しません。")
            if not 1 <= len(self.reference_images) <= 8:
                raise PH3Error("参照画像を1〜8枚指定してください。")
            from .editing import open_source
            for path in self.reference_images:
                open_source(path)
        elif len(self.reference_images) > 7:
            raise PH3Error("参照画像は最大7枚（メイン画像を含めて8枚）です。")
        if self.mode == "edit" and self.reference_images and not self.source_image:
            raise PH3Error("編集対象のメイン画像を選択してください。")
        if self.source_image:
            from .editing import open_source, scaled_size
            image = open_source(self.source_image)
            expected = scaled_size(*image.size, self.image_scale)
            if (self.width, self.height) != expected:
                raise PH3Error(f"入力画像×倍率の出力サイズ {expected[0]}×{expected[1]} と指定サイズが一致しません。")
            for path in self.reference_images:
                open_source(path)
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise PH3Error("プロンプトを入力してください。")
        if len(self.prompt) > 24000:
            raise PH3Error("プロンプトは24,000文字以内にしてください。切り詰めは行いません。")
        for name in ("width", "height", "seed", "count", "tile_size", "max_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value):
                raise PH3Error(f"{name} は整数にしてください。")
        if not (256 <= self.width <= 4096 and 256 <= self.height <= 4096):
            raise PH3Error("幅・高さは256〜4096にしてください。")
        if self.width % 32 or self.height % 32:
            raise PH3Error("幅・高さは32の倍数にしてください。自動変更は行いません。")
        if self.width * self.height > 9_000_000:
            raise PH3Error("この実験版では900万画素を上限にしています。")
        if self.seed < -1 or self.seed > 2**63 - 1:
            raise PH3Error("Seedは-1（ランダム）または0〜2^63-1にしてください。")
        if not 1 <= self.count <= 16:
            raise PH3Error("連続生成枚数は1〜16です。GPU上の同時バッチは常に1です。")
        if isinstance(self.reserve_gib, bool) or not isinstance(self.reserve_gib, (int, float)) or not math.isfinite(self.reserve_gib) or not 3 <= self.reserve_gib <= 12:
            raise PH3Error("作業用VRAMの目安は3〜12 GiBにしてください。")
        if self.tile_size not in (128, 256, 512):
            raise PH3Error("VAEタイルは128 / 256 / 512から選んでください。")
        if not 64 <= self.max_tokens <= 2048:
            raise PH3Error("トークン上限は64〜2048にしてください。")
        if check_files:
            for name in ("dit", "text_encoder", "vae"):
                weight_path(getattr(self, name))
        return self

    def seeds(self) -> list[int]:
        first = secrets.randbelow(2**63) if self.seed == -1 else int(self.seed)
        return [(first + i) % 2**63 for i in range(int(self.count))]

    def as_dict(self) -> dict:
        return asdict(self)


def fingerprint(path: str | Path) -> tuple:
    p = clean_path(str(path))
    stat = p.stat()
    return str(p), stat.st_size, stat.st_mtime_ns


def stream_shapes(width: int, height: int) -> tuple[tuple, tuple]:
    """The image's single latent frame and its two stereo audio latents."""
    return (1, 24, 1, height // 16, width // 16), (1, 32, 2, 2)
