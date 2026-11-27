# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single-picture instruction editing, with explicit output-size calculation."""
from __future__ import annotations
import math
import re
from .contracts import PH3Error


def scaled_size(width, height, scale):
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise PH3Error("倍率は0より大きい有限の数値にしてください。")
    # Round half up; the UI shows both requested and aligned dimensions.
    w, h = (32 * math.floor(side * scale / 32 + .5) for side in (width, height))
    if not (256 <= w <= 4096 and 256 <= h <= 4096) or w * h > 9_000_000:
        raise PH3Error(f"倍率適用後は {w}×{h} です。各辺256〜4096、合計900万画素以内の倍率にしてください。")
    return w, h


def size_within_area(width, height, pixels):
    factor = min(1.0, math.sqrt(pixels / (width * height)))
    return tuple(max(32, round(side * factor / 32) * 32) for side in (width, height))


def edit_prompt(prompt):
    if re.match(r"\s*(Edit <Picture \d+>|subject_definitions:)", prompt):
        return prompt.strip()
    return "Edit <Picture 1>. " + prompt


def reference_prompt(prompt, count):
    pictures = ", ".join(f"<Picture {i+1}>" for i in range(count))
    return (f"Create a new composition using {pictures} as subject references. "
            "Use the subjects' identities and appearance from the references; compose their poses, "
            "actions and arrangement according to the following scene. "
            "No reference image is a fixed canvas.\n" + prompt.strip())


def open_source(path):
    from PIL import Image, ImageOps
    with Image.open(path) as original:
        if getattr(original, "n_frames", 1) != 1:
            raise PH3Error("編集の入力には静止画像を1枚指定してください。")
        return ImageOps.exif_transpose(original).convert("RGB")


def image_tensor(image, size):
    """CPU BHWC float pixels; reference conditioning uses Lanczos, no crop."""
    import numpy as np
    import torch
    from PIL import Image
    if image.size != size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    return torch.from_numpy(np.array(image, copy=True)).float().div_(255)[None]


def make_encoder_class(native):
    import torch
    from torch import nn

    class PH3ImageEncoder(native.MiniMaxH3VideoVAE):
        """Only encoder weights reside in this separately staged component."""
        def __init__(self):
            nn.Module.__init__(self)
            self.vae_ratio = 16
            self.tiling = True
            self.tile_size = 256
            self.tile_overlap_min = 64
            self.encoder = native.EncoderFCN3D(
                ch=128, ch_mult=[1, 2, 2, 4, 4, 8], space_down=[2, 2, 2, 2, 1, 1],
                time_down=[1, 2, 2, 1, 1, 1], num_res_blocks=2, in_channels=3,
                z_channels=24, double_z=True)
            self.quant_conv = nn.Conv3d(48, 48, 1)
            self.register_buffer("latents_mean", torch.tensor(native.LATENTS_MEAN))
            self.register_buffer("latents_std", torch.tensor(native.LATENTS_STD))
            self.register_buffer("pixel_mean", torch.tensor(native.IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
            self.register_buffer("pixel_std", torch.tensor(native.IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
            self.cancel_check = lambda: None

        def _encode_moments(self, x):
            self.cancel_check()
            return super()._encode_moments(x)

        def encode(self, x):
            if x.ndim != 5 or x.shape[2] != 1:
                raise PH3Error("編集入力は1フレームの画像にしてください。")
            moments = self._adaptive_encode(self._normalize_pixels(x))
            mean, logvar = moments.float().chunk(2, dim=1)
            # P-H3's trained reference encoding uses posterior sampling with fixed seed 42.
            noise = torch.randn(mean.shape, generator=torch.Generator(device="cpu").manual_seed(42)).to(mean.device)
            z = torch.addcmul(mean, torch.exp(.5 * logvar.clamp(-30, 20)), noise)
            return (z - self.latents_mean.view(1, -1, 1, 1, 1).to(z)) / self.latents_std.view(1, -1, 1, 1, 1).to(z)

    return PH3ImageEncoder
