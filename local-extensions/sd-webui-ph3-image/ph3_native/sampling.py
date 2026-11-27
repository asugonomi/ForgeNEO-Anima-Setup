# SPDX-License-Identifier: AGPL-3.0-or-later
"""The fixed eight-step Euler schedule, independently testable on CPU."""
from __future__ import annotations
import torch
from tqdm.auto import tqdm
from .contracts import SIGMAS, stream_shapes, PH3Error


def sample(dit, context, width, height, seed, device, check=lambda: None, progress=lambda i: None,
           refs=None, token_tags=None):
    shapes = stream_shapes(width, height)
    rng = torch.Generator(device="cpu").manual_seed(seed)
    # Generate both streams, even though only the image is eventually decoded.
    streams = [torch.randn(shape, generator=rng, dtype=torch.float32).to(device) for shape in shapes]
    sigmas = torch.tensor(SIGMAS, dtype=torch.float32, device=device)
    # Cache the refined text once, rather than running the token refiner eight times.
    context = dit.preprocess_text_embeds(context)
    payload = {"audio_scale": 1.0, "seed": seed}
    if refs:
        payload["refs"] = refs
        payload["cond_video_latents"] = [ref["latent"].to(device=device) for ref in refs]
        payload["text_token_tags"] = token_tags
    with tqdm(range(8), desc="P-H3", unit="step", disable=torch.device(device).type != "cuda") as steps:
        for i in steps:
            check()
            velocity = dit.forward_streams(streams, (sigmas[i] * 1000).reshape(1), context,
                                          transformer_options={"sample_sigmas": sigmas}, minimax_payload=payload)
            if len(velocity) != 2:
                raise PH3Error("DiTが画像・音声の2ストリームを返しませんでした。")
            # forward_streams returns velocity in the descending-sigma convention.
            delta = SIGMAS[i + 1] - SIGMAS[i]
            for x, v in zip(streams, velocity):
                if x.shape != v.shape:
                    raise PH3Error("生成中にストリームのサイズが変わりました。")
                x.add_(v, alpha=delta)
            del velocity
            progress(i + 1)
    image_latents = streams[0].detach().cpu()
    if not bool(torch.isfinite(image_latents).all()):
        raise PH3Error("生成された潜在画像にNaN/Infがあります。壊れた画像は保存しません。")
    return image_latents
