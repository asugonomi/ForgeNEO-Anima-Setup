# SPDX-License-Identifier: AGPL-3.0-or-later
# P-H3-specific adaptations of the reference conditioning and image VAE.
# See NOTICE for upstream attribution and the exact source revisions.
from __future__ import annotations
import inspect
import torch
from torch import nn
from .contracts import FRAME_POSITION, HIDDEN_STATES, PH3Error


class TextConnector(nn.Module):
    """Five Qwen hidden states -> normalized projection and the learned sink token."""
    def __init__(self, input_dim=12800, output_dim=5120):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim, bias=False, dtype=torch.float32)
        self.input_mean = nn.Parameter(torch.zeros(input_dim, dtype=torch.float32))
        self.input_std = nn.Parameter(torch.ones(input_dim, dtype=torch.float32))
        self.output_mean = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))
        self.output_std = nn.Parameter(torch.ones(output_dim, dtype=torch.float32))
        self.sink_embedding = nn.Parameter(torch.empty(output_dim, dtype=torch.float32))

    def forward(self, hidden_states: list[torch.Tensor], dtype=torch.bfloat16):
        x = torch.cat(hidden_states, dim=-1).float()
        x = (x - self.input_mean.to(x)) / self.input_std.to(x)
        x = self.proj(x)
        x = x * self.output_std.to(x) + self.output_mean.to(x)
        sink = self.sink_embedding.to(device=x.device, dtype=x.dtype)
        return torch.cat((sink.expand(x.shape[0], 1, -1), x[:, 1:]), dim=1).to(dtype)


def make_text_encoder_class(with_vision=False):
    from backend.nn.llm import llama

    class PH3TextEncoder(nn.Module):
        """Qwen hidden-state connector, with an optional image-editing vision tower."""
        def __init__(self):
            super().__init__()
            config = llama.Qwen3VL_4BConfig()
            if config.hidden_size != 2560 or config.num_hidden_layers != 36:
                raise PH3Error("ForgeのQwen3VL_4BConfigが想定と一致しません。")
            self.model = llama.Llama2_(config)
            if with_vision:
                from backend.nn.llm.qwen35 import QWEN3VL_VISION_COMMON, QWEN3VL_VISION, Qwen3VLVisionModel
                self.visual = Qwen3VLVisionModel({**QWEN3VL_VISION_COMMON, **QWEN3VL_VISION["qwen3vl_4b"],
                                                "out_hidden_size": config.hidden_size})
            self.text_connector = TextConnector()
            self.dtype = torch.bfloat16
            self.cancel_check = lambda: None

        def forward(self, ids, images=None, prefix_ids=None):
            cfg = self.model.config
            x = self.model.embed_tokens(ids).to(dtype=self.dtype)
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            visual_mask, deepstack = None, None
            self.last_token_tags = None
            if images:
                self.cancel_check()
                if not with_vision or prefix_ids is None:
                    raise PH3Error("画像編集用の視覚エンコーダーが読み込まれていません。")
                if len(images) != len(prefix_ids) or len(images) > 8:
                    raise PH3Error("参照画像の枚数と画像タグの数が一致しません。")
                parts, info, offset = [], [], 0
                end = self.model.embed_tokens(torch.tensor([[151653]], device=x.device)).to(x)
                for image, prefix_tokens in zip(images, prefix_ids):
                    self.cancel_check()
                    merged, extra = llama.Qwen3VL.preprocess_embed(self, {"type": "image", "data": image}, x.device)
                    prefix = self.model.embed_tokens(prefix_tokens).to(dtype=self.dtype)
                    index, size = offset + prefix.shape[1], merged.shape[0]
                    parts.extend((prefix, merged[None].to(x), end))
                    info.append({"type": "image", "index": index, "size": size, "extra": extra})
                    offset = index + size + 1
                x = torch.cat((*parts, x), dim=1)
                if x.shape[1] > 12288:
                    raise PH3Error("画像を含む文章系列が12288トークンを超えました。指示文または参照画像を減らしてください。")
                pos, visual_mask, deepstack = llama.Qwen3VL.build_image_inputs(self, x, info)
                self.last_token_tags = torch.ones(x.shape[1], dtype=torch.long)
                for entry in info:
                    self.last_token_tags[entry['index'] - 1:entry['index'] + entry['size'] + 1] = 0
            kw = {"device": x.device}
            if "interleaved_mrope" in inspect.signature(llama.precompute_freqs_cis).parameters:
                kw["interleaved_mrope"] = getattr(cfg, "interleaved_mrope", False)
            freqs = llama.precompute_freqs_cis(cfg.head_dim, pos, cfg.rope_theta,
                                              cfg.rope_scale, cfg.rope_dims, **kw)
            mask = torch.empty(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
            mask.fill_(torch.finfo(x.dtype).min / 4).triu_(1)
            captured = []
            for i, layer in enumerate(self.model.layers):
                self.cancel_check()
                # Transformers hidden_states[n] is the input to layer n, not its output.
                if i in HIDDEN_STATES:
                    captured.append(x.clone())
                x, _ = layer(x=x, attention_mask=mask, freqs_cis=freqs,
                             optimized_attention=llama.attention_function, past_key_value=None)
                if deepstack is not None and i < len(deepstack):
                    x[visual_mask] = x[visual_mask] + deepstack[i].to(x)
            if self.model.norm is not None:
                x = self.model.norm(x)
            captured.append(x)  # hidden_states[36], including final normalization.
            if len(captured) != 5:
                raise PH3Error("必要な5層の隠れ状態を取得できませんでした。")
            return self.text_connector(captured, dtype=self.dtype)

    return PH3TextEncoder


def image_token_ids(height, width, *, device, dtype):
    """The trained image frame position, followed by spatial patch-center coordinates."""
    yy = 2 * ((torch.arange(height, device=device, dtype=dtype) + .5) / height) - 1
    xx = 2 * ((torch.arange(width, device=device, dtype=dtype) + .5) / width) - 1
    y, x = torch.meshgrid(yy, xx, indexing="ij")
    return torch.stack((torch.full_like(y, FRAME_POSITION), y, x), dim=-1).reshape(1, -1, 3)


def make_decoder_class(native):
    class PH3ImageDecoder(native.ViT3DDecoder):
        def __init__(self, **kwargs):
            super().__init__(patch_size_t=1, **kwargs)
            del self.mask_token

        def forward(self, x):
            batch, _, frames, height, width = x.shape
            if frames != 1:
                raise PH3Error("P-H3静止画VAEには1フレームだけを渡してください。")
            h = self.x_embedder(x.flatten(2).transpose(1, 2))
            patches = h.shape[1]
            suffix = 1 + self.num_register_tokens
            registers = native.kernels.cast_to(self.register_tokens, dtype=h.dtype, device=h.device)
            h = torch.cat((h, registers.expand(batch, -1, -1), torch.zeros_like(h[:, :1])), dim=1)
            ids = image_token_ids(height, width, device=x.device, dtype=x.dtype).expand(batch, -1, -1)
            ids = torch.cat((ids, torch.zeros(batch, suffix, 3, device=x.device, dtype=ids.dtype)), dim=1)
            rotary = self.pos_embed(ids)
            for block in self.transformer_blocks:
                h = block(h, rotary)
            pixels = self.proj_out(self.norm_out(h))[:, :patches]
            pixels = pixels.view(batch, 1, height, width, self.out_channels, 1, self.patch_size, self.patch_size)
            return pixels.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous().reshape(
                batch, self.out_channels, 1, height * self.patch_size, width * self.patch_size)

    return PH3ImageDecoder


def make_vae_class(native):
    Decoder = make_decoder_class(native)

    class PH3ImageVAE(native.MiniMaxH3VideoVAE):
        """Decode-only image VAE; no unused video encoder or audio VAE is allocated."""
        def __init__(self, num_layers=36, tile_size=256):
            nn.Module.__init__(self)
            self.vae_ratio = 16
            self.vae_ratio_t = 1
            self.tiling = True
            self.tile_size = tile_size
            self.tile_overlap_min = 64
            self.post_quant_conv = nn.Conv3d(24, 24, 1)
            self.decoder = Decoder(num_layers=num_layers)
            self.register_buffer("latents_mean", torch.tensor(native.LATENTS_MEAN))
            self.register_buffer("latents_std", torch.tensor(native.LATENTS_STD))
            self.register_buffer("pixel_mean", torch.tensor(native.IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
            self.register_buffer("pixel_std", torch.tensor(native.IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
            self.dtype = torch.float16
            self.cancel_check = lambda: None

        def _decode_tile_row(self, z_row, x_idx, x_len):
            # Fixed batch=1, instead of the video's free-memory-based batching.
            for start, length in zip(x_idx, x_len):
                self.cancel_check()
                tile = z_row[..., start // 16:(start + length) // 16]
                yield self._decode_pixels(tile)

        def decode(self, z, output_buffer=None):
            if z.ndim != 5 or z.shape[2] != 1:
                raise PH3Error("P-H3のVAE入力サイズが不正です。")
            return super().decode(z, output_buffer=output_buffer)

    return PH3ImageVAE
