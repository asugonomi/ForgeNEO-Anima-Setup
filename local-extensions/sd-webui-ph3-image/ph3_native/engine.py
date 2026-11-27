# SPDX-License-Identifier: AGPL-3.0-or-later
"""Single-job, staged GPU execution using Forge's own memory manager."""
from __future__ import annotations
from datetime import datetime
from dataclasses import replace
import gc
import json
from pathlib import Path
import threading
import time
import traceback
import uuid

from . import __version__
from .contracts import ROOT, Request, PH3Error, Cancelled, GIB, fingerprint, SIGMAS, UPSTREAM_COMMIT
from .diagnostics import runtime_checks
from .loading import load_component
from .output import save_result
from .sampling import sample


class Engine:
    def __init__(self):
        self.cancel_event = threading.Event()
        self.job_guard = threading.Lock()
        self.components = {}
        self.text_cache = None
        self.last_outputs = []
        self.log_path = None
        self.events = []
        self.cleanup_ok = False

    def cancel(self):
        self.cancel_event.set()

    def check(self):
        from modules import shared
        if self.cancel_event.is_set() or getattr(shared.state, "interrupted", False):
            raise Cancelled("生成を中止しました。GPU上の処理が区切れるまで少し待つ場合があります。")

    def log(self, message, console=True):
        event = {"time": datetime.now().isoformat(timespec="milliseconds"), "message": str(message)}
        self.events.append(event)
        if console:
            print("[P-H3]", message)
        if self.log_path:
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def unload_gpu(self):
        from backend import memory_management as mm
        # Host code also holds queue_lock while using this shared manager.
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()

    def clear_cache(self):
        self.unload_gpu()
        self.components.clear()
        self.text_cache = None
        gc.collect()
        from backend import memory_management as mm
        mm.cleanup_models()
        mm.soft_empty_cache()

    def _get(self, path, kind):
        self.check()
        key = kind, fingerprint(path)
        if key not in self.components:
            # Remove older versions/paths for this component, without growing a model collection in RAM.
            family = {"text_encoder", "text_encoder_edit"} if kind.startswith("text_encoder") else {kind}
            for old in [k for k in self.components if k[0] in family]:
                del self.components[old]
            try:
                import psutil
                if psutil.virtual_memory().available < 8 * GIB:
                    self.log("空きRAMが8 GiB未満のためCPUモデルキャッシュを破棄します。")
                    self.clear_cache()
            except ImportError:
                pass
            self.components[key] = load_component(path, kind, log=self.log)
        return self.components[key]

    def _activate(self, patcher, reserve):
        from backend import memory_management as mm
        self.check()
        self.unload_gpu()
        mm.load_models_gpu([patcher], memory_required=reserve * GIB,
                           minimum_memory_required=reserve * GIB, force_full_load=False)

    def _snapshot(self, label):
        import torch
        from backend import memory_management as mm
        device = mm.get_torch_device()
        free, total = torch.cuda.mem_get_info(device)
        snap = {"phase": label, "torch_allocated_gib": torch.cuda.memory_allocated(device) / GIB,
                "torch_reserved_gib": torch.cuda.memory_reserved(device) / GIB,
                "device_used_gib": (total - free) / GIB}
        try:
            import psutil
            snap["process_rss_gib"] = psutil.Process().memory_info().rss / GIB
        except ImportError:
            pass
        return snap

    def _encode(self, req, reserve, progress):
        import torch
        from .tokenizer import load_tokenizer
        from backend import memory_management as mm
        cache_key = (fingerprint(req.text_encoder), req.prompt, req.max_tokens)
        if self.text_cache and self.text_cache[0] == cache_key:
            self.log("同じプロンプトの文章埋め込みをCPUキャッシュから再利用します。")
            return self.text_cache[1]
        progress(0.02, "文章エンコーダーを準備")
        tokenizer = load_tokenizer(ROOT / "tokenizer")
        ids = tokenizer.encode(req.prompt, add_special_tokens=False)
        if not ids:
            ids = [151643]
        if len(ids) > req.max_tokens:
            raise PH3Error(f"プロンプトは{len(ids)}トークンです。上限{req.max_tokens}を超えました。勝手な切り詰めは行いません。")
        patcher, _ = self._get(req.text_encoder, "text_encoder")
        self._activate(patcher, min(reserve, 4.0))
        patcher.model.cancel_check = self.check
        tensor = torch.tensor([ids], dtype=torch.long, device=mm.get_torch_device())
        with torch.inference_mode():
            context = patcher.model(tensor).detach().cpu()
        self.check()
        self.text_cache = (cache_key, context)
        self.log(f"文章処理完了: {len(ids)} tokens, context {tuple(context.shape)}")
        return context

    def _encode_edit(self, req, reserve, tile_size, progress):
        import hashlib
        import torch
        from backend import memory_management as mm
        from .editing import open_source, image_tensor, size_within_area, edit_prompt, reference_prompt, scaled_size
        from .tokenizer import load_tokenizer
        device = mm.get_torch_device()
        paths = list(req.reference_images) if req.mode == "reference" else [req.source_image, *req.reference_images]
        sources = [open_source(path) for path in paths]
        source = sources[0]
        if req.mode == "edit" and scaled_size(*source.size, req.image_scale) != (req.width, req.height):
            raise PH3Error("編集開始前に入力画像のサイズが変わりました。画像を再選択してください。")
        ref_sizes = [size_within_area(*image.size, req.width * req.height) for image in sources]
        vision_sizes = [size_within_area(*image.size, 1_000_000) for image in sources]
        self.log(f"{req.mode}: 入力 {len(sources)}枚, 出力 {req.width}×{req.height}")
        progress(.01, "元画像をタイルVAEでエンコード")
        encoder, _ = self._get(req.vae, "vae_encoder")
        self._activate(encoder, min(reserve, 4.0))
        encoder.model.tile_size = tile_size
        encoder.model.cancel_check = self.check
        refs = []
        for index, (image, ref_size) in enumerate(zip(sources, ref_sizes)):
            self.check()
            progress(.01 + .03 * index / len(sources), f"入力画像 {index+1}/{len(sources)}をVAEでエンコード")
            pixels = image_tensor(image, ref_size).permute(0, 3, 1, 2).unsqueeze(2)
            with torch.inference_mode():
                z = encoder.model.encode(pixels.to(device=device, dtype=torch.float32).mul(2).sub(1)).detach().cpu()
            if z.shape != (1, 24, 1, ref_size[1] // 16, ref_size[0] // 16) or not bool(torch.isfinite(z).all()):
                raise PH3Error(f"入力画像{index+1}のVAEエンコードが不正です。")
            refs.append({"kind": "image", "latent_h": z.shape[-2], "latent_w": z.shape[-1], "latent": z})
            self.log(f"入力画像{index+1}: 参照潜在用 {ref_size}, 視覚用 {vision_sizes[index]}", console=False)
            del pixels
        self.check()
        del encoder
        self.unload_gpu()
        progress(.05, "元画像の内容と編集指示をエンコード")
        tokenizer = load_tokenizer(ROOT / "tokenizer")
        effective_prompt = reference_prompt(req.prompt, len(sources)) if req.mode == "reference" else edit_prompt(req.prompt)
        prefixes = [tokenizer.encode(f"<Picture {index+1}>: ", add_special_tokens=False) + [151652]
                    for index in range(len(sources))]
        suffix = tokenizer.encode(effective_prompt, add_special_tokens=False)
        token_count = sum(len(prefix) + 1 for prefix in prefixes) + len(suffix)
        if token_count > req.max_tokens:
            raise PH3Error(f"編集指示は参照用タグ込みで{token_count}トークンです。上限{req.max_tokens}を超えました。")
        patcher, _ = self._get(req.text_encoder, "text_encoder_edit")
        self._activate(patcher, min(reserve, 4.0))
        patcher.model.cancel_check = self.check
        hooks = [b.register_forward_pre_hook(lambda *_: self.check()) for b in patcher.model.visual.blocks]
        try:
            with torch.inference_mode():
                context = patcher.model(torch.tensor([suffix], device=device, dtype=torch.long),
                                        images=[image_tensor(image, size) for image, size in zip(sources, vision_sizes)],
                                        prefix_ids=[torch.tensor([prefix], device=device, dtype=torch.long) for prefix in prefixes]).detach().cpu()
                tags = patcher.model.last_token_tags.clone()
        finally:
            for hook in hooks:
                hook.remove()
        if not bool(torch.isfinite(context).all()):
            raise PH3Error("画像・文章の埋め込みにNaN/Infがあります。")
        self.check()
        self.log(f"参照画像 {len(refs)}枚の読み取り完了")
        details = {"source_name": Path(req.source_image).name if req.source_image else None,
                   "source_pixel_sha256": hashlib.sha256(source.tobytes()).hexdigest() if req.source_image else None,
                   "source_size": list(source.size) if req.source_image else None,
                   "image_scale": req.image_scale if req.mode == "edit" else None,
                   "inputs": [{"picture": i+1, "name": Path(path).name, "size": list(image.size),
                               "pixel_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
                               "reference_size": list(ref_sizes[i]), "vision_size": list(vision_sizes[i])}
                              for i, (path, image) in enumerate(zip(paths, sources))],
                   "effective_prompt": effective_prompt, "reference_encode_seed": 42,
                   "size_alignment": "explicit canvas dimensions" if req.mode == "reference" else "nearest multiple of 32, ties up; previewed in UI"}
        return context, tags, refs, details

    def _attempt(self, req, seed, reserve, tile_size, progress):
        import torch
        from PIL import Image
        from backend import memory_management as mm
        device = mm.get_torch_device()
        start = time.perf_counter()
        phases = []
        refs, tags, edit_details = None, None, {}
        if req.source_image or req.mode == "reference":
            context_cpu, tags, refs, edit_details = self._encode_edit(req, reserve, tile_size, progress)
        else:
            context_cpu = self._encode(req, reserve, progress)
        phases.append(self._snapshot("text_encoded"))
        # Explicitly offload the previous stage before constructing the next one.
        self.unload_gpu()
        progress(0.12, "画像生成モデルを準備")
        patcher, _ = self._get(req.dit, "dit")
        self._activate(patcher, reserve)
        self.log(f"DiT: 作業用VRAM目安 {reserve:.1f} GiB（上限保証ではありません）")
        hooks = [b.register_forward_pre_hook(lambda *_: self.check()) for b in patcher.model.blocks]
        try:
            with torch.inference_mode():
                z = sample(patcher.model, context_cpu.to(device=device, dtype=torch.bfloat16),
                           int(req.width), int(req.height), seed, device, check=self.check,
                           progress=lambda step: progress(0.15 + .70 * step / 8, f"画像生成 {step}/8"),
                           **({"refs": refs, "token_tags": tags} if refs else {}))
        finally:
            for hook in hooks:
                hook.remove()
        phases.append(self._snapshot("denoised"))
        self.unload_gpu()
        del patcher, hooks  # CPU cache policy alone decides whether to retain the DiT.
        progress(.88, "VAEで画像を復元")
        vae, _ = self._get(req.vae, "vae")
        self._activate(vae, min(reserve, 4.0))
        vae.model.tile_size = int(tile_size)
        vae.model.cancel_check = self.check
        hooks = [b.register_forward_pre_hook(lambda *_: self.check()) for b in vae.model.decoder.transformer_blocks]
        try:
            with torch.inference_mode():
                pixels = vae.model.decode(z.to(device=device, dtype=torch.float16))
                if not bool(torch.isfinite(pixels).all()):
                    raise PH3Error("VAE出力にNaN/Infがあります。壊れた画像は保存しません。")
                if pixels.shape != (1, 3, 1, int(req.height), int(req.width)):
                    raise PH3Error(f"VAE出力サイズが不正です: {tuple(pixels.shape)}")
                array = (pixels[0, :, 0].clamp(0, 1).permute(1, 2, 0).float() * 255).round().byte().cpu().numpy()
        finally:
            for hook in hooks:
                hook.remove()
        phases.append(self._snapshot("decoded"))
        self.check()
        image = Image.fromarray(array, mode="RGB")
        return image, {"elapsed_seconds": time.perf_counter() - start, "phase_memory": phases,
                       "tile_size": tile_size, "reserve_gib": reserve, **edit_details}

    def generate(self, req: Request, output_root: Path, progress=lambda value, text: None, *, translate=False, task_id="", llm_model="", plamo_model="", lm_base="http://127.0.0.1:1234"):
        import torch
        from backend import memory_management as mm
        from modules import shared, progress as forge_progress
        from modules.call_queue import queue_lock

        req.checked()
        if not self.job_guard.acquire(blocking=False):
            raise PH3Error("P-H3の処理がすでに実行中です。")
        began = False
        import re
        task_id = task_id if re.fullmatch(r"task\([A-Za-z0-9-]+\)", task_id or "") else forge_progress.create_task_id("ph3")
        forge_progress.add_task_to_queue(task_id)
        self.cancel_event.clear()
        try:
            # Share Forge's generation lock, including preparation, retries and cleanup.
            with queue_lock:
                forge_progress.start_task(task_id)
                self.cleanup_ok = False
                self.last_outputs = []
                self.events = []
                logs = ROOT / "logs"
                logs.mkdir(exist_ok=True)
                self.log_path = logs / (datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:8] + ".jsonl")
                runtime = runtime_checks()
                self.log(json.dumps(runtime, ensure_ascii=False, default=str), console=False)
                if hasattr(shared.state, "begin"):
                    shared.state.begin("P-H3-Image")
                    began = True
                    shared.state.job_count = req.count
                    shared.state.sampling_steps = 1000
                try:
                    original_prompt = req.prompt
                    self.unload_gpu()
                    from .local_prompt import process_prompt, mode_key
                    prompt_mode = mode_key(translate)
                    def prompt_status(text):
                        shared.state.textinfo = text
                    req = replace(req, prompt=process_prompt(req.prompt, prompt_mode, llm_model, plamo_model,
                                  lm_base, check=self.check, report=prompt_status,
                                  image_paths=req.reference_images if req.mode == "reference" else tuple(p for p in (req.source_image, *req.reference_images) if p),
                                  composition=req.mode == "reference"))
                    self.check()
                    seeds = req.seeds()
                    for index, seed in enumerate(seeds):
                        self.check()
                        shared.state.job_no = index
                        shared.state.sampling_step = 0
                        self.log(f"画像 {index+1}/{len(seeds)}, seed={seed}")
                        torch.cuda.reset_peak_memory_stats(mm.get_torch_device())
                        attempts = 2 if req.retry_oom else 1
                        for attempt in range(attempts):
                            reserve = min(12.0, req.reserve_gib + 2.0 * attempt)
                            tile = max(128, req.tile_size // (2 ** attempt))
                            def update(value, text):
                                progress((index + value) / len(seeds), f"{index+1}/{len(seeds)} | {text}")
                                if hasattr(shared.state, "textinfo"):
                                    shared.state.textinfo = f"{index+1}/{len(seeds)}枚 · {text}"
                                    shared.state.sampling_step = round(value * 1000)
                            oom_message = None
                            try:
                                image, details = self._attempt(req, seed, reserve, tile, update)
                            except torch.cuda.OutOfMemoryError as e:
                                oom_message = str(e)
                            # Leave the exception handler before GC, so traceback frames don't keep CUDA tensors.
                            if oom_message is not None:
                                self.log("CUDA OOM: " + oom_message)
                                self.unload_gpu()
                                if attempt + 1 < attempts:
                                    self.log("同じSeed・同じ解像度で一度だけ再試行。GPU常駐量を減らし、VAEタイルを縮小します。")
                                    continue
                                raise PH3Error("VRAM不足で停止しました。解像度は変更していません。ログを確認してください。")
                            details.update(version=__version__, native_commit=UPSTREAM_COMMIT,
                                           seed=seed, width=req.width, height=req.height, sigmas=list(SIGMAS),
                                           prompt_upsampler=(prompt_mode == "official"), original_prompt=original_prompt,
                                           translated_prompt=req.prompt if prompt_mode != "none" else None,
                                           prompt_mode=prompt_mode, save_json=req.save_json,
                                           mode="reference_to_image" if req.mode == "reference" else ("instruction_edit" if req.source_image else "text_to_image"), attempt=attempt + 1,
                                           model_files={k: Path(getattr(req, k)).name for k in ("dit", "text_encoder", "vae")},
                                           gpu=runtime["gpu"], torch=runtime["torch"], cuda=runtime["cuda_build"],
                                           peak_torch_allocated_gib=torch.cuda.max_memory_allocated(mm.get_torch_device()) / GIB,
                                           peak_torch_reserved_gib=torch.cuda.max_memory_reserved(mm.get_torch_device()) / GIB)
                            path = save_result(image, output_root, req, seed, details)
                            self.last_outputs.append(str(path))
                            self.log(f"保存: {path}")
                            break
                    progress(1.0, "完了")
                    return list(self.last_outputs)
                except Cancelled as e:
                    self.log(str(e))
                    raise
                except BaseException:
                    self.log(traceback.format_exc())
                    raise
                finally:
                    try:
                        self.unload_gpu()
                        if not req.keep_cpu_cache:
                            self.clear_cache()
                        self.cleanup_ok = True
                    except Exception as e:
                        self.log(f"メモリ解放処理で例外: {e}。解放完了とは見なさず、WebUIの再起動を検討してください。")
                    if began:
                        shared.state.end()
        finally:
            forge_progress.finish_task(task_id)
            forge_progress.pending_tasks.pop(task_id, None)
            self.job_guard.release()


ENGINE = Engine()
