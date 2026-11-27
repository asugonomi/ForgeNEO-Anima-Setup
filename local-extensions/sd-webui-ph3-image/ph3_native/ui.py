# SPDX-License-Identifier: AGPL-3.0-or-later
"""A separate WebUI tab. Does not replace txt2img, samplers, styles, or model detection."""
from __future__ import annotations
from datetime import datetime
import json
import math
import re
from dataclasses import replace
from pathlib import Path
import traceback
import gradio as gr
from .contracts import ROOT, Request, PH3Error, Cancelled
from .components import EditImage
from .settings import output_root
from .local_prompt import MODES, mode_key, models, process_prompt

CONFIG = ROOT / "config.json"


def settings():
    root = ROOT.parent.parent
    models = root / "models" / "P-H3-Image"
    data = {"dit": str(root.parent / "Model" / "DiffusionModels" / "P-H3-Image" / "p-h3-image_int8_convrot.safetensors"),
            "text_encoder": str(models / "text_encoders" / "p-h3-image_text_encoder_int8_convrot.safetensors"),
            "vae": str(models / "vae" / "p-h3-image_vae_int8_convrot.safetensors"),
            "width": 2048, "height": 2048, "seed": -1, "count": 1, "reserve_gib": 6.0,
            "tile_size": 256, "keep_cpu_cache": True, "retry_oom": True, "max_tokens": 1024, "image_scale": 1.0,
            "compose_width": 2048, "compose_height": 2048, "save_json": False,
            "prompt_mode": "公式P-H3 Prompt Upsampler（INT8・ローカル）",
            "llm_model": "gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m",
            "plamo_model": "plamo-3-translate-0.8b", "lm_base": "http://127.0.0.1:1234"}
    if CONFIG.is_file():
        try:
            saved = json.loads(CONFIG.read_text(encoding="utf-8"))
            data.update({k: saved[k] for k in data if k in saved})
        except (ValueError, OSError):
            print("[P-H3] config.json could not be read; it has NOT been overwritten.")
    return data


def _integer(value, name):
    if isinstance(value, bool):
        raise PH3Error(f"{name} は整数にしてください。")
    if isinstance(value, str):
        value = value.strip()
        if not re.fullmatch(r"[+-]?\d+", value):
            raise PH3Error(f"{name} は整数にしてください。")
        return int(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value != int(value):
        raise PH3Error(f"{name} は整数にしてください。")
    return int(value)


def make_request(prompt, dit, te, vae, width, height, seed, count, reserve, tile, keep, retry, tokens):
    return Request(prompt=prompt, dit=dit, text_encoder=te, vae=vae,
                   width=_integer(width, "幅"), height=_integer(height, "高さ"),
                   seed=_integer(seed, "Seed"), count=_integer(count, "枚数"),
                   reserve_gib=float(reserve), tile_size=_integer(tile, "VAEタイル"),
                   keep_cpu_cache=bool(keep), retry_oom=bool(retry),
                   max_tokens=_integer(tokens, "トークン上限"))


def stop():
    from .engine import ENGINE
    ENGINE.cancel()
    return "停止要求を受け付けました。実行中のGPU演算は即座には中断できず、層・タイルの境界で停止します。"


def unload():
    from .engine import ENGINE
    from modules.call_queue import queue_lock
    if not ENGINE.job_guard.acquire(blocking=False):
        return "生成中です。先に［停止］を押し、処理の終了を待ってください。"
    try:
        if not queue_lock.acquire(blocking=False):
            return "Forgeの別のGPU処理が実行中です。解放処理は行いませんでした。"
        try:
            ENGINE.clear_cache()
            return "P-H3のCPUモデル・文章キャッシュを破棄し、ForgeのGPUモデル解放を実行しました。"
        finally:
            queue_lock.release()
    except Exception as e:
        return "解放エラー: " + str(e)
    finally:
        ENGINE.job_guard.release()


def check_environment(dit, te, vae):
    from .diagnostics import diagnose, render_report
    report = diagnose({"dit": dit, "text_encoder": te, "vae": vae})
    text = render_report(report)
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)
    path = logs / ("diagnostic_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".log")
    path.write_text(text, encoding="utf-8")
    summary = "診断完了：モデルと実行環境を確認しました。" if report["api_and_headers_ok"] else "診断で問題が見つかりました。"
    issues = report["errors"] + report["warnings"]
    if issues:
        summary += "\n" + "\n".join(issues)
    return summary, str(path)


def edit_size_preview(source, scale):
    if not source:
        return "画像を選択すると出力サイズを表示します。"
    try:
        from .editing import open_source, scaled_size
        image = open_source(source)
        w, h = scaled_size(*image.size, float(scale))
        return (f"入力 {image.width}×{image.height} × {scale} → 出力 **{w}×{h}**\n\n"
                "出力は32ピクセル単位に丸めます。上記のサイズで編集します。")
    except Exception as e:
        return str(e)


def finish_failed_task(task_id):
    if task_id:
        from modules import progress as forge_progress
        forge_progress.finish_task(task_id)
        forge_progress.pending_tasks.pop(task_id, None)


def run_edit(source, references, scale, prompt, dit, te, vae, seed, count, reserve, tile, keep, retry, tokens,
             progress=gr.Progress(track_tqdm=False), auto_translate=False, llm_model="", plamo_model="", lm_base="http://127.0.0.1:1234", save_json=False, task_id=""):
    from .engine import ENGINE
    from .editing import open_source, scaled_size
    try:
        if not source:
            raise PH3Error("編集する画像を選択してください。")
        width, height = scaled_size(*open_source(source).size, float(scale))
        req = replace(make_request(prompt, dit, te, vae, width, height, seed, count, reserve, tile, keep, retry, tokens),
                      source_image=str(source), image_scale=float(scale),
                      reference_images=tuple(str(path) for path in (references or [])), save_json=bool(save_json))
        req.checked()
    except Exception as e:
        finish_failed_task(task_id)
        return [], str(e), None
    try:
        paths = ENGINE.generate(req, output_root(), progress=lambda value, text: progress(value, desc=text),
                                translate=auto_translate, task_id=task_id, llm_model=llm_model, plamo_model=plamo_model, lm_base=lm_base)
        status = f"編集完了: {len(paths)}枚・{width}×{height}\n保存先: {output_root()}"
        if not ENGINE.cleanup_ok:
            status += "\nメモリ解放でエラーが発生しました。ログを確認してください。"
        return paths, status, str(ENGINE.log_path) if ENGINE.log_path else None
    except Exception as e:
        if not isinstance(e, Cancelled):
            traceback.print_exc()
        finish_failed_task(task_id)
        return list(ENGINE.last_outputs), str(e), str(ENGINE.log_path) if ENGINE.log_path else None


def run_reference(images, prompt, width, height, dit, te, vae, seed, count, reserve, tile, keep, retry, tokens,
                  progress=gr.Progress(track_tqdm=False), auto_translate=False, llm_model="", plamo_model="", lm_base="http://127.0.0.1:1234", save_json=False, task_id=""):
    from .engine import ENGINE
    try:
        req = replace(make_request(prompt, dit, te, vae, width, height, seed, count, reserve, tile, keep, retry, tokens),
                      mode="reference", reference_images=tuple(str(path) for path in (images or [])), save_json=bool(save_json))
        req.checked()
    except Exception as e:
        finish_failed_task(task_id)
        return [], str(e), None
    try:
        paths = ENGINE.generate(req, output_root(), progress=lambda value, text: progress(value, desc=text),
                                translate=auto_translate, task_id=task_id, llm_model=llm_model, plamo_model=plamo_model, lm_base=lm_base)
        status = f"作成完了: {len(paths)}枚・{req.width}×{req.height}\n保存先: {output_root()}"
        if not ENGINE.cleanup_ok:
            status += "\nメモリ解放でエラーが発生しました。ログを確認してください。"
        return paths, status, str(ENGINE.log_path) if ENGINE.log_path else None
    except Exception as e:
        if not isinstance(e, Cancelled):
            traceback.print_exc()
        finish_failed_task(task_id)
        return list(ENGINE.last_outputs), str(e), str(ENGINE.log_path) if ENGINE.log_path else None


CANVAS_PRESETS = {"4MP 正方形 2048×2048": (2048, 2048), "4MP 縦長 1664×2496": (1664, 2496),
                  "4MP 横長 2496×1664": (2496, 1664), "8MP 正方形 2912×2912": (2912, 2912),
                  "8MP 縦長 2368×3552": (2368, 3552), "8MP 横長 3552×2368": (3552, 2368)}


def apply_canvas_preset(preset):
    return CANVAS_PRESETS[preset]


def save_edit_settings(scale, dit, te, vae, seed, count, reserve, tile, keep, retry, tokens,
                       compose_width=2048, compose_height=2048, prompt_mode="そのまま使用", llm_model="", plamo_model="", lm_base="http://127.0.0.1:1234", save_json=False):
    try:
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise PH3Error("倍率は0より大きい有限の数値にしてください。")
        req = make_request("設定保存用", dit, te, vae, 2048, 2048, seed, count, reserve, tile, keep, retry, tokens)
        req.checked(check_files=False)
        data = {k: v for k, v in req.as_dict().items() if k not in ("prompt", "source_image", "reference_images", "width", "height")}
        data["image_scale"] = float(scale)
        canvas = replace(req, width=_integer(compose_width, "幅"), height=_integer(compose_height, "高さ"))
        canvas.checked(check_files=False)
        mode_key(prompt_mode)
        data.update(compose_width=canvas.width, compose_height=canvas.height,
                    prompt_mode=prompt_mode, llm_model=llm_model, plamo_model=plamo_model,
                    lm_base=lm_base, save_json=bool(save_json))
        tmp = CONFIG.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(CONFIG)
        return "編集設定を保存しました。"
    except Exception as e:
        return "設定を保存できませんでした: " + str(e)


def refresh_local_models(base, selected_llm, selected_plamo):
    try:
        available = models(base)
        all_keys = [m["key"] for m in available]
        plamo_keys = [key for key in all_keys if "plamo-3-translate" in key.lower()]
        llm_keys = [key for key in all_keys if key not in plamo_keys]
        choices = [(m.get("display_name") or m["key"], m["key"]) for m in available if m["key"] in llm_keys]
        return (gr.update(choices=choices, value=selected_llm if selected_llm in llm_keys else (llm_keys[0] if llm_keys else None)),
                gr.update(choices=plamo_keys, value=selected_plamo if selected_plamo in plamo_keys else (plamo_keys[0] if plamo_keys else None)),
                f"ローカルモデル {len(all_keys)}件を確認しました。")
    except Exception as e:
        return gr.skip(), gr.skip(), "LM Studioのモデルを取得できません: " + str(e)


def preview_translation(text, mode, llm_model, plamo_model, base, source=None, references=None):
    from .engine import ENGINE
    from modules.call_queue import queue_lock
    if not ENGINE.job_guard.acquire(blocking=False):
        return "P-H3の処理中です。終了後に確認してください。"
    locked = False
    try:
        locked = queue_lock.acquire(blocking=False)
        if not locked: return "Forgeの生成中です。終了後に確認してください。"
        ENGINE.cancel_event.clear()
        def check():
            if ENGINE.cancel_event.is_set(): raise Cancelled("指示文の変換を中止しました。")
        if mode_key(mode) in ("llm", "plamo", "official"): ENGINE.unload_gpu()
        return process_prompt(text, mode, llm_model, plamo_model, base, check=check,
                              image_paths=tuple(p for p in (source, *(references or [])) if p), composition=not bool(source))
    except Exception as e:
        return str(e)
    finally:
        if locked: queue_lock.release()
        ENGINE.job_guard.release()


def append_reference_files(existing, added, limit):
    paths = list(existing or [])
    for path in added or []:
        if str(path) not in {str(p) for p in paths}:
            paths.append(path)
    if len(paths) > limit:
        raise gr.Error(f"参照画像は{limit}枚までです。追加前の一覧は維持されます。")
    return paths


def append_edit_references(existing, added):
    return append_reference_files(existing, added, 7)


def append_composition_references(existing, added):
    return append_reference_files(existing, added, 8)


def preview_composition(text, mode, llm_model, plamo_model, base, references=None):
    return preview_translation(text, mode, llm_model, plamo_model, base, references=references)


def quality_scale(source, megapixels):
    if not source:
        raise gr.Error("先にメイン画像を選択してください。")
    from .editing import open_source
    image = open_source(source)
    area = 2048**2 if megapixels == 4 else 2912**2
    return math.sqrt(area / (image.width * image.height))


def apply_edit_4mp(source):
    return quality_scale(source, 4)


def apply_edit_8mp(source):
    return quality_scale(source, 8)


def generated_folder():
    from .engine import ENGINE
    for name in reversed(ENGINE.last_outputs):
        path = Path(name)
        if path.is_file(): return path.resolve().parent
    root = output_root().resolve()
    latest = max(root.glob("????-??-??/*.png"), key=lambda p: p.stat().st_mtime, default=None)
    folder = latest.parent if latest else root
    if not folder.is_dir():
        raise PH3Error("保存フォルダーはまだありません。先に画像を生成してください。")
    return folder


def open_generated_folder():
    import os
    try:
        folder = generated_folder()
        os.startfile(str(folder), "open")
        return "保存フォルダーを開きました: " + str(folder)
    except Exception as e:
        return "保存フォルダーを開けませんでした: " + str(e)


def build_tab():
    cfg = settings()
    try:
        inventory = models(cfg["lm_base"])
        llm_choices = [(m.get("display_name") or m["key"], m["key"]) for m in inventory if "plamo-3-translate" not in m["key"]]
    except Exception:
        # Offline startup still shows the saved choice; the refresh button reports connection errors.
        llm_choices = []
    if cfg["llm_model"] and cfg["llm_model"] not in [key for _, key in llm_choices]:
        llm_choices.append((cfg["llm_model"], cfg["llm_model"]))
    with gr.Blocks(analytics_enabled=False, elem_id="ph3_editor") as tab:
        gr.Markdown("## P-H3 画像編集")
        with gr.Row(equal_height=False, elem_id="ph3_workspace"):
            with gr.Column(scale=5, min_width=420, elem_id="ph3_inputs"):
                auto_translate = gr.Dropdown(label="日本語の指示をどう処理するか", choices=list(MODES), value=cfg["prompt_mode"])
                with gr.Row():
                    llm_model = gr.Dropdown(label="LM Studioモードで使用するLLM", choices=llm_choices, value=cfg["llm_model"] or None, allow_custom_value=False, scale=4)
                    refresh_models = gr.Button("モデル一覧を更新", scale=1)
                with gr.Tabs():
                    with gr.Tab("メイン画像を編集"):
                        with gr.Column():
                            source = EditImage(label="画像1：編集するメイン画像", type="filepath", sources=["upload", "clipboard"], height=360)
                            with gr.Column():
                                prompt = gr.Textbox(label="編集指示", lines=6, placeholder="例：テーブルの上に白い花を入れた花瓶を追加し、それ以外はそのままにしてください。")
                                scale = gr.Number(label="画像サイズの倍率", value=cfg.get("image_scale", 1.0), minimum=.01)
                                size_preview = gr.Markdown("画像を選択すると出力サイズを表示します。")
                                with gr.Row():
                                    quality4 = gr.Button("4MPの倍率に設定")
                                    quality8 = gr.Button("8MPの倍率に設定")
                        references = gr.File(label="画像2〜8：追加の参照画像（任意・表示順に番号を割り当て）",
                                             file_count="multiple", file_types=["image"], type="filepath", interactive=True)
                        add_edit_references = gr.UploadButton("参照画像を追加（メイン画像とは別・最大7枚）",
                                                               file_count="multiple", file_types=["image"], type="filepath")
                        gr.Markdown("1枚ずつ追加するときは上の追加ボタンを使います。ファイル選択画面ではCtrl／Shiftで複数選択もできます。一覧の順番が画像番号です。")
                        gr.Markdown("指示例：画像1の背景を画像2の背景に変更。出力サイズはメイン画像×倍率で決まります。")
                        generate = gr.Button("画像を編集", variant="primary")
                    with gr.Tab("参照画像から構図を作成"):
                        compose_images = gr.File(label="参照画像1〜8（表示順に番号を割り当て）",
                                                 file_count="multiple", file_types=["image"], type="filepath", interactive=True)
                        add_compose_images = gr.UploadButton("参照画像を追加（最大8枚）",
                                                              file_count="multiple", file_types=["image"], type="filepath")
                        gr.Markdown("追加ボタンから1枚ずつ追加できます。Ctrl／Shiftで複数選択も可能です。一覧の順番が画像番号です。")
                        compose_prompt = gr.Textbox(label="作成する場面・人物の配置・動作", lines=6,
                                                    placeholder="例：画像1・2・3のキャラクター全員が並び、肩を組んでカメラにピース。元の衣装と髪色を保つ。")
                        with gr.Row():
                            compose_width = gr.Number(label="出力の幅（32の倍数）", value=cfg["compose_width"], precision=0)
                            compose_height = gr.Number(label="出力の高さ（32の倍数）", value=cfg["compose_height"], precision=0)
                        with gr.Row():
                            preset = gr.Dropdown(label="サイズの候補", choices=list(CANVAS_PRESETS), value=next(iter(CANVAS_PRESETS)))
                            apply_preset = gr.Button("このサイズを適用")
                        gr.Markdown("画像は人物や衣装などの参照として使います。出力サイズは上の幅・高さで決まります。")
                        compose_generate = gr.Button("参照から作成", variant="primary")
                with gr.Row():
                    seed = gr.Textbox(label="Seed（-1でランダム）", value=str(cfg["seed"]))
                    count = gr.Slider(label="生成枚数", minimum=1, maximum=16, step=1, value=cfg["count"])
                with gr.Accordion("モデルのパス", open=False):
                    dit = gr.Textbox(label="P-H3画像生成モデル（INT8）", value=cfg["dit"])
                    te = gr.Textbox(label="P-H3専用テキストエンコーダー: int8_convrot推奨", value=cfg["text_encoder"])
                    vae = gr.Textbox(label="P-H3専用静止画VAE: int8_convrot推奨", value=cfg["vae"])
                    diagnose_btn = gr.Button("環境・モデルの診断（生成せず確認）")
                with gr.Accordion("メモリ管理", open=False):
                    reserve = gr.Slider(label="DiTの作業用VRAM目安 (GiB)。使用量の上限を保証する設定ではありません", minimum=3, maximum=12, step=.5, value=cfg["reserve_gib"])
                    tile = gr.Dropdown(label="VAEタイルサイズ", choices=[128, 256, 512], value=cfg["tile_size"])
                    tokens = gr.Slider(label="プロンプトの最大トークン数（超過時は停止）", minimum=64, maximum=2048, step=64, value=cfg["max_tokens"])
                    keep = gr.Checkbox(label="生成後もCPUモデル・文章キャッシュを保持する", value=cfg["keep_cpu_cache"])
                    retry = gr.Checkbox(label="CUDA OOM時、同じSeed・解像度で一度だけ省メモリ設定を強めて再試行", value=cfg["retry_oom"])
                with gr.Row():
                    cancel = gr.Button("停止")
                    release = gr.Button("GPU・CPUキャッシュを解放")
                    save = gr.Button("この拡張の設定を保存")
                with gr.Accordion("翻訳・LLMの設定と変換結果", open=False):
                    lm_base = gr.Textbox(label="LM StudioのローカルAPI", value=cfg["lm_base"])
                    plamo_model = gr.Dropdown(label="PLaMo 3 Translateモデル（CPUで翻訳）", choices=[cfg["plamo_model"]] if cfg["plamo_model"] else [], value=cfg["plamo_model"] or None, allow_custom_value=True)
                    with gr.Row():
                        preview_edit = gr.Button("編集指示の変換結果を確認")
                        preview_reference = gr.Button("構図の指示の変換結果を確認")
                    translated = gr.Textbox(label="英文のプレビュー", lines=5, interactive=False)
                save_json = gr.Checkbox(label="生成条件のJSONを画像と一緒に保存", value=cfg["save_json"])
                task_id = gr.Textbox(value="", visible=False)
            with gr.Column(scale=6, min_width=460, elem_id="ph3_results"):
                gr.HTML('<div id="ph3_activity" hidden role="status" aria-live="polite" aria-busy="false">'
                        '<div class="ph3_activity_line"><span class="ph3_spinner" aria-hidden="true"></span>'
                        '<span data-ph3-text>処理を開始しています…</span></div>'
                        '<div class="ph3_track" role="progressbar" aria-label="生成の進捗" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">'
                        '<div data-ph3-bar></div></div></div>')
                gallery = gr.Gallery(label="生成結果", columns=1, height=780, object_fit="contain", preview=True, selected_index=0, elem_id="ph3_gallery")
                open_folder = gr.Button("保存フォルダーをエクスプローラーで開く")
                status = gr.Textbox(label="状態", value="待機中", lines=2, max_lines=4, interactive=False, elem_id="ph3_status")
                with gr.Accordion("詳細ログ", open=False):
                    log_file = gr.File(label="診断・実行ログをダウンロード", interactive=False)
        add_edit_references.upload(append_edit_references, inputs=[references, add_edit_references], outputs=references, concurrency_limit=1)
        add_compose_images.upload(append_composition_references, inputs=[compose_images, add_compose_images], outputs=compose_images, concurrency_limit=1)
        preview_edit.click(preview_translation, inputs=[prompt, auto_translate, llm_model, plamo_model, lm_base, source, references], outputs=translated)
        preview_reference.click(preview_composition, inputs=[compose_prompt, auto_translate, llm_model, plamo_model, lm_base, compose_images], outputs=translated, api_name="preview_translation_1")
        inputs = [source, references, scale, prompt, dit, te, vae, seed, count, reserve, tile, keep, retry, tokens, auto_translate, llm_model, plamo_model, lm_base, save_json, task_id]
        generate.click(run_edit, inputs=inputs, outputs=[gallery, status, log_file], concurrency_limit=1, js="ph3Submit", show_progress="minimal").then(
            fn=None, inputs=[], outputs=[], js="ph3Finish", queue=False, api_name=False)
        compose_generate.click(run_reference,
                               inputs=[compose_images, compose_prompt, compose_width, compose_height, dit, te, vae,
                                       seed, count, reserve, tile, keep, retry, tokens, auto_translate, llm_model, plamo_model, lm_base, save_json, task_id],
                               outputs=[gallery, status, log_file], concurrency_limit=1, js="ph3Submit", show_progress="minimal").then(
                                   fn=None, inputs=[], outputs=[], js="ph3Finish", queue=False, api_name=False)
        refresh_models.click(refresh_local_models, inputs=[lm_base, llm_model, plamo_model], outputs=[llm_model, plamo_model, status], queue=False)
        apply_preset.click(apply_canvas_preset, inputs=preset, outputs=[compose_width, compose_height], queue=False)
        source.change(edit_size_preview, inputs=[source, scale], outputs=size_preview, queue=False)
        scale.change(edit_size_preview, inputs=[source, scale], outputs=size_preview, queue=False)
        quality4.click(apply_edit_4mp, inputs=source, outputs=scale, queue=False)
        quality8.click(apply_edit_8mp, inputs=source, outputs=scale, queue=False)
        open_folder.click(open_generated_folder, inputs=[], outputs=status, queue=False)
        cancel.click(stop, inputs=[], outputs=status, queue=False)
        release.click(unload, inputs=[], outputs=status, queue=False)
        save.click(save_edit_settings, inputs=[scale, dit, te, vae, seed, count, reserve, tile, keep, retry, tokens, compose_width, compose_height, auto_translate, llm_model, plamo_model, lm_base, save_json], outputs=status, queue=False)
        diagnose_btn.click(check_environment, inputs=[dit, te, vae], outputs=[status, log_file], queue=False)
    return [(tab, "P-H3 画像編集", "ph3_native_image")]
