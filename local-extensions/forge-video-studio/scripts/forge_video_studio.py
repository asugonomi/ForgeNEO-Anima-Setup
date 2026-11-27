"""Native Forge H3 tab. Uses Forge's loader, GPU queue and sampler in-process."""
import json
import sys
import time
import uuid
import threading
import copy
import gc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)) if str(ROOT) not in sys.path else None
import gradio as gr
from PIL import Image, ImageOps
from modules import script_callbacks, scripts, shared, sd_models, processing
from modules.call_queue import queue_lock
from modules_forge import main_thread
from video_dimensions import AUTO, dimensions, describe
from forge_h3 import integration
from forge_h3.native import compat
ENHANCER_ROOT = ROOT.parent / 'forge-prompt-enhancer'
if str(ENHANCER_ROOT) not in sys.path:
    sys.path.insert(0, str(ENHANCER_ROOT))
try:
    import forge_prompt_enhancer as llm_optimizer
except ImportError:
    llm_optimizer = None

PROBLEMS = compat.check()
if not PROBLEMS:
    from forge_h3.native import patches
    patches.apply()
else:
    compat.report(PROBLEMS)

MODEL_ROOT = Path('E:/EasyForgeNeo/Model')
OUTPUT = Path('G:/Stable_diffusion_outputs/video/ForgeVideoStudio')
DIT = MODEL_ROOT / 'DiffusionModels/minimax_h3_fl2va_pruned_int8_convrot.safetensors'
TE = MODEL_ROOT / 'TextEncoders/qwen3vl_32b_minimax_h3-Q2_K_M.gguf'
VAE = MODEL_ROOT / 'VAE/minimax_h3_video_vae_int8_convrot.safetensors'
AUDIO = MODEL_ROOT / 'VAE/minimax_h3_audio_vae_fp32.safetensors'
TURBO = 'minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16'
ORBIT = 'minimax_h3_flf2v_lora_v1'
ORBIT_PROMPT = ('One frozen instant. Only the camera moves in a continuous 360 degree orbit. '
    'Preserve every person and object in the same world position, orientation, shape and pose. '
    'Keep faces, hands, clothing and background motionless. Camera parallax is the only apparent movement. '
    'Return smoothly to the original view. No cuts, zoom, morphing or added objects.')
_job_lock = threading.Lock()
_owner = None

def on_main(fn):
    def caught():
        try:
            return True, fn()
        except Exception as e:
            import traceback
            traceback.print_exc()
            (ROOT / 'last_error.log').write_text(traceback.format_exc(), encoding='utf-8')
            # The Forge worker retains its last Task. Never give it an exception
            # with traceback frames that keep failed inference tensors alive.
            message = f'{type(e).__name__}: {e}'
            traceback.clear_frames(e.__traceback__)
        gc.collect()
        from backend import memory_management
        memory_management.soft_empty_cache()
        return False, message
    result = main_thread.run_and_wait_result(caught)
    if not result or not result[0]:
        raise RuntimeError(str(result[1]) if result else 'Forge推論スレッドでエラーが発生しました。CMDを確認してください。')
    return result[1]

class VideoRunner(scripts.ScriptRunner):
    def __init__(self, last, audio, shift):
        super().__init__()
        self.last = last
        self.audio = audio
        self.shift = shift

    def before_process(self, p):
        integration.before_process(p, 'Video', self.audio, 3.0)
        if self.last is not None:
            p.h3_last_frame = self.last
            p.h3_request.last_frame = True

    def process(self, p):
        integration.process(p)
        p.distilled_cfg_scale = self.shift

    def process_before_every_sampling(self, p, **kwargs):
        integration.before_sampling(p, kwargs['noise'])

    def post_sample(self, p, ps):
        integration.after_sampling(p)

    def postprocess(self, p, result):
        integration.postprocess(p, result)

def generate_native(image, last, mode, prompt, size, frames, seed, audio, scale=1.0):
    if PROBLEMS:
        raise RuntimeError('\n'.join(PROBLEMS))
    for path in [DIT, TE, VAE, AUDIO, MODEL_ROOT / f'Lora/{TURBO}.safetensors']:
        if not path.is_file():
            raise ValueError(f'モデルがありません: {path.name}')
    if mode != 'テキスト→動画' and image is None:
        raise ValueError('開始画像を指定してください。')
    if not prompt.strip() and mode != '360度回転':
        raise ValueError('動画プロンプトを入力してください。')
    width, height = dimensions(image, mode, size, scale)
    print(f'[Forge Video Studio] {mode}: {width}x{height}, {frames} frames, scale={scale}', flush=True)
    frames = int(frames)
    if frames not in (73, 124, 192):
        raise ValueError('未対応のフレーム数です。')
    if int(seed) != seed or not -1 <= seed <= 4294967295:
        raise ValueError('Seedは-1または0〜4294967295です。')
    prepare = lambda im: ImageOps.pad(ImageOps.exif_transpose(im).convert('RGB'), (width, height),
                                     method=Image.Resampling.LANCZOS, color=(230,230,230)) if im is not None else None
    image, last = prepare(image), prepare(last)
    if mode == '360度回転':
        last = image
        prompt = ORBIT_PROMPT + ('\n' + prompt.strip() if prompt.strip() else '')
        prompt += f' <lora:{ORBIT}:1>'
    prompt += f' <lora:{TURBO}:1>'
    # Free the image model before constructing/loading the video model.
    sd_models.unload_model_weights()
    from modules_forge import main_entry
    for component in (TE, VAE, AUDIO):
        main_entry.module_list[component.name] = str(component)
    info = sd_models.get_closet_checkpoint_match(DIT.name)
    if info is None:
        info = sd_models.CheckpointInfo(str(DIT))
        info.register()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    overrides = {'sd_model_checkpoint': info.title, 'forge_additional_modules': [str(TE), str(VAE), str(AUDIO)]}
    cls = processing.StableDiffusionProcessingTxt2Img if mode == 'テキスト→動画' else processing.StableDiffusionProcessingImg2Img
    args = dict(prompt=prompt, negative_prompt='', seed=int(seed), steps=4, cfg_scale=1.0,
                distilled_cfg_scale=6.0, sampler_name='Euler', scheduler='Simple', width=width, height=height,
                batch_size=frames, n_iter=1, outpath_samples=str(OUTPUT), outpath_grids=str(OUTPUT),
                do_not_save_samples=True, do_not_save_grid=True, override_settings=overrides)
    if cls is processing.StableDiffusionProcessingImg2Img:
        args.update(init_images=[image], denoising_strength=1.0, resize_mode=0)
    p = cls(**args)
    p.is_api = True
    p.scripts = VideoRunner(last, audio, 6.0)
    p.script_args = (0,)
    previous_options = {k:copy.deepcopy(getattr(shared.opts,k)) for k in ('sd_model_checkpoint','forge_additional_modules')}
    try:
        from backend import memory_management
        previous_reserve = memory_management.EXTRA_RESERVED_VRAM
        previous_pin = memory_management.pin_memory
        # H3 file views must not be host-registered: registration/unregistration
        # of overlapping mmap pages can invalidate subsequent CUDA transfers.
        memory_management.pin_memory = lambda tensor: False
        memory_management.EXTRA_RESERVED_VRAM = max(previous_reserve, 4 * 1024**3)
        result = processing.process_images(p)
        if shared.state.interrupted:
            return None
        path = getattr(result, 'video_path', None)
        if not path or not Path(path).is_file():
            raise RuntimeError('動画が保存されませんでした。CMDのH3エラーを確認してください。')
        return str(path)
    finally:
        memory_management.EXTRA_RESERVED_VRAM = previous_reserve
        memory_management.pin_memory = previous_pin
        p.close()
        p.clear_prompt_cache()
        shared.state.current_latent = None
        shared.state.current_image = None
        if hasattr(p, 'init_latent'):
            p.init_latent = None
        patches.end_sampling()
        from forge_h3.contracts import set_pending_error
        set_pending_error(None)
        sd_models.unload_model_weights()
        # Also restore persisted quicksettings if a UI refresh saved the temporary H3 selection.
        from modules.sysinfo import set_config
        set_config(previous_options, save_config=True)

def generate(image, last, mode, prompt, size, frames, seed, audio, owner, scale=1.0, progress=gr.Progress()):
    global _owner
    if not _job_lock.acquire(False):
        raise gr.Error('動画タブで別の処理が実行中です。')
    try:
        progress(0, desc='Forgeの画像生成終了を待っています')
        with queue_lock:
            _owner = owner
            shared.state.begin('Forge Video Studio')
            start = time.monotonic()
            try:
                progress(0.05, desc='画像モデルを解放して動画を生成しています。詳細はCMDに表示します')
                path = on_main(lambda: generate_native(image, last, mode, prompt, size, frames, seed, audio, scale))
                return path, f'完了 · {time.monotonic()-start:.1f}秒 · {path}' if path else '停止しました。'
            finally:
                _owner = None
                shared.state.end()
    except Exception as e:
        raise gr.Error(str(e))
    finally:
        _job_lock.release()

def stop(owner):
    if _owner == owner:
        shared.state.interrupt()
        return '停止要求を送信しました。'
    return 'この画面で実行中の動画はありません。'

def refresh_models():
    try:
        if llm_optimizer is None:
            raise RuntimeError('Forge Prompt Enhancer拡張を導入してください。')
        found = llm_optimizer.models()
        names = [m['key'] for m in found]
        default = next((n for n in names if 'gemma4-12b' in n), names[0] if names else None)
        return gr.update(choices=names, value=default), 'LM Studio接続済み'
    except Exception as e:
        return gr.update(choices=[], value=None), 'LM Studioのローカルサーバーを起動してください: ' + str(e)

def optimize(image, intent, mode, frames, model, progress=gr.Progress()):
    if llm_optimizer is None:
        raise gr.Error('Forge Prompt Enhancer拡張を導入してください。')
    if not _job_lock.acquire(False):
        raise gr.Error('動画タブで別の処理が実行中です。')
    try:
        with queue_lock:
            on_main(sd_models.unload_model_weights)
            plan, prompt = llm_optimizer.optimize(image, intent, mode, int(frames)/24, model,
                                                 lambda s: progress(0.5, desc=s))
            return plan, prompt, '最適化完了。プロンプトを確認・編集してから生成できます。LLMは解放済み。'
    except Exception as e:
        raise gr.Error(str(e))
    finally:
        _job_lock.release()

def tab():
    with gr.Blocks() as ui:
        gr.Markdown('## 動画生成 · MiniMax H3\nForge Neo単体で実行。生成前に画像モデル、生成後に動画モデルをメモリから解放します。')
        owner = gr.State(lambda: uuid.uuid4().hex)
        with gr.Row():
            with gr.Column():
                mode = gr.Radio(['画像→動画', '360度回転', 'テキスト→動画'], value='画像→動画', label='生成モード')
                with gr.Row():
                    image = gr.Image(type='pil', label='開始画像', sources=['upload', 'clipboard'], height=310)
                    last = gr.Image(type='pil', label='終了画像（任意・360度は開始画像を使用）', height=310)
                prompt = gr.Textbox(label='動画プロンプト（英語推奨・編集可能）', lines=5,
                    placeholder='The character gently turns her head as her hair moves in a light breeze. The camera remains fixed.')
                with gr.Row():
                    size = gr.Dropdown([AUTO, '768 × 768', '512 × 512', '768 × 512', '512 × 768'], value=AUTO, label='解像度の基準')
                    frames = gr.Dropdown([('約3秒 / 73 frames',73), ('約5秒 / 124 frames',124), ('8秒 / 192 frames',192)], value=73, label='長さ')
                    seed = gr.Number(value=-1, precision=0, label='Seed（-1でランダム）')
                scale = gr.Slider(0.125, 2.0, value=1.0, step=0.125, label='入力画像の縮尺（1 = 1×、0.5 = 半分）')
                size_info = gr.Textbox(value='開始画像を指定してください。', label='実際の生成サイズ', interactive=False)
                audio = gr.Checkbox(False, label='音声も生成する（追加時間が必要）')
                gr.Markdown('4ステップ高速版。入力画像×縮尺を32px単位に調整します。固定解像度では縮尺を適用せず、画像は縦横比を保ち余白を付けます。360度では見えない背面をAIが補完します。')
                with gr.Row():
                    run = gr.Button('動画を生成', variant='primary')
                    cancel = gr.Button('この動画を停止')
            with gr.Column():
                video = gr.Video(label='生成動画', interactive=False)
                status = gr.Textbox(label='状態', interactive=False)
                with gr.Accordion('LM Studioでプロンプト最適化', open=True):
                    intent = gr.Textbox(label='希望する動き・カメラ（日本語可）', lines=3)
                    gemma = 'gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m'
                    model = gr.Dropdown([gemma], value=gemma, label='LM Studioモデル')
                    refresh = gr.Button('LLM一覧を更新')
                    enhance = gr.Button('画像と指示から二段階で最適化')
                    plan = gr.Textbox(label='1段階目：ショット構成', lines=5, interactive=False)
                    gr.Markdown('画像対応LLMが入力を解析し、2段階目で英文に整えます。最適化後は使用したLLMを解放します。')
        for control in (image, mode, size, scale):
            control.change(describe, inputs=[image,mode,size,scale], outputs=[size_info], queue=False, api_name=False)
        refresh.click(refresh_models, outputs=[model,status], api_name='forge_video_models')
        enhance.click(optimize, inputs=[image,intent,mode,frames,model], outputs=[plan,prompt,status], api_name='forge_video_optimize')
        run.click(generate, inputs=[image,last,mode,prompt,size,frames,seed,audio,owner,scale], outputs=[video,status], api_name='forge_video_generate')
        cancel.click(stop, inputs=[owner], outputs=[status], queue=False, api_name=False)
    return [(ui, '動画生成', 'forge_video_studio')]

script_callbacks.on_ui_tabs(tab)
