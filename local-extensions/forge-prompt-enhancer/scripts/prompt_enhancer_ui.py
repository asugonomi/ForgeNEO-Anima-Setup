import sys
import threading
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import gradio as gr
from modules import script_callbacks, scripts, sd_models
from modules.call_queue import queue_lock
from modules_forge import main_thread
import forge_prompt_enhancer as core
import importlib
core = importlib.reload(core)

GEMMA = 'gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m'
prompts = {}
lock = threading.Lock()

def capture(component, **kwargs):
    name = getattr(component, 'elem_id', '')
    if name in ('txt2img_prompt', 'img2img_prompt'):
        prompts[name] = component

def enhance(prompt, system, model, progress=gr.Progress()):
    if not lock.acquire(False):
        raise gr.Error('プロンプト最適化を実行中です。')
    try:
        progress(0, desc='画像・動画生成の終了を待っています')
        with queue_lock:
            main_thread.run_and_wait_result(sd_models.unload_model_weights)
            result = core.enhance_image(prompt, system, model, lambda s: progress(.5, desc=s))
        return result, prompt, '完了。元の文は「元に戻す」で復元できます。LLMは解放済み。'
    except Exception as e:
        raise gr.Error(str(e))
    finally:
        lock.release()

class Script(scripts.Script):
    def title(self):
        return 'Prompt Enhancer · Gemma'

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        key = 'img2img_prompt' if is_img2img else 'txt2img_prompt'
        with gr.Accordion('Prompt Enhancer · Gemma 4 12B', open=False):
            model = gr.Dropdown([GEMMA], value=GEMMA, label='最適化モデル')
            system = gr.Textbox(value=core.DEFAULT_IMAGE_SYSTEM, label='システムプロンプト（編集可）', lines=6)
            with gr.Row():
                button = gr.Button('ポジティブプロンプトを強化')
                undo = gr.Button('元に戻す')
            original = gr.State('')
            status = gr.Textbox(label='最適化の状態', interactive=False)
            gr.Markdown('現在のポジティブ欄を強化します。ネガティブ欄は変更しません。実行前に画像モデルを解放します。')
            prompt = prompts.get(key)
            if prompt is None:
                raise RuntimeError('Prompt Enhancer could not find '+key)
            button.click(enhance, inputs=[prompt,system,model], outputs=[prompt,original,status], api_name=key+'_enhance')
            undo.click(lambda old: gr.update(value=old) if old else gr.update(), inputs=[original], outputs=[prompt], api_name=False)
        # No processing arguments: this extension never runs during image generation.
        return []

def reset():
    prompts.clear()

script_callbacks.on_before_ui(reset)
script_callbacks.on_after_component(capture)
