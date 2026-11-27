from pathlib import Path
from types import SimpleNamespace
import os
from PIL import Image
from ph3_native import ui, engine
from ph3_native.editing import scaled_size


def test_gemma_default_and_explicit_saved_selection(tmp_path,monkeypatch):
    import json
    config=tmp_path/'config.json'
    monkeypatch.setattr(ui,'CONFIG',config)
    initial=ui.settings()
    assert initial['prompt_mode']=='LM Studioで英文プロンプト作成'
    assert initial['llm_model']=='gemma4-12b-qat-uncensored-hauhaucs-balanced@q4_k_m'
    config.write_text(json.dumps({'llm_model':'another-model','prompt_mode':'そのまま使用'}),encoding='utf-8')
    selected=ui.settings()
    assert selected['llm_model']=='another-model'
    assert selected['prompt_mode']=='そのまま使用'


def test_explicit_quality_buttons_keep_aspect(tmp_path):
    path=tmp_path/'source.png';Image.new('RGB',(1280,1920)).save(path)
    assert scaled_size(1280,1920,ui.apply_edit_4mp(str(path)))==(1664,2496)
    assert scaled_size(1280,1920,ui.apply_edit_8mp(str(path)))==(2368,3552)


def test_open_actual_generation_folder_after_root_change(tmp_path,monkeypatch):
    old=tmp_path/'old'/'2026-11-06';old.mkdir(parents=True)
    output=old/'image.png';output.write_bytes(b'placeholder')
    monkeypatch.setattr(engine,'ENGINE',SimpleNamespace(last_outputs=[str(output)]))
    monkeypatch.setattr(ui,'output_root',lambda:tmp_path/'new')
    called=[];monkeypatch.setattr(os,'startfile',lambda path,verb:called.append((path,verb)))
    assert '開きました' in ui.open_generated_folder()
    assert called==[(str(old.resolve()),'open')]


def test_folder_fallback_after_restart(tmp_path,monkeypatch):
    directory=tmp_path/'2026-11-06';directory.mkdir();(directory/'1.png').write_bytes(b'x')
    monkeypatch.setattr(engine,'ENGINE',SimpleNamespace(last_outputs=[]))
    monkeypatch.setattr(ui,'output_root',lambda:tmp_path)
    assert ui.generated_folder()==directory.resolve()


def test_llm_choices_have_names_and_do_not_include_translator(monkeypatch):
    monkeypatch.setattr(ui,'models',lambda base:[
        {'key':'qwen','display_name':'Qwen'}, {'key':'gemma','display_name':'Gemma'},
        {'key':'plamo-3-translate-0.8b','display_name':'PLaMo'}])
    llm,plamo,status=ui.refresh_local_models('http://localhost:1234','gemma','')
    assert llm['choices']==[('Qwen','qwen'),('Gemma','gemma')]
    assert llm['value']=='gemma' and plamo['value']=='plamo-3-translate-0.8b'
