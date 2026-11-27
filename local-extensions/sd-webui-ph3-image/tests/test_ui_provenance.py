import hashlib
import inspect
import json
import pytest
from ph3_native.contracts import PH3Error, UPSTREAM_COMMIT
from ph3_native import provenance


def test_vendor_checksum_and_tampering(tmp_path,monkeypatch):
    p=tmp_path/'forge_h3/native/transformer.py';p.parent.mkdir(parents=True);p.write_text('# test')
    manifest={'commit':UPSTREAM_COMMIT,'sha256':{'forge_h3/native/transformer.py':hashlib.sha256(p.read_bytes()).hexdigest()}}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(provenance,'VENDOR',tmp_path)
    assert provenance.verify_vendor()['commit']==UPSTREAM_COMMIT
    p.write_text('# changed')
    with pytest.raises(PH3Error):provenance.verify_vendor()


def test_vendor_missing(tmp_path,monkeypatch):
    monkeypatch.setattr(provenance,'VENDOR',tmp_path)
    with pytest.raises(PH3Error):provenance.verify_vendor()


def test_vendor_path_traversal(tmp_path,monkeypatch):
    d=tmp_path/'vendor';d.mkdir();p=tmp_path/'outside';p.write_text('outside')
    (d/'manifest.json').write_text(json.dumps({'commit':UPSTREAM_COMMIT,'sha256':{'forge_h3/native/transformer.py':'wrong','../outside':'wrong'}}))
    monkeypatch.setattr(provenance,'VENDOR',d)
    with pytest.raises(PH3Error):provenance.verify_vendor()


def values():
    return ['', 'd.safetensors','t.safetensors','v.safetensors',2048,2048,'-1',1,6,256,True,True,1024]


def test_ui_integer_validation_and_full_seed():
    from ph3_native.ui import _integer
    assert _integer(str(2**63-1),'seed')==2**63-1
    for value in [1.5,float('nan'),'1.5','oops',True]:
        with pytest.raises(PH3Error):_integer(value,'number')


def test_settings_can_be_saved_without_prompt(tmp_path,monkeypatch):
    from ph3_native import ui
    monkeypatch.setattr(ui,'CONFIG',tmp_path/'config.json')
    assert '保存しました' in ui.save_edit_settings(1.5, *values()[1:4], *values()[6:])
    saved=json.loads((tmp_path/'config.json').read_text(encoding='utf-8'))
    assert 'prompt' not in saved and saved['image_scale']==1.5 and 'source_image' not in saved


def test_ui_has_explicit_progress_parameter():
    from ph3_native.ui import run_edit
    params=inspect.signature(run_edit).parameters
    assert len(params)==21
    assert params['progress'].kind==inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert not any(p.kind==inspect.Parameter.VAR_POSITIONAL for p in params.values())


def test_gradio_tab_build_smoke(monkeypatch,tmp_path):
    from ph3_native import ui
    monkeypatch.setattr(ui,'CONFIG',tmp_path/'absent.json')
    monkeypatch.setattr(ui,'models',lambda base:[])
    tabs=ui.build_tab()
    assert len(tabs)==1 and tabs[0][1]=='P-H3 画像編集'
    config=tabs[0][0].get_config_file()
    assert len(config['dependencies'])==19
    assert {d['api_name'] for d in config['dependencies'] if d['api_name']} == {'run_edit', 'run_reference', 'apply_canvas_preset', 'edit_size_preview', 'edit_size_preview_1', 'stop', 'unload', 'save_edit_settings', 'check_environment', 'preview_translation', 'preview_translation_1', 'refresh_local_models', 'apply_edit_4mp', 'apply_edit_8mp', 'open_generated_folder', 'append_edit_references', 'append_composition_references'}
    finishes=[d for d in config['dependencies'] if d.get('js')=='ph3Finish']
    assert len(finishes)==2 and all(not d['trigger_only_on_success'] for d in finishes)
    assert all(d['show_progress']=='minimal' for d in config['dependencies'] if d['api_name'] in ('run_edit','run_reference'))
    serialized = json.dumps(config, ensure_ascii=False)
    assert '文章からの画像生成のみ' not in serialized
    assert 'DynamicVRAM' not in serialized


def test_result_layout_and_json_default(monkeypatch,tmp_path):
    from ph3_native import ui
    monkeypatch.setattr(ui,'CONFIG',tmp_path/'absent.json')
    monkeypatch.setattr(ui,'models',lambda base:[])
    cfg=ui.build_tab()[0][0].get_config_file()
    components=cfg['components']
    gallery=next(c for c in components if c['props'].get('elem_id')=='ph3_gallery')
    assert gallery['props']['preview'] and gallery['props']['columns']==1
    assert gallery['props']['height']==780
    checkbox=next(c for c in components if c['props'].get('label')=='生成条件のJSONを画像と一緒に保存')
    assert checkbox['props']['value'] is False


def test_reference_append_preserves_order_and_limit():
    from ph3_native.ui import append_edit_references, append_composition_references
    existing=['first.png']
    assert append_edit_references(existing,['second.png'])==['first.png','second.png']
    assert existing==['first.png']
    assert append_edit_references(existing,['first.png','second.png'])==['first.png','second.png']
    assert len(append_composition_references([str(i) for i in range(7)],['last']))==8
    import gradio as gr
    with pytest.raises(gr.Error):append_edit_references([str(i) for i in range(7)],['last'])
    with pytest.raises(gr.Error):append_composition_references([str(i) for i in range(8)],['last'])


