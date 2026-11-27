"""Mocked Forge orchestration tests plus real PNG/JSON file round-trips. GPU is never used."""
from dataclasses import replace
import json
from pathlib import Path
import sys
import threading
import types
import pytest
import torch
from PIL import Image
from ph3_native.contracts import Request, PH3Error, Cancelled
from ph3_native.output import save_result
from ph3_native import engine as eng


@pytest.fixture
def host(monkeypatch,tmp_path):
    lock=threading.Lock();events=[]
    mm=types.SimpleNamespace(get_torch_device=lambda:torch.device('cpu'),
        unload_all_models=lambda:events.append('unload'),soft_empty_cache=lambda:None,cleanup_models=lambda:None)
    backend=types.ModuleType('backend');backend.__path__=[];backend.memory_management=mm
    state=types.SimpleNamespace(interrupted=False,textinfo='',begin=lambda label:events.append('begin'),end=lambda:events.append('end'))
    modules=types.ModuleType('modules');modules.__path__=[];modules.shared=types.SimpleNamespace(state=state)
    queue=types.ModuleType('modules.call_queue');queue.queue_lock=lock
    pending={}; finished=[]
    modules.progress=types.SimpleNamespace(create_task_id=lambda kind:'task(test)',
        add_task_to_queue=lambda task:pending.update({task:True}),
        start_task=lambda task:pending.pop(task,None),
        finish_task=lambda task:finished.append(task),pending_tasks=pending,finished_tasks=finished)
    for k,v in [('backend',backend),('backend.memory_management',mm),('modules',modules),('modules.call_queue',queue)]:
        monkeypatch.setitem(sys.modules,k,v)
    monkeypatch.setattr(eng,'ROOT',tmp_path)
    monkeypatch.setattr(eng,'runtime_checks',lambda:{'gpu':'MOCK','torch':'CPU TEST','cuda_build':None})
    monkeypatch.setattr(torch.cuda,'reset_peak_memory_stats',lambda device:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda device:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda device:0)
    for name in ['dit','text','vae']:(tmp_path/f'{name}.safetensors').write_bytes(b'placeholder')
    req=Request('test',str(tmp_path/'dit.safetensors'),str(tmp_path/'text.safetensors'),str(tmp_path/'vae.safetensors'),width=256,height=256,seed=7)
    return lock,events,state,req,tmp_path


def test_png_and_json_metadata(tmp_path):
    req=Request('a landscape','dit.safetensors','te.safetensors','vae.safetensors',width=256,height=256,save_json=True)
    p=save_result(Image.new('RGB',(4,4)),tmp_path,req,10,{'tile_size':128,'test':True})
    with Image.open(p) as im:
        assert 'Seed: 10' in im.info['parameters']
        assert 'VAE tile: 128' in im.info['parameters']
        assert json.loads(im.info['ph3_generation'])['test'] is True
    assert json.loads(p.with_suffix('.json').read_text())['test'] is True
    assert not list(tmp_path.rglob('*.tmp'))


def test_generation_lock_and_cleanup(host,monkeypatch):
    lock,events,state,req,root=host;e=eng.Engine()
    def attempt(req,seed,reserve,tile,progress):
        assert lock.locked() and e.job_guard.locked()
        return Image.new('RGB',(4,4)),{'tile_size':tile}
    monkeypatch.setattr(e,'_attempt',attempt)
    results=e.generate(req,root)
    assert len(results)==1 and Path(results[0]).is_file()
    assert e.cleanup_ok and not lock.locked() and not e.job_guard.locked()
    assert events[0]=='begin' and events[-1]=='end' and events.count('unload')>=2


def test_oom_retry_keeps_seed_and_resolution(host,monkeypatch):
    lock,events,state,req,root=host;e=eng.Engine();attempts=[]
    def attempt(req,seed,reserve,tile,progress):
        attempts.append((seed,req.width,req.height,reserve,tile))
        if len(attempts)==1:raise torch.cuda.OutOfMemoryError('mock oom')
        return Image.new('RGB',(4,4)),{'tile_size':tile}
    monkeypatch.setattr(e,'_attempt',attempt)
    assert len(e.generate(req,root))==1
    assert attempts==[(7,256,256,6.,256),(7,256,256,8.,128)]
    assert e.cleanup_ok


def test_repeated_oom_is_not_success(host,monkeypatch):
    lock,events,state,req,root=host;e=eng.Engine()
    def attempt(*a):raise torch.cuda.OutOfMemoryError('mock oom')
    monkeypatch.setattr(e,'_attempt',attempt)
    with pytest.raises(PH3Error,match='VRAM不足'):e.generate(req,root)
    assert not e.last_outputs and e.cleanup_ok and not lock.locked() and not e.job_guard.locked()


def test_cancel_cleans_up(host,monkeypatch):
    lock,events,state,req,root=host;e=eng.Engine()
    def attempt(*a):e.cancel();e.check()
    monkeypatch.setattr(e,'_attempt',attempt)
    with pytest.raises(Cancelled):e.generate(req,root)
    assert not any('Traceback' in event['message'] for event in e.events)
    assert e.cleanup_ok and not lock.locked() and not e.job_guard.locked()


def test_cleanup_failure_is_reported(host,monkeypatch):
    lock,events,state,req,root=host;e=eng.Engine();calls=[]
    def unload():
        calls.append(1)
        if len(calls)>1:raise RuntimeError('mock cleanup failure')
    monkeypatch.setattr(e,'unload_gpu',unload)
    monkeypatch.setattr(e,'_attempt',lambda *a:(Image.new('RGB',(4,4)),{}))
    e.generate(req,root)
    assert e.cleanup_ok is False
    assert any('メモリ解放処理で例外' in x['message'] for x in e.events)
    assert not lock.locked() and not e.job_guard.locked()


def test_preflight_failure_releases_guard(host,monkeypatch):
    lock,events,state,req,root=host;e=eng.Engine()
    def fail():raise PH3Error('preflight failed')
    monkeypatch.setattr(eng,'runtime_checks',fail)
    with pytest.raises(PH3Error):e.generate(req,root)
    assert not lock.locked() and not e.job_guard.locked() and not e.cleanup_ok
    assert sys.modules['modules'].progress.finished_tasks == ['task(test)']
    assert not sys.modules['modules'].progress.pending_tasks


def test_native_progress_and_translation_metadata(host, monkeypatch):
    from ph3_native import translation
    lock, events, state, req, root = host
    e = eng.Engine()
    monkeypatch.setattr(translation, 'translate_prompt', lambda text: 'Translated instruction')
    def attempt(req, seed, reserve, tile, progress):
        assert req.prompt == 'Translated instruction'
        progress(.5, '画像生成 4/8')
        assert state.job_count == 1 and state.job_no == 0
        assert state.sampling_step / state.sampling_steps == .5
        assert '4/8' in state.textinfo
        return Image.new('RGB', (4, 4)), {}
    monkeypatch.setattr(e, '_attempt', attempt)
    result = e.generate(req, root, translate=True, task_id='task(ph3-test)')
    data = json.loads(Image.open(result[0]).info['ph3_generation'])
    assert data['original_prompt'] == req.prompt
    assert data['translated_prompt'] == 'Translated instruction'
    assert sys.modules['modules'].progress.finished_tasks == ['task(ph3-test)']


def test_translation_error_releases_task_without_inference(host, monkeypatch):
    from ph3_native import translation
    lock, events, state, req, root = host
    e = eng.Engine()
    def fail(text): raise PH3Error('translation failed')
    monkeypatch.setattr(translation, 'translate_prompt', fail)
    monkeypatch.setattr(e, '_attempt', lambda *a: pytest.fail('inference must not run'))
    with pytest.raises(PH3Error, match='translation failed'):
        e.generate(req, root, translate=True)
    assert not lock.locked() and not e.job_guard.locked() and e.cleanup_ok
    assert sys.modules['modules'].progress.finished_tasks == ['task(test)']


def test_cache_discard_setting(host,monkeypatch):
    lock,events,state,req,root=host;e=eng.Engine()
    e.components={'fake':'data'};e.text_cache=('a','b')
    monkeypatch.setattr(e,'_attempt',lambda *a:(Image.new('RGB',(4,4)),{}))
    e.generate(replace(req,keep_cpu_cache=False),root)
    assert not e.components and e.text_cache is None
