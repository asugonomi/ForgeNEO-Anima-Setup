from types import SimpleNamespace
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import pytest
from PIL import Image
from ph3_native.contracts import Request, PH3Error
from ph3_native.output import save_result
from ph3_native import local_prompt as lp
import base64
import io


@pytest.mark.parametrize('text,expected',[
    ('画像1の女性を画像2の女性に変えて。',True),
    ('画像２のキャラクターを画像１のキャラクターに置き換えて',True),
    ('Replace the woman in <Picture 1> with the woman from <Picture 2>.',True),
    ('画像1の背景を画像2の背景に変えて',False),
    ('画像1の女性と画像2の女性が一緒に食事する',False),
    ('画像1の女性の髪色を画像2の女性と同じにして',False),
])
def test_person_replacement_scope(text, expected):
    assert lp.is_person_replacement(lp.normalize_references(text)) is expected


def test_vision_payload_order_size_and_original_preserved(tmp_path):
    paths=[]
    for name, color in [('first','red'),('second','blue')]:
        path=tmp_path/(name+'.png');Image.new('RGB',(1280,1920),color).save(path);paths.append(path)
    before=[p.read_bytes() for p in paths]
    items=lp.vision_input('instruction',paths)
    assert items[-1]['content'].endswith('instruction')
    for index,color in enumerate([(255,0,0),(0,0,255)]):
        assert f'Picture {index+1}' in items[index*2]['content']
        raw=base64.b64decode(items[1+index*2]['data_url'].split(',',1)[1])
        with Image.open(io.BytesIO(raw)) as im:
            assert im.size==(512,768) and im.getpixel((0,0))==color
    assert before==[p.read_bytes() for p in paths]
    with pytest.raises(PH3Error):lp.vision_input('test',paths*5)


def test_nonvision_model_fails_before_loading(monkeypatch):
    monkeypatch.setattr(lp,'models',lambda base:[{'key':'text','capabilities':{'vision':False}}])
    monkeypatch.setattr(lp.subprocess,'run',lambda *a,**k:pytest.fail('must not load'))
    with pytest.raises(PH3Error,match='画像入力'):
        lp.local_transform('test','llm','text','http://127.0.0.1:1234',image_paths=['unused.png'])


def test_output_counter_and_optional_json(tmp_path):
    req=Request('test','dit','te','vae',width=256,height=256)
    a=save_result(Image.new('RGB',(8,8)),tmp_path,req,999,{'original_prompt':'日本語'})
    b=save_result(Image.new('RGB',(8,8)),tmp_path,replace(req,save_json=True),1,{})
    assert a.name=='00000001-ph3-999.png' and b.name=='00000002-ph3-1.png'
    assert not a.with_suffix('.json').exists() and b.with_suffix('.json').exists()
    assert json.loads(Image.open(a).info['ph3_generation'])['original_prompt']=='日本語'
    original=a.read_bytes()
    c=save_result(Image.new('RGB',(8,8)),tmp_path,req,999,{})
    assert c.name.startswith('00000003-') and a.read_bytes()==original


def test_concurrent_output_never_overwrites(tmp_path):
    req=Request('test','dit','te','vae')
    def write(i): return save_result(Image.new('RGB',(8,8),(i,0,0)),tmp_path,req,i,{})
    with ThreadPoolExecutor(4) as pool: paths=list(pool.map(write,range(8)))
    assert len({p.name.split('-')[0] for p in paths})==8
    assert len(set(paths))==8 and len(list(tmp_path.rglob('*.png')))==8


@pytest.mark.parametrize('url',['https://example.com','http://192.168.0.1:1234','http://localhost@evil.com','http://localhost:1234/evil'])
def test_local_url_rejects_nonlocal(url):
    with pytest.raises(PH3Error): lp.local_base(url)


def test_plamo_pfml_and_reference_validation():
    text=lp.normalize_references('画像１の衣装を保つ')
    assert text=='<Picture 1>の衣装を保つ'
    prompt=lp.pfml(text)
    assert 'lang<|plamo:val|>en' in prompt and not prompt.startswith('<|plamo:bos|>')
    with pytest.raises(PH3Error): lp.pfml('<|plamo:tag|>input')
    with pytest.raises(PH3Error): lp.check_result(text,'Keep clothing.')


def test_local_failure_unloads_only_own_instance(monkeypatch):
    state={'loaded':None};unloaded=[]
    monkeypatch.setattr(lp.Path,'is_file',lambda p:True)
    def model_list(base):
        return [{'key':'plamo-3-translate-test','loaded_instances':[{'id':state['loaded']}] if state['loaded'] else []}]
    monkeypatch.setattr(lp,'models',model_list)
    def run(args,**kw):
        state['loaded']=args[args.index('--identifier')+1]
        assert args[args.index('--gpu')+1]=='off'
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(lp.subprocess,'run',run)
    def api(base,route,payload,**kw):
        if route.endswith('/unload'):
            unloaded.append(payload['instance_id']);state['loaded']=None;return {}
        raise PH3Error('inference failed')
    monkeypatch.setattr(lp,'api',api)
    with pytest.raises(PH3Error,match='inference failed'):
        lp.local_transform('画像1の人','plamo','plamo-3-translate-test','http://127.0.0.1:1234')
    assert len(unloaded)==1 and unloaded[0].startswith('ph3-') and state['loaded'] is None


def test_existing_lm_instance_is_not_touched(monkeypatch):
    monkeypatch.setattr(lp,'models',lambda base:[{'key':'test','loaded_instances':[{'id':'user-model'}]}])
    monkeypatch.setattr(lp.subprocess,'run',lambda *a,**k:pytest.fail('must not load'))
    with pytest.raises(PH3Error,match='別のモデル'):
        lp.local_transform('test','llm','test','http://127.0.0.1:1234')
