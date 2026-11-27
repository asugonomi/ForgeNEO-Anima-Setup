"""CPU validation tests. Header classification cases use explicit mocks, not real model weights."""
from dataclasses import replace
import json
import struct
import pytest
import torch
from safetensors.torch import save_file
from ph3_native.contracts import Request, PH3Error, stream_shapes, SIGMAS
from ph3_native import headers


def request(**kwargs):
    return replace(Request('a detailed landscape', 'dit.safetensors', 'te.safetensors', 'vae.safetensors'), **kwargs)


def test_valid_request():
    assert request().checked(False).width == 2048
    assert len(SIGMAS) == 9 and SIGMAS[0] == 1 and SIGMAS[-1] == 0
    assert all(a > b for a, b in zip(SIGMAS, SIGMAS[1:]))


@pytest.mark.parametrize('changes', [
    {'prompt':''}, {'prompt':' '}, {'prompt':'x'*24001}, {'width':2049}, {'height':128},
    {'width':4096,'height':4096}, {'width':float('nan')}, {'width':True}, {'width':1024.5},
    {'seed':-2}, {'seed':2**63}, {'count':0}, {'count':17}, {'reserve_gib':2},
    {'reserve_gib':13}, {'reserve_gib':float('nan')}, {'reserve_gib':True},
    {'tile_size':192}, {'max_tokens':63}, {'max_tokens':2049}
])
def test_invalid_request(changes):
    with pytest.raises(PH3Error):
        request(**changes).checked(False)


def test_seed_sequence_and_wrap():
    assert request(seed=17,count=3).seeds() == [17,18,19]
    assert request(seed=2**63-1,count=2).seeds() == [2**63-1,0]


def test_random_seed_once(monkeypatch):
    monkeypatch.setattr('ph3_native.contracts.secrets.randbelow', lambda limit: 42)
    assert request(seed=-1,count=2).seeds() == [42,43]


def test_stream_shapes():
    assert stream_shapes(2048,1536) == ((1,24,1,96,128),(1,32,2,2))


def test_real_safetensors_header(tmp_path):
    p=tmp_path/'small.safetensors'
    save_file({'small':torch.arange(6,dtype=torch.float32).view(2,3)},str(p),metadata={'a':'b'})
    hdr,meta=headers.read_header(p)
    assert hdr['small']['shape'] == [2,3] and meta == {'a':'b'}


@pytest.mark.parametrize('data', [b'', b'123', struct.pack('<Q',2**63),struct.pack('<Q',4)+b'xxxx',
    struct.pack('<Q',2)+b'[]'])
def test_broken_headers(tmp_path,data):
    p=tmp_path/'bad.safetensors';p.write_bytes(data)
    with pytest.raises(PH3Error): headers.read_header(p)


def test_truncated_tensor_payload(tmp_path):
    p=tmp_path/'bad.safetensors'
    h=json.dumps({'a':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}}).encode()
    p.write_bytes(struct.pack('<Q',len(h))+h+b'1234')
    with pytest.raises(PH3Error): headers.read_header(p)


def test_layer_count_gap():
    assert headers.count_layers(['blocks.0.w','blocks.1.w'],'blocks.') == 2
    with pytest.raises(PH3Error): headers.count_layers(['blocks.0.w','blocks.2.w'],'blocks.')


def test_normalized_keys():
    assert headers.normalized_key('model.language_model.layers.0.w') == 'model.layers.0.w'
    assert headers.normalized_key('model.visual.weight') == 'visual.weight'
    assert headers.normalized_key('text_connector.proj.weight') == 'text_connector.proj.weight'


@pytest.mark.parametrize('kind', ['dit','text_encoder','vae'])
def test_classification_mocked_header(monkeypatch,tmp_path,kind):
    def tensor(shape): return {'dtype':'BF16','shape':shape,'data_offsets':[0,0]}
    if kind=='dit':
        h={k:tensor([96,4]) for k in ('video_patch_proj.weight','audio_patch_proj.weight','condition_proj.weight','final_layer.video_out.weight')}
        h['blocks.0.w']=tensor([1])
    elif kind=='text_encoder':
        h={'text_connector.proj.weight':tensor([5120,12800])}
        h.update({f'model.layers.{i}.w':tensor([1]) for i in range(36)})
    else:
        h={'decoder.proj_out.weight':tensor([768,4]),'decoder.transformer_blocks.0.scale1':tensor([1])}
    p=tmp_path/'placeholder.safetensors';p.write_bytes(b'not a real checkpoint')
    monkeypatch.setattr(headers,'read_header',lambda path:(h,{}))
    monkeypatch.setattr(headers,'_quant_formats',lambda *a:[])
    assert headers.inspect_component(p,kind)['kind'] == kind
    if kind=='text_encoder': h.pop('text_connector.proj.weight')
    elif kind=='vae': h['decoder.proj_out.weight']['shape'][0]=3072
    else: h.pop('video_patch_proj.weight')
    with pytest.raises(PH3Error): headers.inspect_component(p,kind)


def test_embedded_quant_metadata(tmp_path):
    p=tmp_path/'quant.safetensors';payload=json.dumps({'format':'int8'}).encode()
    h={'layer.comfy_quant':{'dtype':'U8','shape':[len(payload)],'data_offsets':[0,len(payload)]}}
    encoded=json.dumps(h).encode();p.write_bytes(struct.pack('<Q',len(encoded))+encoded+payload)
    assert headers._quant_formats(p,h,{}) == ['int8']
