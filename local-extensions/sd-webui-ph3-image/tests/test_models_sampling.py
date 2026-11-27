"""Real CPU tensor tests with small synthetic models. No P-H3 weight inference is performed."""
import sys
import types
import pytest
import torch
from torch import nn
from ph3_native import models
from ph3_native.models import TextConnector, image_token_ids, make_decoder_class
from ph3_native.sampling import sample
from ph3_native.contracts import SIGMAS, FRAME_POSITION, Cancelled, PH3Error, stream_shapes


def test_connector_normalization_and_sink_replacement():
    c=TextConnector(input_dim=5,output_dim=2)
    with torch.no_grad():
        c.proj.weight.copy_(torch.tensor([[1,0,0,0,0],[0,0,0,0,1]],dtype=torch.float32))
        c.input_mean.fill_(1);c.input_std.fill_(2)
        c.output_std.fill_(3);c.output_mean.fill_(4);c.sink_embedding.copy_(torch.tensor([99.,100.]))
    h=[torch.tensor([[[float(i)],[float(i+2)],[float(i+4)]]]) for i in range(5)]
    out=c(h,dtype=torch.float32)
    assert out.shape == (1,3,2)  # the sink REPLACES, never prepends, the first token
    assert c.proj.bias is None
    assert torch.equal(out[0,0],torch.tensor([99.,100.]))
    assert torch.equal(out[0,1],torch.tensor([5.5,11.5]))
    assert all(p.dtype==torch.float32 for p in c.parameters())


def test_image_coordinates():
    ids=image_token_ids(2,3,device='cpu',dtype=torch.float32)
    assert ids.shape==(1,6,3)
    torch.testing.assert_close(ids[...,0],torch.full((1,6),FRAME_POSITION))
    torch.testing.assert_close(ids[0,:,1],torch.tensor([-.5,-.5,-.5,.5,.5,.5]))
    torch.testing.assert_close(ids[0,:,2],torch.tensor([-2/3,0,2/3,-2/3,0,2/3]))


def test_selected_qwen_layers_with_mock_backend(monkeypatch):
    class Add(nn.Module):
        def forward(self,x,**kwargs): return x+1,None
    class Norm(nn.Module):
        def forward(self,x): return x+100
    class Llama(nn.Module):
        def __init__(self,config):
            super().__init__();self.config=config
            self.embed_tokens=nn.Embedding(2,1);self.embed_tokens.weight.data.zero_()
            self.layers=nn.ModuleList([Add() for _ in range(36)]);self.norm=Norm()
    class Capture(nn.Module):
        def forward(self,captured,dtype): return torch.cat(captured,-1).to(dtype)
    config=lambda:types.SimpleNamespace(hidden_size=2560,num_hidden_layers=36,head_dim=1,rope_theta=1,rope_scale=None,rope_dims=None)
    llama=types.SimpleNamespace(Qwen3VL_4BConfig=config,Llama2_=Llama,precompute_freqs_cis=lambda *a,**k:None,attention_function=None)
    backend=types.ModuleType('backend');backend.__path__=[]
    bnn=types.ModuleType('backend.nn');bnn.__path__=[]
    llm=types.ModuleType('backend.nn.llm');llm.llama=llama
    for key,val in [('backend',backend),('backend.nn',bnn),('backend.nn.llm',llm)]:monkeypatch.setitem(sys.modules,key,val)
    monkeypatch.setattr(models,'TextConnector',Capture)
    model=models.make_text_encoder_class()()
    out=model(torch.tensor([[0,1]]))
    assert out[0,0].tolist()==[13,19,25,31,136]


def test_single_frame_decoder_layout():
    class Pos(nn.Module):
        def forward(self,x):self.ids=x;return None
    class Decoder(nn.Module):
        def __init__(self,patch_size_t):
            super().__init__();assert patch_size_t==1
            self.patch_size=2;self.out_channels=3;self.num_register_tokens=2
            self.x_embedder=nn.Linear(1,4);self.register_tokens=nn.Parameter(torch.zeros(1,2,4))
            self.register_buffer('mask_token',torch.zeros(1,1,4))
            self.pos_embed=Pos();self.transformer_blocks=nn.ModuleList([])
            self.norm_out=nn.Identity();self.proj_out=nn.Linear(4,12)
    native=types.SimpleNamespace(ViT3DDecoder=Decoder,kernels=types.SimpleNamespace(cast_to=lambda x,**k:x.to(**k)))
    decoder=make_decoder_class(native)()
    out=decoder(torch.zeros(1,1,1,2,3))
    assert out.shape==(1,3,1,4,6)
    assert not hasattr(decoder,'mask_token')
    assert decoder.pos_embed.ids.shape==(1,9,3)
    assert torch.equal(decoder.pos_embed.ids[0,-3:],torch.zeros(3,3))
    with pytest.raises(PH3Error):decoder(torch.zeros(1,1,2,2,3))


class FakeDiT:
    def __init__(self,nan=False):self.calls=[];self.refined=0;self.nan=nan
    def preprocess_text_embeds(self,context):self.refined+=1;return context
    def forward_streams(self,streams,t,context,transformer_options,minimax_payload):
        self.calls.append((t.item(),[tuple(x.shape) for x in streams],minimax_payload))
        return [torch.full_like(x,float('nan') if self.nan else 1.) for x in streams]


def test_sampler_exact_schedule_audio_and_direction():
    dit=FakeDiT();steps=[]
    result=sample(dit,torch.zeros(1,1,1),32,32,123,'cpu',progress=steps.append)
    rng=torch.Generator('cpu').manual_seed(123)
    initial=torch.randn(stream_shapes(32,32)[0],generator=rng)
    torch.testing.assert_close(result,initial-1,atol=1e-6,rtol=1e-6)
    assert dit.refined==1 and len(dit.calls)==8 and steps==list(range(1,9))
    for call,sigma in zip(dit.calls,SIGMAS):
        assert abs(call[0]-sigma*1000)<1e-3
        assert call[1][1]==(1,32,2,2)
        assert call[2]['audio_scale']==1.


def test_sampler_deterministic_seed():
    args=(torch.zeros(1,1,1),32,32,12,'cpu')
    assert torch.equal(sample(FakeDiT(),*args),sample(FakeDiT(),*args))


def test_sampler_nan_rejected():
    with pytest.raises(PH3Error):sample(FakeDiT(True),torch.zeros(1,1,1),32,32,12,'cpu')


def test_sampler_cancel():
    dit=FakeDiT()
    def check():raise Cancelled('stop')
    with pytest.raises(Cancelled):sample(dit,torch.zeros(1,1,1),32,32,12,'cpu',check=check)
    assert not dit.calls
