# SPDX-License-Identifier: AGPL-3.0-or-later
"""Official P-H3 INT8 upsampler weights; Forge-native Qwen3.5 inference.

Input template: PrunaAI/P-H3-Image-Comfy/workflows/p-h3-image_edit.json.
Cached attention adapts Forge's Qwen3.5 math; no Comfy runtime is imported.
"""
from pathlib import Path
import gc
import lzma
import torch
import torch.nn.functional as F
from .contracts import ROOT, PH3Error

NAME = 'p-h3-image_prompt_upsampler_int8_convrot.safetensors'

def template(text, count):
    pictures = ''.join(f'<Picture {i+1}>: <|vision_start|><|image_pad|><|vision_end|>' for i in range(count))
    return '<|im_start|>user\n' + pictures + 'Edit <Picture 1>. ' + text + '<|im_end|>\n<|im_start|>assistant\n<think>\n'

def extract_result(raw):
    _, sep, result = raw.partition('</think>')
    if not sep or not result.strip():
        raise PH3Error('公式Upsamplerが有効な編集指示を返しませんでした。思考のみの出力は画像生成に使用しません。')
    return result.strip()

def _load():
    from backend import memory_management as mm
    from backend.operations import using_forge_operations
    from backend.state_dict import convert_quantization, detect_quantization
    from backend.utils import load_torch_file
    from backend.patcher.base import ModelPatcher
    from backend.nn.llm.llama import Qwen35_9B
    from transformers.modeling_utils import no_init_weights
    from .loading import _strict_load
    path = ROOT.parent.parent / 'models/P-H3-Image/text_encoders' / NAME
    if not path.is_file(): raise PH3Error(f'公式Upsamplerの重みがありません: {path}')
    sd, md = load_torch_file(str(path), safe_load=True, device=mm.cpu, return_metadata=True)
    sd, md = convert_quantization(sd, md)
    sd = {k.replace('model.language_model.', 'model.').replace('model.visual.', 'visual.'): v for k,v in sd.items()}
    # MTP is the optional speculative draft head. Use the verified main head only;
    # greedy decoding does not need draft predictions. Never ignore other keys.
    draft = [k for k in sd if k.startswith('mtp.')]
    print(f'[P-H3 Upsampler] MTP speculative decoding disabled; {len(draft)} draft tensors excluded explicitly.')
    sd = {k:v for k,v in sd.items() if k not in draft}
    quant = dict(detect_quantization(sd, is_unet=False) or {})
    quant.pop('layout', None)
    class Model(Qwen35_9B):
        def __init__(self):
            super().__init__({})
            self.lm_head = torch.nn.Linear(4096,248320,bias=False)
    with no_init_weights(), using_forge_operations(device=mm.cpu,dtype=torch.bfloat16,manual_cast_enabled=True,extra_dtype=quant):
        model=Model()
    _strict_load(model,sd,'official_upsampler')
    del sd
    return ModelPatcher(model,load_device=mm.get_torch_device(),offload_device=mm.cpu)

def _linear(layer, x, cache):
    from backend.nn.llm.llama import torch_chunk_gated_delta_rule
    b,s,_=x.shape
    projected=layer.in_proj_qkv(x).transpose(1,2)
    previous=cache.get('conv')
    joined=projected if previous is None else torch.cat((previous,projected),-1)
    cache['conv']=F.pad(joined,(max(0,3-joined.shape[-1]),0))[...,-3:].clone()
    mixed=F.silu(layer.conv1d(joined)[..., :joined.shape[-1]])[...,-s:].transpose(1,2)
    q,k,v=mixed.split([layer.key_dim,layer.key_dim,layer.value_dim],-1)
    q=q.reshape(b,s,layer.num_key_heads,layer.key_head_dim).repeat_interleave(layer.num_value_heads//layer.num_key_heads,2)
    k=k.reshape(b,s,layer.num_key_heads,layer.key_head_dim).repeat_interleave(layer.num_value_heads//layer.num_key_heads,2)
    v=v.reshape(b,s,layer.num_value_heads,layer.value_head_dim)
    beta=layer.in_proj_b(x).sigmoid()
    g=-layer.A_log.to(x.device,torch.float32).exp()*F.softplus(layer.in_proj_a(x).float()+layer.dt_bias.to(x.device,torch.float32))
    if s == 1 and 'state' in cache:
        # Single-token gated delta recurrence (same update as chunked prefill).
        qn=F.normalize(q,dim=-1).float()[:,0]
        kn=F.normalize(k,dim=-1).float()[:,0]
        state=cache['state']*g[:,0].exp()[...,None,None]
        delta=(v[:,0].float()-(state*kn[...,None]).sum(-2))*beta[:,0].float()[...,None]
        state=state+kn[...,None]*delta[...,None,:]
        out=((state*qn[...,None]).sum(-2)/(layer.key_head_dim**.5))[:,None].to(x.dtype)
    else:
        out,state=torch_chunk_gated_delta_rule(q,k,v,g,beta,initial_state=cache.get('state'),output_final_state=True)
    cache['state']=state
    z=layer.in_proj_z(x)
    out=layer.norm(out.reshape(-1,layer.value_head_dim))*F.silu(z.reshape(-1,layer.value_head_dim))
    return layer.out_proj(out.reshape(b,s,-1))

def _attention(layer,x,freqs,cache):
    from backend.nn.llm.llama import apply_partial_rope, attention_function
    b,s,_=x.shape
    qg=layer.q_proj(x).view(b,s,layer.num_heads,layer.head_dim*2)
    q,gate=qg[...,:layer.head_dim],qg[...,layer.head_dim:]
    q=layer.q_norm(q).transpose(1,2)
    k=layer.k_norm(layer.k_proj(x).view(b,s,layer.num_kv_heads,layer.head_dim)).transpose(1,2)
    v=layer.v_proj(x).view(b,s,layer.num_kv_heads,layer.head_dim).transpose(1,2)
    q,k=apply_partial_rope(q,k,freqs,layer.rotary_dim)
    old=cache.get('kv')
    if old is not None: k,v=torch.cat((old[0],k),2),torch.cat((old[1],v),2)
    cache['kv']=(k,v)
    mask=None
    if s>1:
        mask=torch.full((s,k.shape[2]),torch.finfo(x.dtype).min/4,device=x.device,dtype=x.dtype).triu_(1+k.shape[2]-s)
    out=attention_function(q,k,v,layer.num_heads,mask=mask,skip_reshape=True,enable_gqa=True)
    return layer.o_proj(out*gate.reshape(b,s,-1).sigmoid())

def _step(model,x,pos,caches,check):
    from backend.nn.llm.llama import precompute_freqs_cis
    cfg=model.model.config
    freqs=precompute_freqs_cis(int(cfg.head_dim*cfg.partial_rotary_factor),pos,cfg.rope_theta,cfg.rope_scale,cfg.mrope_section,interleaved_mrope=True,device=x.device)
    for layer,cache in zip(model.model.layers,caches):
        check()
        h=layer.input_layernorm(x)
        h=_linear(layer.linear_attn,h,cache) if layer.layer_type=='linear_attention' else _attention(layer.self_attn,h,freqs,cache)
        x=x+h
        x=x+layer.mlp(layer.post_attention_layernorm(x))
    x=model.model.norm(x[:,-1:])
    return int(model.lm_head(x).float().argmax(-1).item())

@torch.inference_mode()
def generate(text, image_paths, check=lambda:None, report=lambda s:None):
    from PIL import Image,ImageOps
    import numpy as np
    from tokenizers import Tokenizer
    from backend import memory_management as mm
    from backend.huggingface import Token
    from backend.nn.llm.qwen_vl import qwen2vl_mrope_position_ids
    if not 1<=len(image_paths)<=8: raise PH3Error('公式Upsamplerは参照画像1〜8枚が必要です。')
    tokenizer=Tokenizer.from_str(lzma.open(Token.moss_compress,'rt',encoding='utf-8').read())
    report('公式INT8 Upsamplerを読み込み中')
    patcher=None
    caches=[]
    try:
        check()
        mm.unload_all_models()
        patcher=_load()
        mm.load_models_gpu([patcher],memory_required=2*1024**3,minimum_memory_required=2*1024**3,force_full_load=False)
        model=patcher.model
        device=mm.get_torch_device()
        ids=tokenizer.encode(template(text,len(image_paths)),add_special_tokens=False).ids
        image_id=tokenizer.token_to_id('<|image_pad|>')
        pieces=[];info=[];start=0;length=0;index=0
        with Image.open(image_paths[0]) as im: w,h=im.size
        scale=(1024*1024/(w*h))**.5
        size=(max(1,round(w*scale)),max(1,round(h*scale)))
        for i,t in enumerate(ids):
            if t!=image_id: continue
            check()
            tokens=torch.tensor([ids[start:i]],device=device)
            prefix=model.model.embed_tokens(tokens).to(torch.bfloat16)
            pieces.append(prefix);length+=prefix.shape[1]
            with Image.open(image_paths[index]) as im:
                rgb=ImageOps.exif_transpose(im).convert('RGB')
                rgb=rgb.resize(size,Image.Resampling.LANCZOS) if index==0 else ImageOps.pad(rgb,size,method=Image.Resampling.LANCZOS,color='white')
                pixels=torch.from_numpy(np.array(rgb).astype(np.float32)/255)[None]
            merged,grid=model.preprocess_embed({'type':'image','data':pixels},device)
            pieces.append(merged[None].to(torch.bfloat16))
            info.append({'type':'image','index':length,'size':merged.shape[0],'extra':grid})
            length+=merged.shape[0];start=i+1;index+=1
        if index!=len(image_paths):raise PH3Error('画像タグと入力枚数が一致しません。')
        pieces.append(model.model.embed_tokens(torch.tensor([ids[start:]],device=device)).to(torch.bfloat16))
        x=torch.cat(pieces,1);del pieces,merged,prefix,pixels
        pos=qwen2vl_mrope_position_ids(info,x.shape[1],device)
        next_position=int(pos.max().item())+1
        caches=[{} for _ in model.model.layers]
        output=[]
        stop={tokenizer.token_to_id('<|im_end|>'),tokenizer.token_to_id('<|endoftext|>')}
        for i in range(3072):
            check()
            if i%16==0: report(f'公式Upsamplerで編集指示を作成中 ({i}/3072 tokens)')
            t=_step(model,x,pos,caches,check)
            if t in stop: break
            output.append(t)
            x=model.model.embed_tokens(torch.tensor([[t]],device=device)).to(torch.bfloat16)
            pos=torch.full((3,1),next_position,device=device,dtype=torch.float32);next_position+=1
        else:raise PH3Error('公式Upsamplerが3072トークン上限に達しました。画像生成は開始しません。')
        return extract_result(tokenizer.decode(output,skip_special_tokens=False))
    finally:
        report('公式Upsamplerを解放中')
        caches.clear()
        mm.unload_all_models()
        del patcher
        gc.collect();mm.soft_empty_cache()
