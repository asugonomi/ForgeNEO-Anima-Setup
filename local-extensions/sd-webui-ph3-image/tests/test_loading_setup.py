"""Small strict-loading tests and mocked-network setup tests. No remote downloads are made."""
import io
import importlib.util
import json
from pathlib import Path
import zipfile
import pytest
import torch
from ph3_native.contracts import PH3Error,UPSTREAM_COMMIT
from ph3_native.loading import _strict_load,_restore_plain_fp32


def test_strict_load_rejects_missing_key():
    model=torch.nn.Linear(2,2)
    with pytest.raises(PH3Error):_strict_load(model,{'weight':torch.ones(2,2)},'toy')


def test_strict_load_and_fp32_restore():
    model=torch.nn.Linear(2,2,dtype=torch.float16)
    _strict_load(model,{'weight':torch.ones(2,2,dtype=torch.float16),'bias':torch.zeros(2,dtype=torch.float16)},'toy')
    assert not model.training and not model.weight.requires_grad
    _restore_plain_fp32(model,{'weight':torch.ones(2,2,dtype=torch.float32)})
    assert model.weight.dtype==torch.float32


def test_restore_rejects_unknown_key():
    with pytest.raises(PH3Error):_restore_plain_fp32(torch.nn.Linear(2,2),{'missing':torch.ones(2)})


def setup_module_from_path():
    path=Path(__file__).resolve().parents[1]/'tools/setup_native.py'
    spec=importlib.util.spec_from_file_location('ph3_setup_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def archive(entries):
    blob=io.BytesIO()
    with zipfile.ZipFile(blob,'w') as z:
        for path,data in entries.items():z.writestr('repository/'+path,data)
    return blob.getvalue()


def test_setup_native_extracts_only_library_not_scripts(tmp_path,monkeypatch):
    tool=setup_module_from_path();monkeypatch.setattr(tool,'ROOT',tmp_path)
    blob=archive({'forge_h3/native/transformer.py':'# source','forge_h3/__init__.py':'',
                  'LICENSE':'test license fixture','scripts/register.py':'raise RuntimeError()',
                  'install.py':'raise RuntimeError()'})
    monkeypatch.setattr(tool,'get',lambda *a,**k:blob)
    tool.install_native()
    root=tmp_path/'_vendor/h3'
    assert (root/'forge_h3/native/transformer.py').is_file()
    assert not (root/'scripts/register.py').exists() and not (root/'install.py').exists()
    assert json.loads((root/'manifest.json').read_text())['commit']==UPSTREAM_COMMIT


def test_setup_rejects_archive_traversal(tmp_path,monkeypatch):
    tool=setup_module_from_path();monkeypatch.setattr(tool,'ROOT',tmp_path)
    blob=archive({'forge_h3/../../outside.py':'bad'})
    monkeypatch.setattr(tool,'get',lambda *a,**k:blob)
    with pytest.raises(RuntimeError,match='Unsafe'):tool.install_native()
    assert not (tmp_path/'outside.py').exists()


def test_tokenizer_setup_is_pinned_and_has_receipt(tmp_path,monkeypatch):
    tool=setup_module_from_path();monkeypatch.setattr(tool,'ROOT',tmp_path);calls=[]
    sha='1'*40
    files={'tokenizer.json':b'{}','tokenizer_config.json':b'{}'}
    def get(url,**kwargs):
        calls.append(url)
        if '/api/models/' in url:return json.dumps({'sha':sha,'siblings':[{'rfilename':'tokenizer/'+n} for n in files]}).encode()
        assert '/resolve/'+sha+'/' in url
        return files[url.rsplit('/',1)[-1]]
    monkeypatch.setattr(tool,'get',get)
    tool.install_tokenizer()
    assert json.loads((tmp_path/'tokenizer/setup_receipt.json').read_text())['revision']==sha
    assert len(calls)==3
    tool.install_tokenizer()
    assert len(calls)==3
