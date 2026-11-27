import sys
from types import SimpleNamespace
from ph3_native import local_prompt

def test_official_preserves_request_and_image_order(monkeypatch):
    received={}
    def generate(text, image_paths, **kwargs):
        received.update(text=text, images=image_paths)
        return 'Edit <Picture 1>. Test result.'
    monkeypatch.setitem(sys.modules,'ph3_native.upsampler',SimpleNamespace(generate=generate))
    text='画像１の女性を画像２の馬耳を持った女性に書き換えて。'
    result=local_prompt.process_prompt(text,'official',image_paths=('base.png','reference.png'))
    assert result=='Edit <Picture 1>. Test result.'
    assert received=={'text':text,'images':('base.png','reference.png')}

def test_official_template_and_incomplete_output():
    import pytest
    from ph3_native.upsampler import template, extract_result
    from ph3_native.contracts import PH3Error
    text='画像１の女性を画像２の女性に変えて。'
    formatted=template(text,2)
    assert text in formatted
    assert formatted.count('<|image_pad|>')==2
    assert formatted.index('<Picture 1>')<formatted.index('<Picture 2>')
    assert formatted.endswith('<think>\n')
    assert extract_result('reasoning</think> Edit <Picture 1>.')=='Edit <Picture 1>.'
    with pytest.raises(PH3Error):extract_result('unfinished reasoning')
    with pytest.raises(PH3Error):extract_result('reasoning</think> ')
