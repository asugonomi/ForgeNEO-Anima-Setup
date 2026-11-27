import pytest
from ph3_native import translation as t
from ph3_native.contracts import PH3Error


def test_english_does_not_call_network(monkeypatch):
    t.translate_prompt.cache_clear()
    monkeypatch.setattr(t.requests, 'get', lambda *a, **k: pytest.fail('network'))
    assert t.translate_prompt('Keep <Picture 1> unchanged.') == 'Keep <Picture 1> unchanged.'


def test_references_preserved_and_only_text_sent(monkeypatch):
    t.translate_prompt.cache_clear()
    class Response:
        def raise_for_status(self): pass
        def json(self): return [[['Use PHREF0TOKEN and PHREF1TOKEN. Keep the clothes unchanged.']]]
    def get(url, **kw):
        assert kw['params']['q'] == 'PHREF0TOKENとPHREF1TOKEN。衣装は変えない。'
        assert set(kw) == {'params', 'timeout'}
        return Response()
    monkeypatch.setattr(t.requests, 'get', get)
    assert t.translate_prompt('画像１と<Picture 2>。衣装は変えない。') == 'Use <Picture 1> and <Picture 2>. Keep the clothes unchanged.'


def test_failure_does_not_fall_back(monkeypatch):
    t.translate_prompt.cache_clear()
    def fail(*a, **k): raise TimeoutError('secret prompt in url')
    monkeypatch.setattr(t.requests, 'get', fail)
    with pytest.raises(PH3Error) as err: t.translate_prompt('テスト指示')
    assert 'secret' not in str(err.value)


def test_missing_reference_is_error(monkeypatch):
    t.translate_prompt.cache_clear()
    class Response:
        def raise_for_status(self): pass
        def json(self): return [[['reference disappeared']]]
    monkeypatch.setattr(t.requests, 'get', lambda *a, **k: Response())
    with pytest.raises(PH3Error): t.translate_prompt('画像1の服を保つ')
