import json
from pathlib import Path
import pytest

from ph3_native.tokenizer import load_tokenizer


def test_official_tokenizer_preserves_ids_and_assets():
    from tokenizers import Tokenizer
    path = Path(__file__).resolve().parents[1] / "tokenizer"
    if not (path / "tokenizer.json").is_file():
        pytest.skip("Optional real-asset integration check: run setup_native first")
    before = (path / "tokenizer_config.json").read_bytes()
    tokenizer = load_tokenizer(path)
    reference = Tokenizer.from_file(str(path / "tokenizer.json"))
    for text in ("A red teapot on a wooden table.", "赤いティーポット", "<|im_start|>user\nhello<|im_end|>"):
        assert tokenizer.encode(text, add_special_tokens=False) == reference.encode(text, add_special_tokens=False).ids
    for token in json.loads(before)["extra_special_tokens"]:
        assert token in tokenizer.all_special_tokens
        assert tokenizer.convert_tokens_to_ids(token) == reference.token_to_id(token)
    assert (path / "tokenizer_config.json").read_bytes() == before


def test_list_special_tokens_are_preserved(tmp_path, monkeypatch):
    from transformers import Qwen2TokenizerFast
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "extra_special_tokens": ["<a>", "<b>"],
        "additional_special_tokens": ["<a>", "<c>"],
    }))
    calls = []
    monkeypatch.setattr(Qwen2TokenizerFast, "from_pretrained", lambda *a, **kw: calls.append(kw))
    load_tokenizer(tmp_path)
    assert calls == [{"local_files_only": True, "extra_special_tokens": {},
                      "additional_special_tokens": ["<a>", "<c>", "<b>"]}]


def test_named_special_tokens_are_not_overridden(tmp_path, monkeypatch):
    from transformers import Qwen2TokenizerFast
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"extra_special_tokens": {"image_token": "<image>"}}))
    calls = []
    monkeypatch.setattr(Qwen2TokenizerFast, "from_pretrained", lambda *a, **kw: calls.append(kw))
    load_tokenizer(tmp_path)
    assert calls == [{"local_files_only": True}]
