"""Read upstream tokenizer assets without changing them or host transformers."""
import json
from pathlib import Path


def load_tokenizer(path):
    from transformers import Qwen2TokenizerFast

    path = Path(path)
    config = json.loads((path / "tokenizer_config.json").read_text(encoding="utf-8"))
    overrides = {}
    extra = config.get("extra_special_tokens")
    if isinstance(extra, list):
        # Transformers 4.x uses a named-token mapping here; newer assets use a
        # list. Preserve those tokens through the 4.x additional-token API.
        overrides["extra_special_tokens"] = {}
        overrides["additional_special_tokens"] = list(dict.fromkeys(
            (config.get("additional_special_tokens") or []) + extra))
    return Qwen2TokenizerFast.from_pretrained(str(path), local_files_only=True, **overrides)
