"""Text-only Japanese translation. No images or model data are transmitted."""
import re
from functools import lru_cache

import requests

from .contracts import PH3Error


@lru_cache(maxsize=64)
def translate_prompt(text):
    if not re.search(r"[\u3040-\u30ff\u3400-\u9fff]", text):
        return text
    if len(text) > 5000:
        raise PH3Error("翻訳する指示は5000文字以内にしてください。")
    # Protect explicit reference IDs rather than trusting a translator to retain them.
    tokens = []
    def protect(match):
        number = match.group(1) or match.group(2)
        token = f"PHREF{len(tokens)}TOKEN"
        tokens.append((token, f"<Picture {int(number)}>"))
        return token
    protected = re.sub(r"<Picture\s+(\d+)>|画像\s*([0-9０-９]+)", protect, text)
    try:
        response = requests.get(
            "https://translate.googleapis.com/translate_a/single",
            params={"client": "gtx", "sl": "ja", "tl": "en", "dt": "t", "q": protected},
            timeout=(5, 20),
        )
        response.raise_for_status()
        payload = response.json()
        result = "".join(part[0] for part in payload[0] if part and isinstance(part[0], str))
        if not result.strip():
            raise ValueError("empty translation")
        for token, replacement in tokens:
            if result.count(token) != 1:
                raise ValueError("reference number was changed")
            result = result.replace(token, replacement)
        return result
    except Exception as exc:
        # Do not echo network exception URLs (which contain the prompt) into the UI.
        raise PH3Error("日本語の翻訳に失敗しました。接続を確認するか、翻訳をオフにして英語で入力してください。"
                       f" ({type(exc).__name__})") from exc


def preview_translation(text):
    try:
        return translate_prompt(text)
    except PH3Error as exc:
        return str(exc)
