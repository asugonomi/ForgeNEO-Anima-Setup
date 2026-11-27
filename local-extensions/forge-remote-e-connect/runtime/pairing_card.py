"""Make a self-contained pairing card on this PC. No code is sent to a QR service."""
import os
from pathlib import Path
import re
import tempfile
import webbrowser

from qr.qrcodegen import QrCode


def qr_svg(text):
    qr = QrCode.encode_text(text, QrCode.Ecc.MEDIUM)
    size = qr.get_size()
    runs = []
    for y in range(size):
        x = 0
        while x < size:
            if not qr.get_module(x, y):
                x += 1
                continue
            start = x
            while x < size and qr.get_module(x, y):
                x += 1
            width = x - start
            runs.append(f"M{start+4} {y+4}h{width}v1h-{width}z")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size+8} {size+8}" '
            'role="img" aria-label="端末登録用QRコード" shape-rendering="crispEdges">'
            '<rect width="100%" height="100%" fill="white"/>'
            f'<path d="{"".join(runs)}" fill="black"/></svg>')


def show_pairing_card(code: str, directory: Path, *, open_browser: bool = True) -> Path:
    if not re.fullmatch(r"FER1\.[A-Za-z0-9_-]{16,1800}", code):
        raise ValueError("Invalid pairing code")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    svg = qr_svg("forgeremote://pair#" + code)
    html = f"""<!doctype html><html lang="ja"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>端末を登録 · Forge Remote E</title>
<style>
*{{box-sizing:border-box}}body{{margin:0;padding:24px 16px;background:#10191c;color:#f1f5f4;font:16px/1.6 system-ui,sans-serif}}
main{{max-width:640px;margin:auto}}h1{{margin:0;font-size:1.7rem}}p{{margin:.6em 0 1em}}
svg{{display:block;background:#fff;width:min(100%,410px);height:auto;margin:20px auto}}
details{{padding:.6em 0}}summary{{cursor:pointer}}textarea{{width:100%;font:13px monospace;margin-top:8px}}
</style><main><h1>Forge Remote E（Everyone）に端末を登録</h1>
<p>スマートフォンをこのPCと同じWi-Fiにつなぎ、カメラで読み取ってForge Remote Eで開いてください。</p>
{svg}
<p>このQRは1台・15分間有効です。端末に登録が済んだらこのページを閉じてください。</p>
<details><summary>コードをコピーして登録する</summary>
<p>アプリの「接続」で登録コードを貼り付けます。</p>
<textarea aria-label="登録コード" readonly rows="4">{code}</textarea></details></main></html>"""
    target = directory / "pairing.html"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                     prefix=".card-", suffix=".tmp", delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(html)
        except Exception:
            file.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    if open_browser:
        webbrowser.open(target.resolve().as_uri())
    return target
