# SPDX-License-Identifier: AGPL-3.0-or-later
"""Manual, stdlib-only setup. Does NOT pip-install, update Forge, or download weights."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ph3_native.contracts import UPSTREAM_COMMIT, UPSTREAM_REPO


def get(url: str, limit=128 * 1024**2) -> bytes:
    print("GET", url)
    req = urllib.request.Request(url, headers={"User-Agent": "Forge-PH3-Setup/0.1"})
    with urllib.request.urlopen(req, timeout=90) as r:
        chunks, count = [], 0
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            count += len(chunk)
            if count > limit:
                raise RuntimeError("Download exceeds the setup size limit; no large model is downloaded here.")
            chunks.append(chunk)
    return b"".join(chunks)


def install_native(repair=False):
    target = ROOT / "_vendor" / "h3"
    if target.exists() and not repair:
        from ph3_native.provenance import verify_vendor
        verify_vendor()
        print("Native library already installed and checksums verified.")
        return
    blob = get(f"https://codeload.github.com/{UPSTREAM_REPO}/zip/{UPSTREAM_COMMIT}")
    with tempfile.TemporaryDirectory(prefix="ph3-native-") as temp:
        staging = Path(temp) / "h3"
        staging.mkdir()
        hashes, expanded = {}, 0
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            for entry in archive.infolist():
                rel = Path(*Path(entry.filename).parts[1:])
                if entry.is_dir() or not rel.parts:
                    continue
                if not (str(rel).replace("\\", "/").startswith("forge_h3/") or rel.as_posix() == "LICENSE"):
                    continue
                destination = (staging / rel).resolve()
                if not destination.is_relative_to(staging.resolve()):
                    raise RuntimeError("Unsafe archive path.")
                expanded += entry.file_size
                if expanded > 64 * 1024**2:
                    raise RuntimeError("Unexpectedly large native source archive.")
                data = archive.read(entry)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                hashes[rel.as_posix()] = hashlib.sha256(data).hexdigest()
        if "forge_h3/native/transformer.py" not in hashes or "LICENSE" not in hashes:
            raise RuntimeError("The pinned archive has an unexpected layout.")
        manifest = {"repository": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT,
                    "archive_sha256": hashlib.sha256(blob).hexdigest(), "sha256": hashes}
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        target.parent.mkdir(exist_ok=True)
        if target.exists():
            # --repair is an explicit user action. Do not touch the WebUI or any models.
            shutil.rmtree(target)
        shutil.copytree(staging, target)
    print("Native library installed. Its WebUI scripts are NOT activated.")


def install_tokenizer(repair=False):
    destination = ROOT / "tokenizer"
    receipt = destination / "setup_receipt.json"
    if receipt.is_file() and not repair:
        meta = json.loads(receipt.read_text(encoding="utf-8"))
        for name, checksum in meta["sha256"].items():
            p = destination / name
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != checksum:
                raise RuntimeError("Tokenizer files changed or incomplete. Use --repair only to explicitly replace them.")
        print("Tokenizer already installed and checksums verified.")
        return
    repo = "PrunaAI/P-H3-Image"
    info = json.loads(get(f"https://huggingface.co/api/models/{repo}", limit=8 * 1024**2))
    revision = info["sha"]
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise RuntimeError("Invalid model revision from the repository API.")
    available = {v["rfilename"] for v in info["siblings"]}
    names = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "added_tokens.json"]
    selected = [n for n in names if "tokenizer/" + n in available]
    if "tokenizer_config.json" not in selected or not ("tokenizer.json" in selected or {"vocab.json", "merges.txt"}.issubset(selected)):
        raise RuntimeError("Official tokenizer layout changed. No fallback to an unrelated tokenizer is made.")
    with tempfile.TemporaryDirectory(prefix="ph3-tokenizer-") as temp:
        staging = Path(temp)
        hashes = {}
        for name in selected:
            data = get(f"https://huggingface.co/{repo}/resolve/{revision}/tokenizer/{name}", limit=32 * 1024**2)
            (staging / name).write_bytes(data)
            hashes[name] = hashlib.sha256(data).hexdigest()
        if destination.exists() and any(destination.iterdir()) and not repair:
            raise RuntimeError("Tokenizer folder exists without a valid setup receipt. Refusing to overwrite it.")
        destination.mkdir(exist_ok=True)
        for name in selected:
            shutil.copy2(staging / name, destination / name)
        receipt.write_text(json.dumps({"repository": repo, "revision": revision, "sha256": hashes}, indent=2), encoding="utf-8")
    print("Tokenizer installed. Model weights were NOT downloaded.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair", action="store_true", help="Explicitly replace the extension's vendored source/tokenizer only")
    args = parser.parse_args()
    print("This installs pinned native source + tokenizer inside this extension only.")
    print("No pip, no Forge update, no model weights, no ComfyUI installation/server.")
    install_native(args.repair)
    install_tokenizer(args.repair)
    print("Setup complete. Start Forge Neo and open the P-H3 Image tab; run Diagnostics first.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("SETUP FAILED:", e, file=sys.stderr)
        sys.exit(1)
