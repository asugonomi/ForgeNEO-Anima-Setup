# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit opt-in download of exactly three official P-H3 files, never the whole repository."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ph3_native.contracts import MODEL_REPO


def main():
    default = ROOT.parent.parent / "models" / "P-H3-Image" if ROOT.parent.name == "extensions" else ROOT / "models"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=default)
    parser.add_argument("--accept-model-license", action="store_true")
    parser.add_argument("--precision", choices=["w4a8", "int8"], default="w4a8")
    args = parser.parse_args()
    print("Official model terms: https://huggingface.co/PrunaAI/P-H3-Image-Comfy#license")
    print("This extension is experimental. Its 16 GB GPU inference has NOT been verified.")
    print("The default download is approximately 22 GB, without the optional prompt upsampler.")
    if not args.accept_model_license:
        if input("After reading the model terms, type AGREE to download, or anything else to cancel: ").strip() != "AGREE":
            print("Cancelled. No model weights downloaded.")
            return
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError as e:
        raise RuntimeError("Run this script with your Forge Neo Python (huggingface_hub required). No packages are auto-installed.") from e
    dit = "p-h3-image_w4a8.safetensors" if args.precision == "w4a8" else "p-h3-image_int8_convrot.safetensors"
    files = [f"diffusion_models/{dit}", "text_encoders/p-h3-image_text_encoder_int8_convrot.safetensors",
             "vae/p-h3-image_vae_int8_convrot.safetensors"]
    info = HfApi().model_info(MODEL_REPO, files_metadata=True)
    by_name = {item.rfilename: item for item in info.siblings}
    if any(f not in by_name for f in files):
        raise RuntimeError("Official filenames changed; refusing to download a guessed replacement.")
    size = sum(by_name[f].size or 0 for f in files)
    target = args.target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    missing = sum(by_name[f].size or 0 for f in files if not (target / f).is_file())
    if shutil.disk_usage(target).free < missing * 1.1 + 2 * 1024**3:
        raise RuntimeError("Insufficient free disk space for these model files.")
    print(f"Target: {target}\nRevision: {info.sha}\nSelected size: {size / 10**9:.2f} GB")
    snapshot_download(MODEL_REPO, revision=info.sha, local_dir=str(target),
                      allow_patterns=files + ["README.md", "LICENSE", "LICENSE.txt"], max_workers=2)
    receipt = {"repository": MODEL_REPO, "revision": info.sha,
               "downloaded_at": datetime.now(timezone.utc).isoformat(), "files": files,
               "sizes": {f: by_name[f].size for f in files}}
    (target / "ph3_download_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print("Downloaded only the selected files. Set these three paths in the P-H3 Image tab:")
    for f in files:
        print(target / f)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("DOWNLOAD FAILED:", e, file=sys.stderr)
        sys.exit(1)
