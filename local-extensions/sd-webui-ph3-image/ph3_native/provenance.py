# SPDX-License-Identifier: AGPL-3.0-or-later
"""Isolated import of a commit-pinned native H3 library, not its WebUI scripts."""
from __future__ import annotations
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from .contracts import ROOT, UPSTREAM_COMMIT, PH3Error

NAMESPACE = "_ph3_frozen_h3"
VENDOR = ROOT / "_vendor" / "h3"


def verify_vendor() -> dict:
    manifest = VENDOR / "manifest.json"
    if not manifest.is_file():
        raise PH3Error("ネイティブ推論ライブラリが未設定です。setup_native.batを実行してください。")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("commit") != UPSTREAM_COMMIT:
        raise PH3Error("ネイティブ推論ライブラリの固定コミットが一致しません。")
    entries = data.get("sha256", {})
    if "forge_h3/native/transformer.py" not in entries:
        raise PH3Error("推論ライブラリのマニフェストが不完全です。")
    base = VENDOR.resolve()
    for name, digest in entries.items():
        p = (VENDOR / name).resolve()
        if not p.is_relative_to(base) or not p.is_file():
            raise PH3Error(f"ライブラリが欠けています: {name}")
        if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            raise PH3Error(f"ライブラリが変更されています: {name}。自動修復は行いません。")
    return data


def native_module(name: str):
    if NAMESPACE not in sys.modules:
        verify_vendor()
        package_dir = VENDOR / "forge_h3"
        spec = importlib.util.spec_from_file_location(NAMESPACE, package_dir / "__init__.py",
                                                     submodule_search_locations=[str(package_dir)])
        if spec is None or spec.loader is None:
            raise PH3Error("ネイティブライブラリの読込準備に失敗しました。")
        module = importlib.util.module_from_spec(spec)
        sys.modules[NAMESPACE] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            sys.modules.pop(NAMESPACE, None)
            raise
    return importlib.import_module(f"{NAMESPACE}.native.{name}")
