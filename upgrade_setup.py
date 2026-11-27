"""Apply the setup manager to an existing 1.x installation."""

import msvcrt
import os
import shutil
import sys
from pathlib import Path


def upgrade(root):
    source = Path(__file__).resolve().parent
    if not (root / ".setup-v1").is_file():
        raise RuntimeError("ForgeNEO-Anima のインストール先を確認してください。")
    state = root / "state"
    state.mkdir(exist_ok=True)
    with (state / "session.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as err:
            raise RuntimeError(
                "ForgeNEOの起動・更新を終了してからBATを再実行してください。"
            ) from err
        for name in (
            "defaults/config.json",
            "defaults/ui-config.json",
            "manager.py",
            "manifest.json",
        ):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".setup-new")
            shutil.copyfile(source / name, temp)
            os.replace(temp, target)
        (root / ".setup-v1.1.0").touch()


if __name__ == "__main__":
    upgrade(Path(sys.argv[1]).resolve())
