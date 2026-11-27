"""Update independent repositories while preserving local changes and user data."""

import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

from manager import (
    FORGE,
    MANIFEST,
    PY,
    REPO,
    ROOT,
    STATE,
    digest,
    envpython,
    git_tool,
    probe,
    read,
    run,
    say,
    write,
)


def blob(repo, ref, path):
    entry = subprocess.check_output(
        [git_tool_path, "-C", repo, "ls-tree", "-z", ref, "--", path]
    )
    if not entry:
        return None
    mode, kind, sha = entry.split(b"\t", 1)[0].split()
    if mode not in (b"100644", b"100755") or kind != b"blob":
        raise ValueError(f"通常ファイル以外の更新を保留しました　{path}")
    return subprocess.check_output(
        [git_tool_path, "-C", repo, "cat-file", "blob", sha.decode()]
    )


def equivalent(a, b):
    if a == b:
        return True
    return (
        a is not None
        and b is not None
        and b"\0" not in a
        and a.replace(b"\r\n", b"\n") == b.replace(b"\r\n", b"\n")
    )


def destination(folder, name):
    p = (folder / name).resolve()
    if (
        not p.is_relative_to(folder.resolve())
        or ".git" in Path(name).parts
        or ":" in name
        or Path(name).is_absolute()
    ):
        raise ValueError(name)
    return p


def stage_runtime(journal, folder, meta, changed):
    backend = meta["backend"]
    previous = envpython(backend)
    trial = ROOT / "runtimes" / meta["trial"]
    run([PY, "-m", "venv", "--without-pip", "--system-site-packages", trial])
    python = trial / "Scripts/python.exe"
    paths = json.loads(
        run(
            [
                previous,
                "-c",
                "import sys,json; print(json.dumps([p for p in sys.path if p.lower().endswith('site-packages')]))",
            ],
            capture=True,
        )
    )
    site = trial / "Lib/site-packages"
    (site / "previous.pth").write_text("\n".join(paths) + "\n", "utf-8")
    versions = run(
        [
            previous,
            "-c",
            "import torch,torchvision;print('torch=='+torch.__version__+'\\ntorchvision=='+torchvision.__version__)",
        ],
        capture=True,
    )
    constraints = journal / "protected.txt"
    constraints.write_text(versions + "\n", "utf-8")
    old_env = os.environ.copy()
    try:
        os.environ.update(
            PIP_CONSTRAINT=str(constraints),
            VIRTUAL_ENV=str(trial),
            PYTHONPATH=str(FORGE),
            PATH=str(python.parent) + os.pathsep + old_env["PATH"],
        )
        for name in changed:
            if Path(name).name == "requirements.txt" and (folder / name).is_file():
                run(
                    [
                        python,
                        "-m",
                        "pip",
                        "install",
                        "--disable-pip-version-check",
                        "--no-cache-dir",
                        "-r",
                        str(folder / name),
                    ],
                    cwd=folder,
                )
        if "install.py" in changed and (folder / "install.py").is_file():
            run([python, "install.py"], cwd=folder)
        probe(python, backend)
        run(
            [
                python,
                "-c",
                "import gradio,fastapi,ultralytics,cv2,transformers,diffusers",
            ]
        )
    finally:
        os.environ.clear()
        os.environ.update(old_env)
    write(trial / "installed.json", True)
    write(STATE / ("runtime-" + backend + ".json"), trial.name)


def replace(path, data):
    if data is None:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".update-part")
    tmp.write_bytes(data)
    tmp.replace(path)


def apply(label, folder, changes, old, new):
    journal = (
        ROOT
        / "backups"
        / (
            time.strftime("%Y%m%d-%H%M%S")
            + "-"
            + str(time.time_ns())
            + "-"
            + folder.name
        )
    )
    journal.mkdir(parents=True)
    entries = []
    for name, data in changes:
        p = destination(folder, name)
        before = p.read_bytes() if p.exists() else None
        if before is not None:
            target = journal / "before" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(before)
        entries.append(
            dict(
                name=name,
                before=before is not None,
                oldsha=hashlib.sha256(before).hexdigest()
                if before is not None
                else None,
                sha=hashlib.sha256(data).hexdigest() if data is not None else None,
            )
        )
    meta = dict(
        label=label,
        folder=folder.relative_to(FORGE).as_posix(),
        old=old,
        new=new,
        entries=entries,
    )
    changed = [
        name
        for name, _ in changes
        if name == "install.py" or Path(name).name == "requirements.txt"
    ]
    if changed:
        backend = read(STATE / "backend.json")
        meta.update(
            backend=backend,
            previous=envpython(backend).parents[1].name,
            trial=backend + "-update-" + str(time.time_ns()),
        )
    write(journal / "journal.json", meta)
    write(STATE / "pending.json", journal.name)
    for name, data in changes:
        replace(destination(folder, name), data)
    if changed:
        stage_runtime(journal, folder, meta, changed)
    revs = read(STATE / "revisions.json") if (STATE / "revisions.json").exists() else {}
    revs[label] = new
    write(STATE / "revisions.json", revs)
    (STATE / "pending.json").unlink()


def restore(journal):
    m = read(journal / "journal.json")
    root = destination(FORGE, m["folder"])
    for e in m["entries"]:
        file = destination(root, e["name"])
        current = digest(file) if file.exists() else None
        if current not in (e["sha"], e["oldsha"]):
            raise RuntimeError(
                f"更新後に編集されています。復元前に退避してください　{file}"
            )
    write(STATE / "pending.json", journal.name)
    if "backend" in m:
        write(STATE / ("runtime-" + m["backend"] + ".json"), m["previous"])
    for e in m["entries"]:
        source = journal / "before" / e["name"]
        replace(
            destination(root, e["name"]), source.read_bytes() if e["before"] else None
        )
    revisions = (
        read(STATE / "revisions.json") if (STATE / "revisions.json").exists() else {}
    )
    revisions[m["label"]] = m["old"]
    write(STATE / "revisions.json", revisions)
    (journal / "restored").touch()
    (STATE / "pending.json").unlink()
    say("復元しました　" + m["label"])


def recover():
    pending = STATE / "pending.json"
    if pending.exists():
        restore(ROOT / "backups" / read(pending))


def rollback():
    recover()
    folders = sorted((ROOT / "backups").glob("*/journal.json"), reverse=True)
    for entry in folders:
        if not (entry.parent / "restored").exists():
            restore(entry.parent)
            return
    say("復元できる更新履歴はありません。")


def update_repo(spec, revisions, work):
    folder = destination(FORGE, spec["path"])
    label = spec["path"]
    previous = revisions.get(label, spec["commit"])
    url = spec["url"]
    branch = "refs/heads/" + spec["branch"]
    args = [
        git_tool_path,
        "-c",
        "credential.interactive=false",
        "-c",
        "core.longpaths=true",
    ]
    result = run([*args, "ls-remote", "--exit-code", url, branch], capture=True)
    current = result.split()[0]
    if previous == current:
        return
    cache = work / ("repo-" + folder.name)
    run([*args, "init", "--bare", "-q", cache])
    git = [*args, "-C", cache]
    run([*git, "fetch", "--quiet", "--depth=1", url, previous, current])
    prefix = spec.get("prefix", "")
    raw = subprocess.check_output(
        [
            *git,
            "diff",
            "--no-renames",
            "--name-only",
            "-z",
            previous,
            current,
            "--",
            prefix or ".",
        ]
    )
    changes = []
    for encoded in raw.split(b"\0"):
        if not encoded:
            continue
        upstream = encoded.decode("utf-8")
        name = upstream[len(prefix) :] if prefix else upstream
        if name.startswith((".github/", ".git/")):
            continue
        file = destination(folder, name)
        old = blob(cache, previous, upstream)
        new = blob(cache, current, upstream)
        actual = file.read_bytes() if file.is_file() else None
        if equivalent(actual, new):
            continue
        if not equivalent(actual, old):
            raise RuntimeError(
                f"ローカル変更と競合しました。元の版を使用します　{name}"
            )
        if new is not None and name.endswith(".py"):
            compile(new, name, "exec")
        changes.append((name, new))
    apply(label, folder, changes, previous, current)
    say(f"更新済み　{label}　{previous[:8]} → {current[:8]}")


def update(extensions):
    global git_tool_path
    git_tool_path = git_tool()
    recover()
    specs = [s for s in MANIFEST["repos"] if (s["path"] != ".") == extensions]
    revisions = (
        read(STATE / "revisions.json") if (STATE / "revisions.json").exists() else {}
    )
    if extensions:
        for name in MANIFEST["localExtensions"]:
            specs.append(
                dict(
                    path="extensions/" + name,
                    url="https://github.com/" + REPO + ".git",
                    branch="main",
                    commit=MANIFEST["localBaseline"],
                    prefix="local-extensions/" + name + "/",
                )
            )
    failed = []
    with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
        for spec in specs:
            say("確認中　" + spec["path"])
            try:
                update_repo(spec, revisions, Path(tmp))
            except Exception as err:
                recover()
                failed.append(spec["path"])
                say(str(err))
    say(
        "更新確認完了"
        + ("　保留 " + ", ".join(failed) if failed else "　すべて最新です。")
    )
