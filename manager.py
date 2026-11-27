"""Windows x64 personal Forge Neo distribution manager (Python 3.13)."""

import hashlib
import json
import msvcrt
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
FORGE = ROOT / "forge"
CACHE = ROOT / "downloads"
STATE = ROOT / "state"
REPO = "asugonomi/ForgeNEO-Anima-Setup"
RAW = f"https://raw.githubusercontent.com/{REPO}/main/"
MANIFEST = json.loads((ROOT / "manifest.json").read_text("utf-8-sig"))
HW = json.loads((ROOT / "hardware.json").read_text("utf-8-sig"))
RELEASE = f"https://github.com/{REPO}/releases/download/{MANIFEST['tag']}/"
BASE = "U-net/anima-base-v1.0.safetensors"
FN = "U-net/AnimaTools/FnMoChIrMrgBl5_int8ConvRot.safetensors"
FN_SHA = "56098d414b8fb58525259a44d89934114a3e996f3e21bf6215ed61fe50e0bacb"
PY = ROOT / "python/python.exe"
LOG = None


def say(message):
    print(message, flush=True)
    if LOG:
        print(message, file=LOG, flush=True)


def read(path):
    return json.loads(path.read_text("utf-8-sig"))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".new")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run(args, *, cwd=ROOT, capture=False):
    args = list(map(str, args))
    if capture:
        return subprocess.check_output(
            args, cwd=cwd, encoding="utf-8", errors="replace"
        ).strip()
    with subprocess.Popen(
        args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    ) as proc:
        try:
            for line in proc.stdout:
                say(line.rstrip("\r\n"))
        except KeyboardInterrupt:
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                proc.terminate()
            raise
        if proc.wait():
            raise RuntimeError(f"{Path(args[0]).name} の終了コード {proc.returncode}")


def remote_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "ForgeNEO-Anima-Setup/1"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def download(item, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if (
        target.is_file()
        and target.stat().st_size == item.get("size", target.stat().st_size)
        and digest(target) == item["sha"]
    ):
        return
    tmp = target.with_suffix(target.suffix + ".part")
    curl = Path(os.environ["SystemRoot"]) / "System32/curl.exe"
    say(f"取得中  {target.name}")
    for attempt in range(3):
        if tmp.exists() and "size" in item and tmp.stat().st_size >= item["size"]:
            if tmp.stat().st_size == item["size"] and digest(tmp) == item["sha"]:
                break
            tmp.unlink()
        command = [
            str(curl),
            "-fLsS",
            "--retry",
            "3",
            "--connect-timeout",
            "20",
            "--speed-limit",
            "1024",
            "--speed-time",
            "120",
            "--proto",
            "=https",
            "--proto-redir",
            "=https",
            "-C",
            "-",
            "-o",
            str(tmp),
            item["url"],
        ]
        with subprocess.Popen(command) as transfer:
            while True:
                try:
                    status = transfer.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    if tmp.exists() and item.get("size"):
                        with tmp.open("rb") as partial:
                            received = partial.seek(0, os.SEEK_END)
                        say(
                            f"  {target.name}  {received / item['size']:.0%}  ({received / 2**20:.0f} MiB)"
                        )
                except KeyboardInterrupt:
                    transfer.terminate()
                    raise
        if not status and digest(tmp) == item["sha"]:
            break
        if not status or status in (33, 36):
            tmp.unlink(missing_ok=True)
        if attempt == 2:
            raise RuntimeError(
                f"{target.name} の取得またはSHA-256照合に失敗しました。BATの再実行で再試行できます。"
            )
    tmp.replace(target)


def unpack(archive, dest, *, keep_existing=False):
    with zipfile.ZipFile(archive) as z:
        for entry in z.infolist():
            file = (dest / entry.filename).resolve()
            if (
                not file.is_relative_to(dest.resolve())
                or ":" in entry.filename
                or ".." in Path(entry.filename).parts
            ):
                raise ValueError(entry.filename)
            if entry.is_dir():
                file.mkdir(parents=True, exist_ok=True)
                continue
            if keep_existing and file.exists():
                continue
            file.parent.mkdir(parents=True, exist_ok=True)
            tmp = file.with_name(file.name + ".setup-part")
            with z.open(entry) as src, tmp.open("wb") as out:
                shutil.copyfileobj(src, out, 1 << 20)
            tmp.replace(file)


def asset(item):
    marker = STATE / (item["name"] + ".json")
    if marker.is_file() and read(marker) == item["sha"]:
        return
    cached = CACHE / item["name"]
    download(dict(item, url=RELEASE + item["name"]), cached)
    dest = {
        "python-packages.zip": ROOT / "python",
        "presets.zip": ROOT / "defaults",
    }.get(item["name"], FORGE)
    say(f"展開中  {item['name']}")
    unpack(cached, dest)
    write(marker, item["sha"])
    cached.unlink()


def git_tool():
    git = shutil.which("git") or str(ROOT / "tools/git/cmd/git.exe")
    if not Path(git).is_file():
        item = HW["git"]
        zip_path = CACHE / "mingit.zip"
        download(dict(url=item["url"], sha=item["sha256"]), zip_path)
        unpack(zip_path, ROOT / "tools/git")
    os.environ["GIT_PYTHON_GIT_EXECUTABLE"] = git
    os.environ["GIT"] = git
    os.environ["PATH"] = str(Path(git).parent) + os.pathsep + os.environ["PATH"]
    return git


def hardware():
    ps = (
        Path(os.environ["SystemRoot"])
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    script = "[Console]::OutputEncoding=[Text.UTF8Encoding]::new($false); $ErrorActionPreference='Stop'; [pscustomobject]@{gpu=@(Get-CimInstance Win32_VideoController|% Name);cpu=@(Get-CimInstance Win32_Processor|% Name)}|ConvertTo-Json -Compress"
    try:
        return json.loads(
            run([ps, "-NoProfile", "-NonInteractive", "-Command", script], capture=True)
        )
    except (subprocess.SubprocessError, ValueError):
        return {"gpu": [], "cpu": [platform.processor()]}


def choices(gpus, build):
    names = " ".join(gpus)
    if re.search("NVIDIA", names, re.I) or shutil.which("nvidia-smi"):
        yield "cu130"
        yield "cu126"
    if re.search(r"Intel.*(?:Arc|Graphics)", names, re.I):
        yield "xpu"
    if build >= 26200:
        for gpu in gpus:
            gpu = gpu.replace("(TM)", "").replace("(tm)", "").strip()
            for profile in HW["gpus"]:
                if re.fullmatch(profile["pattern"], gpu, re.I):
                    yield "rocm-" + profile["arch"]
                    break


def gpu_candidates():
    info = hardware()
    say("CPU  " + " / ".join(info["cpu"]) + "\nGPU  " + " / ".join(info["gpu"]))
    candidates = list(
        dict.fromkeys(choices(info["gpu"], sys.getwindowsversion().build))
    )
    if not candidates:
        raise RuntimeError(
            "対応GPUを検出できませんでした。GPUとドライバーを確認してください。"
        )
    return candidates


def saved_backend():
    path = STATE / "backend.json"
    if not path.is_file():
        return None
    backend = read(path)
    if backend == "cpu" and not (STATE / "cpu-manual.json").is_file():
        return None
    return backend


def envpython(backend):
    active = STATE / ("runtime-" + backend + ".json")
    if active.exists():
        return ROOT / "runtimes" / read(active) / "Scripts/python.exe"
    return ROOT / "runtimes" / backend / "Scripts/python.exe"


def probe(python, backend):
    device = "cpu" if backend == "cpu" else "xpu" if backend == "xpu" else "cuda"
    code = (
        "import torch,torchvision; from torchvision.ops import nms; "
        f"d='{device}'; a=torch.ones((8,8),device=d); assert (a@a).sum().item()==512; "
        "nms(torch.tensor([[0.,0.,1.,1.]],device=d),torch.ones(1,device=d),0.5); "
        "print(torch.__version__,d,flush=True)"
    )
    run([python, "-c", code])


def configure_backend(candidates):
    for backend in candidates:
        python = envpython(backend)
        runtime = python.parents[1]
        created_here = not runtime.exists()
        try:
            say(f"演算テスト  {backend}")
            if not python.exists():
                run(
                    [
                        PY,
                        "-m",
                        "venv",
                        "--without-pip",
                        "--system-site-packages",
                        python.parents[1],
                    ]
                )
            marker = python.parents[1] / "installed.json"
            if not marker.exists():
                pip = [
                    python,
                    "-m",
                    "pip",
                    "--isolated",
                    "install",
                    "--disable-pip-version-check",
                    "--no-cache-dir",
                ]
                if backend.startswith("rocm-"):
                    extra = "device-" + backend.split("-", 1)[1]
                    run(
                        [
                            *pip,
                            "--index-url",
                            HW["amdIndex"],
                            f"torch[{extra}]=={HW['torch']}",
                            f"torchvision[{extra}]=={HW['torchvision']}",
                        ]
                    )
                else:
                    args = [
                        "--index-url",
                        f"https://download.pytorch.org/whl/{backend}",
                    ]
                    if backend != "xpu":
                        args += ["--no-deps"]
                    run([*pip, *args, "torch==2.11.0", "torchvision==0.26.0"])
                write(marker, True)
            probe(python, backend)
            consent = STATE / "cpu-manual.json"
            if backend == "cpu":
                write(consent, True)
            else:
                consent.unlink(missing_ok=True)
            write(STATE / "backend.json", backend)
            return backend
        except (RuntimeError, OSError) as err:
            say(f"{backend} を利用できません。{err}")
            if (
                created_here
                and runtime.exists()
                and runtime.resolve().parent == (ROOT / "runtimes").resolve()
            ):
                try:
                    shutil.rmtree(runtime)
                except OSError as cleanup_error:
                    say(f"使用中の一時環境を消去できません　{cleanup_error}")
    raise RuntimeError(
        "指定した演算デバイスを使用できません。logsフォルダーを確認してください。"
    )


def fn_url():
    local = ROOT / "models.json"
    value = read(local).get("url", "") if local.exists() else ""
    if not value:
        try:
            value = remote_json(RAW + "models.json").get("url", "")
        except (OSError, ValueError) as err:
            say(f"FnMoの配布設定を照会できませんでした。{err}")
    return validate_model_url(value) if value else ""


def validate_model_url(value):
    uri = urlparse(value.strip())
    if (
        uri.scheme != "https"
        or uri.netloc != "huggingface.co"
        or uri.fragment
        or uri.query not in ("", "download=true")
    ):
        raise ValueError("Hugging Faceの公開ファイルのURLを指定してください。")
    path = uri.path.replace("/blob/", "/resolve/", 1)
    if not re.fullmatch(r"/[^/]+/[^/]+/resolve/[^/]+/.+\.safetensors", path, re.I):
        raise ValueError("Hugging FaceのsafetensorsファイルのURLを指定してください。")
    return "https://huggingface.co" + path


def sync_models():
    model = FORGE / "models/Stable-diffusion"
    for item in MANIFEST["hf"]:
        folder = (
            model / "U-net"
            if item["name"].startswith("anima-base")
            else FORGE
            / "models"
            / ("VAE" if "_vae" in item["name"] else "text_encoder")
        )
        download(item, folder / item["name"])
    path = model / FN
    if not (path.is_file() and digest(path) == FN_SHA):
        url = fn_url()
        if url:
            download(dict(url=url, sha=FN_SHA, size=3082496712), path)
        elif path.exists():
            raise ValueError("配置済みFnMoが原本のSHA-256と異なります。")
        else:
            say("FnMoのURLが未登録です。Anima Base v1.0を使用できます。")
    for item in MANIFEST["tipo"]:
        download(item, FORGE / "extensions/z-tipo-extension/models" / item["name"])
    return path.exists()


def settings(has_fn):
    for file in ("config.json", "ui-config.json"):
        target = FORGE / file
        if target.exists():
            continue
        source = ROOT / "defaults" / file
        cfg = read(source)
        for k, v in list(cfg.items()):
            if isinstance(v, list):
                cfg[k] = [
                    str(FORGE / "models" / x[8:])
                    if isinstance(x, str) and x.startswith("@models/")
                    else x
                    for x in v
                ]
            elif isinstance(v, str) and re.match(r"(?i)^([a-z]:[\\/])", v):
                cfg[k] = (
                    str(FORGE / "outputs" / Path(v).name)
                    if k.startswith("outdir") or k.endswith("output_dir")
                    else ""
                )
        if file == "config.json":
            cfg.pop("sd_checkpoint_hash", None)
            cfg["sd_model_checkpoint"] = cfg["forge_checkpoint_anima"] = str(
                Path(FN if has_fn else BASE)
            )
        write(target, cfg)


def install(*, cpu_requested=False):
    first = not (STATE / "installed.json").exists()
    if (
        first
        and not any(STATE.glob("*.zip.json"))
        and shutil.disk_usage(ROOT).free < 24 * 1024**3
    ):
        raise OSError("セットアップ開始時は空き容量24 GiB以上が必要です。")
    selected = saved_backend()
    if cpu_requested and selected != "cpu":
        selected = None
    if selected is None:
        candidates = ["cpu"] if cpu_requested else gpu_candidates()
    git_tool()
    for item in MANIFEST["assets"]:
        asset(item)
    if selected is None:
        selected = configure_backend(candidates)
    probe(envpython(selected), selected)
    cuda = selected == "cu130"
    tag = "v0.3.34-cu130" if cuda else "v0.3.34"
    sha = (
        "0ed0e5b47e9b0bec115edd30ff228763e2f82bad4913674aa2410189b9fe5fed"
        if cuda
        else "6526fff614e5ef7e439e6369e076a78073e45e1d791dbe1d5e5d42661f46ca1a"
    )
    wheel = (
        f"https://github.com/abetlen/llama-cpp-python/releases/download/{tag}/"
        f"llama_cpp_python-0.3.34-py3-none-win_amd64.whl#sha256={sha}"
    )
    installed = run(
        [
            envpython(selected),
            "-c",
            "import importlib.util; print(int(importlib.util.find_spec('llama_cpp') is not None))",
        ],
        capture=True,
    )
    if installed == "0":
        run(
            [
                envpython(selected),
                "-m",
                "pip",
                "--isolated",
                "install",
                "--no-cache-dir",
                "--no-deps",
                wheel,
            ]
        )
    has_fn = sync_models()
    settings(has_fn)
    # Some extensions use Ultralytics as an alternative name for ADetailer.
    alias = FORGE / "models/Ultralytics"
    alias.mkdir(exist_ok=True)
    for src in (FORGE / "models/adetailer").rglob("*.pt"):
        dst = alias / src.relative_to(FORGE / "models/adetailer")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.link(src, dst)
            except OSError:
                shutil.copy2(src, dst)
    run(
        [
            envpython(selected),
            "-c",
            "import gradio,fastapi,cv2,dynamicprompts,ultralytics; print('Forge dependencies OK')",
        ]
    )
    write(STATE / "installed.json", MANIFEST["version"])
    say("セットアップ完了。ForgeNEO.exe をダブルクリックしてください。")


def launch():
    git_tool()
    backend = saved_backend()
    if backend is None:
        raise RuntimeError(
            "GPUか手動指定のCPUを設定してください。Install-ForgeNEO.bat を再実行してください。"
        )
    python = envpython(backend)
    try:
        probe(python, backend)
    except (RuntimeError, OSError) as err:
        raise RuntimeError(
            f"{backend} の演算テストに失敗しました。GPUの再検出は更新メニューの「4」で実行できます。"
        ) from err
    flags = ["--api", "--skip-prepare-environment"]
    if os.environ.get("FORGENEO_NO_BROWSER") != "1":
        flags.append("--autolaunch")
    else:
        os.environ["SD_WEBUI_RESTARTING"] = "1"
    if os.environ.get("FORGENEO_PORT"):
        flags.extend(["--port", str(int(os.environ["FORGENEO_PORT"]))])
    if backend == "cpu":
        flags += ["--cpu", "--force-fp32", "--skip-torch-cuda-test"]
    # The venv is selected explicitly for subprocesses spawned by extensions.
    os.environ["VIRTUAL_ENV"] = str(python.parents[1])
    os.environ["PATH"] = str(python.parent) + os.pathsep + os.environ["PATH"]
    run([python, "-u", "launch.py", *flags], cwd=FORGE)


def menu():
    from updater import update, rollback

    while True:
        say(
            "\n1  Forge本体を更新\n2  拡張機能を更新\n3  Hugging Faceモデルを同期\n4  GPUを再検出\n5  ソフトウェア更新を1件戻す\n6  FnMoのHugging Face URLを登録\n0  終了"
        )
        choice = input("番号を入力 > ").strip()
        if choice == "0":
            return
        if choice == "1":
            update(False)
        if choice == "2":
            update(True)
        if choice == "3":
            has_fn = sync_models()
            if has_fn:
                cfg = read(FORGE / "config.json")
                if cfg.get("forge_checkpoint_anima") == str(Path(BASE)):
                    cfg["forge_checkpoint_anima"] = str(Path(FN))
                    if cfg.get("forge_preset") == "anima":
                        cfg["sd_model_checkpoint"] = str(Path(FN))
                        cfg.pop("sd_checkpoint_hash", None)
                    write(FORGE / "config.json", cfg)
            say("モデル同期完了")
        if choice == "4":
            configure_backend(gpu_candidates())
        if choice == "5":
            rollback()
        if choice == "6":
            value = input("FnMoの公開ファイルURL > ").strip()
            if value:
                data = read(ROOT / "models.json")
                data["url"] = validate_model_url(value)
                write(ROOT / "models.json", data)
                say("登録しました。3を選ぶとモデルを同期します。")


def main():
    global LOG
    sys.modules["manager"] = sys.modules[__name__]
    for k in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "COMMANDLINE_ARGS",
        "TORCH_COMMAND",
    ):
        os.environ.pop(k, None)
    os.environ.update(
        PYTHONNOUSERSITE="1",
        PYTHONUTF8="1",
        PYTHONIOENCODING="utf-8",
        PIP_CONFIG_FILE=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GRADIO_ANALYTICS_ENABLED="False",
        NO_COLOR="1",
        TEMP=str(ROOT / "temp"),
        TMP=str(ROOT / "temp"),
        HF_HOME=str(ROOT / "cache/huggingface"),
        TORCH_HOME=str(ROOT / "cache/torch"),
        YOLO_CONFIG_DIR=str(ROOT / "cache/ultralytics"),
        MPLCONFIGDIR=str(ROOT / "cache/matplotlib"),
    )
    for folder in [STATE, CACHE, ROOT / "logs", ROOT / "temp"]:
        folder.mkdir(exist_ok=True)
    action = sys.argv[1] if len(sys.argv) > 1 else "launch"
    with (
        (ROOT / "logs" / (time.strftime("%Y%m%d-%H%M%S") + "-" + action + ".log")).open(
            "a", encoding="utf-8"
        ) as f,
        (STATE / "session.lock").open("a+b") as lock,
    ):
        LOG = f
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            say("この環境を使用中です。起動または更新中のウィンドウを閉じてください。")
            return 2
        try:
            from updater import recover

            recover()
            if action == "install":
                install(cpu_requested=sys.argv[2:] == ["--cpu"])
            elif not (STATE / "installed.json").exists():
                raise RuntimeError("先にInstall-ForgeNEO.batを実行してください。")
            elif action == "launch":
                launch()
            elif action == "update":
                menu()
            else:
                raise ValueError(action)
        except KeyboardInterrupt:
            say("中断しました。")
            return 130
        except Exception as err:
            import traceback

            traceback.print_exc(file=f)
            say(f"エラー　{err}\nログ　{f.name}")
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
