"""Start the paired Android connection with this Forge process."""
import atexit
from pathlib import Path
import subprocess
import sys

from modules import script_callbacks, shared
from modules.paths_internal import script_path

directory = Path(__file__).resolve().parents[1]
child = None


def start(demo, app):
    global child
    if child is not None and child.poll() is None:
        return
    port = getattr(demo, "server_port", None) or shared.cmd_opts.port or 7860
    child = subprocess.Popen(
        [sys.executable, "-B", "-u", str(directory / "worker.py"), script_path, str(port)],
        stdin=subprocess.PIPE, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def stop():
    global child
    if child is not None:
        child.stdin.close()
        child = None


script_callbacks.on_app_started(start)
script_callbacks.on_script_unloaded(stop)
atexit.register(stop)
