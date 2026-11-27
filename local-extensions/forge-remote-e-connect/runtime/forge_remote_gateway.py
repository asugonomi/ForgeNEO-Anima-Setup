"""One-command launcher for the opt-in LAN/Tailscale gateway."""
import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import shutil
import socket
import ssl
import subprocess
import sys
import threading
import time

from gateway_discovery import announce
from gateway_http import GatewayServer
from pairing_card import show_pairing_card

LAN = tuple(map(ipaddress.ip_network, ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")))
TAIL = (ipaddress.ip_network("100.64.0.0/10"),)


def in_network(address, networks):
    try:
        value = ipaddress.IPv4Address(address)
    except ValueError:
        return False
    return any(value in network and value not in (network.network_address,network.broadcast_address)
               for network in networks)


def _stdout(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=8, check=True)
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def physical_lan():
    if os.name=="nt":
        # Limit automatic selection to active physical adapters. An explicit override may select another LAN adapter.
        query = ("Get-NetAdapter -Physical | Where-Object { $_.Status -eq 'Up' } | ForEach-Object { "
                 "Get-NetIPAddress -AddressFamily IPv4 -InterfaceIndex $_.InterfaceIndex "
                 "-ErrorAction SilentlyContinue | Where-Object { $_.AddressState -eq 'Preferred' } "
                 "| Select-Object -ExpandProperty IPAddress }")
        powershell = Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0/powershell.exe"
        return sorted({ip for ip in _stdout([str(powershell), "-NoProfile", "-NonInteractive", "-Command", query]).splitlines()
                       if in_network(ip, LAN)})
    try:
        hosts = socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
    except OSError:
        return []
    return sorted({r[4][0] for r in hosts if in_network(r[4][0], LAN)})


def tailscale_address():
    cmd = shutil.which("tailscale")
    if not cmd and os.name=="nt":
        installed = Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Tailscale/tailscale.exe"
        if installed.is_file():
            cmd = str(installed)
    address = _stdout([cmd, "ip", "-4"]) if cmd else ""
    return address if in_network(address, TAIL) else None


def bind_addresses(explicit_lan, no_lan, no_tail):
    if no_lan and explicit_lan:
        raise ValueError("Choose either -LanAddress or -NoLan.")
    if explicit_lan and not in_network(explicit_lan, LAN):
        raise ValueError("Choose the private IPv4 address of the PC's home network adapter.")
    lans = [] if no_lan else ([explicit_lan] if explicit_lan else physical_lan())
    if len(lans)>1:
        raise ValueError("Several home adapters are available. Pass -LanAddress with one of: " + ", ".join(lans))
    lan = next(iter(lans), None)
    tail = None if no_tail else tailscale_address()
    if not (lan or tail):
        raise ValueError("No usable address. Connect the PC to a home network and pass -LanAddress, or sign in to Tailscale.")
    return lan, tail


def _companion_ready(port):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
    try:
        conn.request("GET", "/forge-remote/v1/capabilities")
        result = conn.getresponse()
        body = result.read(32769)
        if result.status!=200 or len(body)>32768:
            return False
        cap = json.loads(body)
        return (isinstance(cap,dict) and cap.get("version")==1 and cap.get("gallery") is True
                and "tag_dictionary" in cap and "tag_status" in cap)
    except (OSError, http.client.HTTPException, ValueError):
        return False
    finally:
        conn.close()


@contextmanager
def local_companion(args, *, start_process=None):
    if _companion_ready(args.companion_port):
        print("Using the running E companion on loopback.")
        yield
        return
    with socket.socket() as probe:
        probe.settimeout(1)
        if probe.connect_ex(("127.0.0.1", args.companion_port))==0:
            raise ValueError("The companion port is already occupied. Check the existing companion or choose -CompanionPort.")
    cmd = [sys.executable, "-B", str(Path(__file__).with_name("forge_remote_api.py")), "--root", str(args.root),
           "--port", str(args.companion_port), "--forge-port", str(args.forge_port)]
    for flag in ("data_root", "config", "tag_extension", "ar_extension"):
        value = getattr(args, flag)
        if value:
            cmd.extend(["--"+flag.replace("_","-"), str(value)])
    for item in args.lora_root:
        cmd.extend(["--lora-root", str(item)])
    process = (start_process or subprocess.Popen)(cmd, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError("Companion startup failed. Check Forge --api, Forge root and ports.")
            if _companion_ready(args.companion_port):
                break
            time.sleep(0.25)
        else:
            raise RuntimeError("The local companion did not become ready.")
        yield
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def active_identity(address, port, expected):
    # No HTTP data is sent; compare the presented leaf with our locally stored certificate.
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    try:
        with socket.create_connection((address, port), timeout=1.5) as raw:
            with ctx.wrap_socket(raw, server_hostname=address) as tls:
                return hashlib.sha256(tls.getpeercert(binary_form=True)).hexdigest()==expected
    except OSError:
        return False


def display_invite(state, origins, no_open):
    code = state.invite(origins, socket.gethostname())
    path = show_pairing_card(code, state.directory, open_browser=not no_open)
    print("Pairing card for one phone (15 minutes):", path)


def default_state():
    if os.name=="nt":
        return Path(os.environ["LOCALAPPDATA"]) / "ForgeRemoteE/Gateway"
    return Path(os.environ.get("XDG_STATE_HOME", Path.home()/".local/state")) / "forge-remote-e/gateway"


def run(args, stop_event=None):
    from gateway_state import GatewayState
    if len({args.port,args.companion_port,args.forge_port})!=3 or not all(
        1<=p<=65535 for p in (args.port,args.companion_port,args.forge_port)):
        raise ValueError("Gateway, companion and Forge require three different ports in 1..65535.")
    state = GatewayState(args.state_dir or default_state())
    if args.list_devices:
        devices = state.devices()
        for identifier, name in devices:
            print(identifier, name)
        if not devices:
            print("No devices paired.")
        return
    if args.revoke:
        if not state.revoke(args.revoke):
            raise ValueError("That device ID was not registered.")
        print("Device revoked.")
        return
    lan, tail = bind_addresses(args.lan_address, args.no_lan, args.no_tailscale)
    addresses = [ip for ip in (lan,tail) if ip]
    for ip in addresses:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as check:
            try:
                check.bind((ip, 0))
            except OSError as exc:
                raise ValueError("The selected IPv4 address is not available on this PC: " + ip) from exc
    active = [ip for ip in addresses if active_identity(ip,args.port,state.pin)]
    if active:
        if args.pair:
            display_invite(state,[f"https://{ip}:{args.port}" for ip in active],args.no_open)
        print("This gateway is already running. To change interfaces, stop it in the original window and start it again.")
        return
    ports = {"forge":args.forge_port, "companion":args.companion_port}
    context = state.tls()
    slots = threading.BoundedSemaphore(24)
    origins = [f"https://{ip}:{args.port}" for ip in addresses]
    with ExitStack() as stack:
        # Bind all selected addresses before starting any worker or helper.
        servers = [stack.enter_context(GatewayServer((ip,args.port),state,context,ports,slots)) for ip in addresses]
        stack.enter_context(local_companion(args))
        for server in servers:
            threading.Thread(target=server.serve_forever, daemon=True).start()
            stack.callback(server.shutdown)
        stack.enter_context(announce(state.id, lan, args.port))
        print("Secure gateway listening:", ", ".join(origins), flush=True)
        if args.pair or not state.devices():
            display_invite(state,origins,args.no_open)
        print("Leave this window open. Press Ctrl+C to stop.",flush=True)
        stop_event = stop_event or threading.Event()
        try:
            while not stop_event.wait(1):
                continue
        except KeyboardInterrupt:
            pass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--port",type=int,default=17873)
    p.add_argument("--companion-port",type=int,default=17872)
    p.add_argument("--forge-port",type=int,default=7860)
    p.add_argument("--lan-address")
    p.add_argument("--no-lan",action="store_true")
    p.add_argument("--no-tailscale",action="store_true")
    p.add_argument("--pair",action="store_true")
    p.add_argument("--no-open",action="store_true")
    p.add_argument("--state-dir",type=Path)
    admin = p.add_mutually_exclusive_group()
    admin.add_argument("--list-devices",action="store_true")
    admin.add_argument("--revoke")
    for flag in ("--data-root","--config","--tag-extension","--ar-extension"):
        p.add_argument(flag)
    p.add_argument("--lora-root",action="append",default=[])
    try:
        run(p.parse_args())
    except (ValueError, RuntimeError, OSError) as exc:
        p.exit(2, str(exc)+"\n")


if __name__=="__main__":
    main()
