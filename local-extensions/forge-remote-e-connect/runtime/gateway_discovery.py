"""Advertise a paired PC's identity on its selected LAN interface (DNS-SD).

An advertisement is only an address hint. Android authenticates the PC by its
stored TLS certificate before sending any device credential.
"""
from contextlib import contextmanager
import ctypes
import ipaddress
from pathlib import Path
import socket
import subprocess
import sys
import threading
import uuid

SERVICE = "_forgeremote._tcp.local."


def _names(server_id):
    short = uuid.UUID(str(server_id)).hex[:12]
    return "ForgeRemote-" + short + "." + SERVICE, "forge-remote-" + short + ".local."


def _windows(server_id, address, port):
    """Run WinDNS in a child. Windows ties this registration to process lifetime."""
    from ctypes import wintypes as w

    api = ctypes.WinDLL("dnsapi")
    helper = ctypes.WinDLL("iphlpapi")
    ip = w.DWORD.from_buffer_copy(socket.inet_aton(address))
    adapter = w.DWORD()
    helper.GetBestInterface.argtypes = [w.DWORD, ctypes.POINTER(w.DWORD)]
    if helper.GetBestInterface(ip, ctypes.byref(adapter)) or not adapter.value:
        raise OSError("LAN interface unavailable for discovery")
    callback_type = ctypes.WINFUNCTYPE(None, w.DWORD, ctypes.c_void_p, ctypes.c_void_p)

    class Request(ctypes.Structure):
        _fields_ = [("version", w.DWORD), ("interface", w.DWORD), ("service", ctypes.c_void_p),
                    ("callback", callback_type), ("context", ctypes.c_void_p),
                    ("credentials", w.HANDLE), ("unicast", w.BOOL)]

    api.DnsServiceConstructInstance.argtypes = [
        w.LPCWSTR, w.LPCWSTR, ctypes.POINTER(w.DWORD), ctypes.c_void_p,
        w.WORD, w.WORD, w.WORD, w.DWORD, ctypes.POINTER(w.LPCWSTR), ctypes.POINTER(w.LPCWSTR)]
    api.DnsServiceConstructInstance.restype = ctypes.c_void_p
    api.DnsServiceFreeInstance.argtypes = [ctypes.c_void_p]
    api.DnsServiceFreeInstance.restype = None
    api.DnsServiceRegister.argtypes = [ctypes.POINTER(Request), ctypes.c_void_p]
    api.DnsServiceRegister.restype = w.DWORD
    api.DnsServiceDeRegister.argtypes = [ctypes.POINTER(Request), ctypes.c_void_p]
    api.DnsServiceDeRegister.restype = w.DWORD
    keys = (w.LPCWSTR * 2)("id", "v")
    values = (w.LPCWSTR * 2)(str(server_id), "1")
    service, host = _names(server_id)
    instance = api.DnsServiceConstructInstance(service, host, ctypes.byref(ip), None,
                                               port, 0, 0, 2, keys, values)
    if not instance:
        raise OSError("Could not create a DNS-SD advertisement")
    event = threading.Event()
    state = []

    @callback_type
    def completed(status, _context, result):
        state.append(status)
        if result and result != instance:
            api.DnsServiceFreeInstance(result)
        event.set()

    request = Request(1, adapter.value, instance, completed, None, None, False)
    pending = 9506  # DNS_REQUEST_PENDING
    status = api.DnsServiceRegister(ctypes.byref(request), None)
    if status not in (0, pending):
        api.DnsServiceFreeInstance(instance)
        raise OSError(status, "DNS-SD registration unavailable")
    # Keep the callback, request and instance alive until deregistration or process exit.
    sys.stdin.buffer.read()
    if event.is_set() and state and state[0] == 0:
        event.clear()
        if api.DnsServiceDeRegister(ctypes.byref(request), None) in (0, pending):
            if event.wait(2):
                api.DnsServiceFreeInstance(instance)
    # A pending native callback may still own references. Process exit releases them safely.
    import os
    os._exit(0)


@contextmanager
def announce(server_id, lan_ipv4, port):
    child = None
    zeroconf = info = None
    try:
        if lan_ipv4:
            address = str(ipaddress.IPv4Address(lan_ipv4))
            service, host = _names(server_id)
            if sys.platform == "win32":
                child = subprocess.Popen([sys.executable, "-B", str(Path(__file__).resolve()),
                                          str(server_id), address, str(port)], stdin=subprocess.PIPE,
                                         creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                from zeroconf import ServiceInfo, Zeroconf
                zeroconf = Zeroconf(interfaces=[address])
                info = ServiceInfo(SERVICE, service, addresses=[socket.inet_aton(address)],
                                   port=port, properties={"id": str(server_id), "v": "1"}, server=host)
                zeroconf.register_service(info)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print("LAN discovery unavailable; saved direct addresses remain usable:", type(exc).__name__, file=sys.stderr)
    try:
        yield
    finally:
        if child is not None:
            child.stdin.close()
            try:
                child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=4)
        if zeroconf is not None:
            try:
                if info is not None:
                    zeroconf.unregister_service(info)
            finally:
                zeroconf.close()


if __name__ == "__main__":
    try:
        _windows(*[sys.argv[1], sys.argv[2], int(sys.argv[3])])
    except (OSError, ValueError, AttributeError) as error:
        print("Windows DNS-SD unavailable; saved direct addresses remain usable:", type(error).__name__, file=sys.stderr)
