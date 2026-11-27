"""Maintain private LAN and Tailscale listeners using the edition's existing keys."""
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
import sys
import threading

directory = Path(__file__).resolve().parent
sys.path.insert(0, str(directory / "runtime"))
import forge_remote_gateway as gateway
from gateway_state import GatewayState


def serve(args, stop, private):
    state = GatewayState(gateway.default_state())
    tls = state.tls()
    slots = threading.BoundedSemaphore(24)
    ports = {"forge": args.forge_port, "companion": args.companion_port}
    ready_args = (args.companion_port, args.forge_port) if private else (args.companion_port,)
    with ExitStack() as lifetime:
        lifetime.enter_context(gateway.local_companion(args))
        listeners = {}
        while not stop.is_set():
            lans = gateway.physical_lan()
            tail = gateway.tailscale_address()
            targets = set(lans if len(lans) == 1 else [])
            if tail:
                targets.add(tail)
            for address in set(listeners) - targets:
                listeners.pop(address).close()
            for address in targets - listeners.keys():
                if gateway.active_identity(address, args.port, state.pin):
                    continue
                try:
                    with ExitStack() as binding:
                        server = binding.enter_context(gateway.GatewayServer(
                            (address, args.port), state, tls, ports, slots))
                        threading.Thread(target=server.serve_forever, daemon=True).start()
                        binding.callback(server.shutdown)
                        if address in lans:
                            binding.enter_context(gateway.announce(state.id, address, args.port))
                        listeners[address] = binding.pop_all()
                        lifetime.callback(listeners[address].close)
                except OSError as error:
                    print("Forge Remote connection:", error, flush=True)
            if stop.wait(15) or not gateway._companion_ready(*ready_args):
                break
        for binding in listeners.values():
            binding.close()


def main():
    private = (directory / "runtime" / "forge-mobile-gallery").is_dir()
    args = Namespace(root=Path(sys.argv[1]), forge_port=int(sys.argv[2]),
                     port=17863 if private else 17873, companion_port=17862 if private else 17872,
                     lora_root=[], data_root=None, config=None, tag_extension=None, ar_extension=None)
    stop = threading.Event()
    def parent_closed():
        sys.stdin.buffer.read()
        stop.set()
    threading.Thread(target=parent_closed, daemon=True).start()
    while not stop.is_set():
        try:
            serve(args, stop, private)
        except Exception as error:
            print("Forge Remote connection:", error, flush=True)
        stop.wait(5)


if __name__ == "__main__":
    main()
