"""TLS gateway for the native application's explicit Forge and companion routes."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import socket
import threading
from urllib.parse import parse_qsl, unquote, urlsplit


COMPANION = "/forge-remote/v1/"
SD = "/sdapi/v1/"
MIB = 1024 * 1024
GET_FORGE = {SD + part for part in ("samplers", "schedulers", "scripts", "script-info", "sd-modules", "options", "loras")}
GET_FORGE.update(("/adetailer/v1/ad_model", "/adetailer/v1/schema"))
GET_COMPANION = {COMPANION + part for part in ("generation-defaults", "capabilities", "ad-controls",
                 "size-presets", "loras", "tags", "gallery", "gallery-index/dates", "gallery-index/page")}
POST_LIMIT = {SD+"txt2img": 2*MIB, SD+"img2img": 64*MIB, SD+"png-info": 64*MIB,
              "/internal/progress": 8192, COMPANION+"size-presets/ratio": 4096, COMPANION+"job-control": 4096}
QUERY = {COMPANION+"tags": {"q", "formatted"}, COMPANION+"gallery-index/dates": {"refresh"},
         COMPANION+"gallery-index/page": {"offset", "limit", "version", "day"}}
OPTIONS = {"live_previews_enable", "show_progress_every_n_steps", "live_preview_refresh_period",
           "lora_preferred_name", "extra_networks_default_multiplier"}


def destination(method, target):
    if (len(target)>4096 or not target.startswith("/") or target.startswith("//")
            or any(c in target for c in ("#", "\\")) or any(ord(c)<33 or ord(c)>126 for c in target)):
        return None
    parts = urlsplit(target)
    path = parts.path
    if method == "POST":
        if parts.query or path not in POST_LIMIT:
            return None
        return "companion" if path.startswith(COMPANION) else "forge"
    if method != "GET":
        return None
    allowed = QUERY.get(path, set())
    if re.fullmatch(COMPANION + r"gallery/[0-9a-f]{32}(?:/(?:thumb|display|info))?", path):
        local = "companion"
        if path.endswith("/display"):
            allowed = {"large"}
    elif path in GET_COMPANION or re.fullmatch(COMPANION + r"loras/[0-9a-f]{32}/thumb", path):
        local = "companion"
    elif path in GET_FORGE:
        local = "forge"
    elif path.startswith("/tacapi/v1/lora-info/"):
        name = unquote(path[len("/tacapi/v1/lora-info/"):], errors="strict")
        if (not name or name in (".", "..") or len(name)>255 or any(not c.isprintable() for c in name)
                or any(c in name for c in '/\\%:#?')):
            return None
        local = "forge"
    else:
        return None
    if not parts.query:
        return local
    keys = [k for k, _ in parse_qsl(parts.query, keep_blank_values=True, strict_parsing=True, max_num_fields=5)]
    return local if set(keys) <= allowed and len(set(keys))==len(keys) else None


class RequestError(Exception):
    def __init__(self, status):
        self.status = status


def safe_image(value):
    return isinstance(value,str) and (re.fullmatch(r"(?:data:image/[a-z0-9.+-]+;base64,)?[A-Za-z0-9+/=\r\n]*", value) is not None)


def read_payload(handler, limit):
    header = handler.headers.get_all("Content-Length", [])
    if len(header)!=1 or not re.fullmatch(r"[0-9]{1,10}",header[0]) or handler.headers.get("Transfer-Encoding"):
        raise RequestError(411)
    n = int(header[0])
    if n>limit:
        raise RequestError(413)
    if handler.headers.get_content_type()!="application/json":
        raise RequestError(415)
    data = handler.rfile.read(n)
    if len(data)!=n:
        raise RequestError(400)
    try:
        payload = json.loads(data)
    except (ValueError, UnicodeError, RecursionError):
        raise RequestError(400)
    if not isinstance(payload,dict):
        raise RequestError(400)
    return data, payload


class GatewayHandler(BaseHTTPRequestHandler):
    server_version = "ForgeRemoteE"
    sys_version = ""
    def log_message(self, *args):
        pass

    def setup(self):
        super().setup()
        self.connection.do_handshake()

    def reply(self, status, data=None):
        body = json.dumps(data or {}, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self.handle_api()

    def do_POST(self):
        self.handle_api()

    def handle_api(self):
        try:
            self.run_api()
        except RequestError as err:
            self.reply(err.status)
        except (ValueError, UnicodeError, RecursionError):
            self.reply(400)
        except (OSError, http.client.HTTPException):
            self.reply(502)

    def run_api(self):
        if self.headers.get("Origin") or self.headers.get("Transfer-Encoding"):
            raise RequestError(400)
        # BaseHTTPRequestHandler normalizes leading // before dispatch.
        target = self.requestline.split()[1]
        if self.command=="GET" and (self.headers.get("Content-Length") or self.headers.get("Expect")):
            raise RequestError(400)
        state = self.server.state
        if target == "/pair" and self.command == "POST":
            _, form = read_payload(self, 4096)
            token = state.redeem(form.get("secret"), form.get("device_name", "Android"))
            if not token:
                raise RequestError(403)
            self.reply(200, dict(version=1, server_id=state.id, token=token))
            return
        auth = self.headers.get_all("Authorization", [])
        if len(auth)!=1 or not state.authenticated(auth[0]):
            raise RequestError(401)
        if target == "/health" and self.command == "GET":
            self.reply(200, dict(version=1, server_id=state.id))
            return
        dest = destination(self.command, target)
        if dest is None:
            raise RequestError(404)
        outgoing = None
        if self.command == "POST":
            outgoing, form = read_payload(self, POST_LIMIT[target])
            # Image submissions from the native picker always contain bytes. No remote URL fetching.
            if target == SD+"png-info" and not safe_image(form.get("image")):
                raise RequestError(400)
            if target == SD+"img2img":
                images = form.get("init_images")
                if (not isinstance(images, list) or not 1<=len(images)<=8 or not all(safe_image(v) for v in images)
                        or ("mask" in form and form["mask"] is not None and not safe_image(form["mask"]))):
                    raise RequestError(400)
            if target in (SD+"img2img",SD+"txt2img"):
                if (any(k.startswith("outpath_") for k in form)
                        or "override_settings" in form
                        or form.get("override_settings_restore_afterwards", True) is not True):
                    raise RequestError(400)
        self.forward(dest, target, outgoing)

    def forward(self, dest, target, outgoing):
        # Exactly one request, fixed upstream and no redirects: important for generation submissions.
        conn = http.client.HTTPConnection("127.0.0.1", self.server.ports[dest],
                                         timeout=1200 if target in (SD+"img2img",SD+"txt2img") else 60)
        try:
            headers = {"Accept-Encoding":"identity"}
            if outgoing is not None:
                headers["Content-Type"] = "application/json"
            conn.request(self.command, target, body=outgoing, headers=headers)
            result = conn.getresponse()
            if 300<=result.status<400:
                raise RequestError(502)
            if result.status >=400:
                # Forge tracebacks may expose local paths and inputs. Preserve just the status.
                self.reply(result.status)
                return
            limit = (8 if target=="/internal/progress" else 96)*MIB
            length = result.getheader("Content-Length")
            if length is not None and (not length.isdecimal() or int(length)>limit):
                raise RequestError(502)
            if target==SD+"options":
                payload = result.read(2*MIB+1)
                if len(payload)>2*MIB:
                    raise RequestError(502)
                options = json.loads(payload)
                self.reply(200,{k:options[k] for k in OPTIONS if k in options})
                return
            mime = (result.getheader("Content-Type") or "application/octet-stream").split(";")[0].lower().strip()
            if mime not in ("image/jpeg", "image/png", "image/webp", "image/gif", "application/json", "application/octet-stream"):
                raise RequestError(502)
            # Unknown length must be completed and bounded before we claim success.
            body = None
            if length is None:
                body = result.read(limit+1)
                if len(body)>limit:
                    raise RequestError(502)
                length = str(len(body))
            self.send_response(result.status)
            self.send_header("Content-Type",mime)
            self.send_header("Content-Length",length)
            self.send_header("Cache-Control","no-store")
            self.send_header("X-Content-Type-Options","nosniff")
            self.end_headers()
            try:
                if body is not None:
                    self.wfile.write(body)
                else:
                    while chunk := result.read(65536):
                        self.wfile.write(chunk)
            except (OSError,http.client.HTTPException):
                self.close_connection = True
        finally:
            conn.close()


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    def __init__(self, address, state, context, ports, slots):
        self.state, self.context, self.ports, self.slots = state, context, ports, slots
        super().__init__(address, GatewayHandler)

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.allow_reuse_address = False
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def get_request(self):
        sock, peer = super().get_request()
        sock.settimeout(20)
        try:
            return self.context.wrap_socket(sock, server_side=True, do_handshake_on_connect=False), peer
        except Exception:
            sock.close()
            raise

    def process_request(self, request, client_address):
        if self.slots.acquire(blocking=False):
            try:
                super().process_request(request, client_address)
            except BaseException:
                self.slots.release()
                raise
        else:
            self.shutdown_request(request)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()

    def handle_error(self, request, client_address):
        # Browser probes/disconnects must not log headers, payloads, names or tokens.
        pass
