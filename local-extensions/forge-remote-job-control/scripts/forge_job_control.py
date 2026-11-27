"""Forge Remote control. Only the sampling callback may set Forge's skip/interrupt flags."""
import hashlib
import ipaddress
import json
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from fastapi import HTTPException, Request


PUBLIC_TASK = re.compile(r"task\(fr-[0-9a-f]{64}\)\Z")
PROOF = re.compile(r"[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}\Z")
EPOCH = re.compile(r"[a-f0-9]{32}\Z")
INTERNAL_ROUTE = "/forge-remote-control/v1/command"


def task_id(proof):
    return "task(fr-" + hashlib.sha256(("jp.local.forgedesk/v1:" + proof).encode("ascii")).hexdigest() + ")"


def validate(data):
    if not isinstance(data, dict) or not isinstance(data.get("proof"), str) or not PROOF.fullmatch(data["proof"]):
        raise HTTPException(422, "Invalid job proof")
    action = data.get("action")
    if action == "poll" and set(data) == {"proof", "action"}:
        return
    if action not in ("skip", "interrupt", "receipt") or set(data) != {"proof", "action", "epoch", "unit", "request_id"}:
        raise HTTPException(422, "Invalid command")
    if (not isinstance(data["epoch"], str) or not EPOCH.fullmatch(data["epoch"])
            or type(data["unit"]) is not int or not 0 <= data["unit"] <= 2**31 - 1
            or not isinstance(data["request_id"], str) or not PROOF.fullmatch(data["request_id"])):
        raise HTTPException(422, "Invalid control ticket")


@dataclass(frozen=True)
class Scope:
    task: str
    started: float
    unit: int
    processing: int
    iteration: int
    hires: bool


class Control:
    def __init__(self):
        self.lock = threading.Lock()
        self.scope = None
        self.epoch = None
        self.pending = None
        self.receipts = OrderedDict()

    def sample(self, scope, apply):
        # Called on Forge's generation thread, while that request owns queue_lock.
        with self.lock:
            if self.scope != scope:
                self.scope = scope
                self.epoch = uuid.uuid4().hex if scope is not None else None
                self.pending = None
            if self.pending is not None:
                key, action = self.pending
                if key not in self.receipts:
                    apply(action)
                    self.receipts[key] = action
                    if len(self.receipts) > 256:
                        self.receipts.popitem(last=False)

    def handle(self, data, current):
        validate(data)
        task = task_id(data["proof"])
        action = data["action"]
        with self.lock:
            scope = self.scope
            # 'current' reads the installed module's live task, time_start and job_no.
            active = scope is not None and (task, scope.started, scope.unit) == current() and task == scope.task
            status = "idle"
            if self.pending:
                status = "applied" if self.pending[0] in self.receipts else "queued"
            if action == "poll":
                if not active:
                    return {"available": True, "active": False}
                return {"available": True, "active": True, "epoch": self.epoch, "unit": scope.unit, "status": status}
            key = (task, data["epoch"], data["unit"], data["request_id"])
            applied = self.receipts.get(key)
            if applied is not None and action in ("receipt", applied):
                return {"status": "applied", "action": applied}
            matches = active and data["epoch"] == self.epoch and data["unit"] == scope.unit
            if action == "receipt":
                return {"status": "queued" if matches and self.pending and self.pending[0] == key else "expired"}
            if not matches or (self.pending is not None and self.pending != (key, action)):
                raise HTTPException(409, "Sampling unit changed or already controlled")
            self.pending = (key, action)
            return {"status": "queued", "action": action}


def local_request(request):
    try:
        local_peer = request.client and ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        local_peer = False
    if (not local_peer or request.url.hostname not in ("localhost", "127.0.0.1", "::1")
            or request.headers.get("x-forge-remote-internal") != "1"
            or request.headers.get("origin") or request.headers.get("sec-fetch-site", "none") != "none"
            or any(key == "forwarded" or key.startswith("x-forwarded-") for key in request.headers)):
        raise HTTPException(403, "Local companion required")


def register():
    from modules import progress, script_callbacks, shared

    control = Control()

    def current():
        return progress.current_task, shared.state.time_start, shared.state.job_no

    def sampling(params):
        task, started, unit = current()
        p = params.denoiser.p
        scope = None
        if isinstance(task, str) and PUBLIC_TASK.fullmatch(task) and p is not None and started is not None:
            scope = Scope(task, started, unit, id(p), getattr(p, "iteration", -1), bool(getattr(p, "is_hr_pass", False)))
        if scope is not None or control.scope is not None:
            def apply(action):
                if action == "skip":
                    shared.state.skip()
                else:
                    shared.state.interrupt()
            control.sample(scope, apply)

    def started(_demo, app):
        @app.post(INTERNAL_ROUTE, include_in_schema=False)
        async def handle(request: Request):
            local_request(request)
            if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
                raise HTTPException(415, "JSON required")
            body = bytearray()
            async for part in request.stream():
                if len(body) + len(part) > 4096:
                    raise HTTPException(413, "Command too large")
                body.extend(part)
            try:
                data = json.loads(body)
            except (ValueError, UnicodeError, RecursionError) as exc:
                raise HTTPException(422, "Invalid JSON") from exc
            return control.handle(data, current)

    script_callbacks.on_cfg_denoiser(sampling, name="forge_remote_job_control_sampling")
    script_callbacks.on_app_started(started, name="forge_remote_job_control_routes")


if __name__ != "forge_remote_control_test":
    register()
