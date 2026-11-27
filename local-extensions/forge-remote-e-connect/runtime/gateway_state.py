"""Local identity, one-time invitations and revocable paired devices."""
import base64
from contextlib import contextmanager
import ctypes
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
import ssl
import tempfile
import time
import uuid

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def _protect(data, decrypt=False):
    if os.name != "nt":
        return data
    class Blob(ctypes.Structure):
        _fields_ = [("size", ctypes.c_uint32), ("data", ctypes.c_void_p)]
    buffer = ctypes.create_string_buffer(data)
    source = Blob(len(data), ctypes.addressof(buffer))
    result = Blob()
    fn = (ctypes.windll.crypt32.CryptUnprotectData if decrypt
          else ctypes.windll.crypt32.CryptProtectData)
    if not fn(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(result)):
        raise RuntimeError("Windows could not protect or unlock the gateway identity.")
    try:
        return ctypes.string_at(result.data, result.size)
    finally:
        ctypes.windll.kernel32.LocalFree(ctypes.c_void_p(result.data))


def _identity():
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "Forge Remote E gateway")])
    now = datetime.now(timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(days=1)).not_valid_after(now + timedelta(days=3650))
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), False)
            .sign(key, hashes.SHA256()))
    return (str(uuid.uuid4()), cert.public_bytes(serialization.Encoding.DER),
            _protect(key.private_bytes(serialization.Encoding.DER, serialization.PrivateFormat.PKCS8,
                                       serialization.NoEncryption())))


def _digest(value):
    return hashlib.sha256(value.encode("ascii")).digest()


def _label(name):
    return "".join(c for c in str(name) if c.isprintable())[:64].strip()


class GatewayState:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / "identity.sqlite3"
        os.close(os.open(self.path, os.O_CREAT | os.O_WRONLY, 0o600))
        with self.db() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS identity(id INTEGER PRIMARY KEY, uuid TEXT, cert BLOB, key BLOB);
                CREATE TABLE IF NOT EXISTS invite(id INTEGER PRIMARY KEY, digest BLOB, deadline REAL);
                CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY, name TEXT, digest BLOB UNIQUE);
            """)
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT uuid,cert FROM identity WHERE id=1").fetchone()
            if not row:
                uid, cert, key = _identity()
                db.execute("INSERT INTO identity VALUES(1,?,?,?)", (uid, cert, key))
                row = uid, cert
            self.id, self.der = row
        self.pin = hashlib.sha256(self.der).hexdigest()

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.path, timeout=10)
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def tls(self):
        with self.db() as db:
            data = db.execute("SELECT key FROM identity WHERE id=1").fetchone()[0]
        key = serialization.load_der_private_key(_protect(data, decrypt=True), password=None)
        password = secrets.token_bytes(32).hex().encode("ascii")
        pem = x509.load_der_x509_certificate(self.der).public_bytes(serialization.Encoding.PEM)
        pem += key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                 serialization.BestAvailableEncryption(password))
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        with tempfile.NamedTemporaryFile(dir=self.directory, delete=False) as handle:
            path = Path(handle.name)
            try:
                handle.write(pem)
            finally:
                handle.close()
        try:
            ctx.load_cert_chain(path, password=password)
        finally:
            path.unlink(missing_ok=True)
        return ctx

    def invite(self, endpoints, name, now=None):
        secret = secrets.token_urlsafe(32)
        with self.db() as db:
            db.execute("REPLACE INTO invite VALUES(1,?,?)",
                       (_digest(secret), (time.time() if now is None else now) + 900))
        obj = dict(v=1, server_id=self.id, name=_label(name), endpoints=endpoints, pin=self.pin, secret=secret)
        payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode()
        return "FER1." + base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")

    def redeem(self, secret, name):
        if not isinstance(secret, str) or re.fullmatch(r"[A-Za-z0-9_-]{43}", secret) is None:
            return None
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT digest,deadline FROM invite WHERE id=1").fetchone()
            if not row or row[1] <= time.time() or not secrets.compare_digest(row[0], _digest(secret)):
                return None
            token = secrets.token_urlsafe(32)
            db.execute("INSERT INTO devices VALUES(?,?,?)", (uuid.uuid4().hex, _label(name) or "Android", _digest(token)))
            db.execute("DELETE FROM invite WHERE id=1")
            return token

    def authenticated(self, authorization):
        match = re.fullmatch(r"Bearer ([A-Za-z0-9_-]{43})", authorization or "")
        if not match:
            return False
        with self.db() as db:
            return db.execute("SELECT 1 FROM devices WHERE digest=?", (_digest(match[1]),)).fetchone() is not None

    def devices(self):
        with self.db() as db:
            return db.execute("SELECT id,name FROM devices ORDER BY rowid").fetchall()

    def revoke(self, identifier):
        with self.db() as db:
            return db.execute("DELETE FROM devices WHERE id=?", (identifier,)).rowcount == 1
