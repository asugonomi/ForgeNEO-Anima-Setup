"""Native companion for Forge Remote. Also runnable with the installed Forge venv."""
import argparse
import base64
import binascii
import csv
import hashlib
import io
import ipaddress
import json
import math
import os
import sqlite3
import re
import secrets
import threading
import time
import unicodedata
from pathlib import Path
from datetime import date, datetime
from fractions import Fraction
from collections import OrderedDict, Counter
from urllib.parse import urlsplit

import requests
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps
from starlette.concurrency import run_in_threadpool

PREFIX = "/forge-remote/v1"
EXTENSIONS = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}

# Keep the installed TAC option's spelling and its default exclusion list.
UNDERSCORE_EXCLUSIONS = "0_0,(o)_(o),+_+,+_-,._.,<o>_<o>,<|>_<|>,=_=,>_<,3_3,6_9,>_o,@_@,^_^,o_o,u_u,x_x,|_|,||_||"
TAG_BRACKET = re.compile(r"(\\*)([()[\]])")


def format_tag(name, kind, opts, exclusions):
    text = name.replace("_", " ") if opts.get("tac_replaceUnderscores", True) and name not in exclusions else name
    if kind == "main" and opts.get("tac_escapeParentheses", True):
        # A pre-escaped literal retains its slash. An even slash count leaves the bracket syntactic.
        def escape(match):
            slashes, bracket = match.groups()
            return slashes + ("\\" if len(slashes) % 2 == 0 else "") + bracket
        text = TAG_BRACKET.sub(escape, text)
    return text


def tag_post_count(value):
    value = value.strip()
    if not (value.isascii() and value.isdecimal()):
        if not re.fullmatch(r"[0-9]{1,3}(?:,[0-9]{3})+", value):
            return None
        value = value.replace(",", "")
    try:
        return int(value)
    except ValueError:
        return None


class TagCandidate:
    __slots__ = ("name", "kind", "words", "ranked_words", "count")

    def __init__(self, name, kind, words, count):
        self.name = name
        self.kind = kind
        self.words = self.ranked_words = words
        self.count = count


def loopback(host):
    try:
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def authorize(request: Request):
    token = os.getenv("FORGE_REMOTE_TOKEN", "")
    supplied = request.headers.get("authorization", "")
    if token and secrets.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
        return
    forwarded = any(k == "forwarded" or k.startswith("x-forwarded-") for k in request.headers)
    if (request.client and loopback(request.client.host)
            and request.url.hostname in ("localhost", "127.0.0.1", "::1")
            and not forwarded and not request.headers.get("origin")
            and request.headers.get("sec-fetch-site", "none") == "none"):
        return
    raise HTTPException(403, "Paired local connection or bridge token required")


def gallery_roots(root, opts):
    paths = [("Samples",opts["outdir_samples"])] if opts.get("outdir_samples") else [
        ("txt2img",opts.get("outdir_txt2img_samples","output/txt2img-images")),
        ("img2img",opts.get("outdir_img2img_samples","output/img2img-images")),
        ("Extras",opts.get("outdir_extras_samples","output/extras-images")),("Hires",opts.get("outdir_hires_samples"))]
    paths.extend([("Grids",opts["outdir_grids"])] if opts.get("outdir_grids") else [
        ("txt2img grid",opts.get("outdir_txt2img_grids","output/txt2img-grids")),
        ("img2img grid",opts.get("outdir_img2img_grids","output/img2img-grids"))])
    paths.append(("Saved",opts.get("outdir_save","output/images")))
    groups={}
    for kind,path in paths:
        if path: groups.setdefault((root/path).resolve(),[]).append(kind)
    return tuple((p," / ".join(kinds)) for p,kinds in groups.items())


def is_link(entry):
    if entry.is_symlink():
        return True
    return bool(getattr(entry.stat(follow_symlinks=False), "st_file_attributes", 0) & 0x400)


def folder_date(folder, root):
    for part in reversed(folder.relative_to(root).parts):
        if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}",part):
            try: return date.fromisoformat(part).isoformat()
            except ValueError: continue
    return None


class OutputIndex:
    """Private temporary SQLite index. Only names, paths and stat metadata are scanned."""
    def __init__(self):
        self.lock=threading.RLock()
        self.db=sqlite3.connect("",check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE images(id TEXT PRIMARY KEY,name TEXT NOT NULL,folder TEXT NOT NULL,
                root TEXT NOT NULL,kind TEXT NOT NULL,day TEXT NOT NULL,basis TEXT NOT NULL,mtime INTEGER NOT NULL,legacy INTEGER NOT NULL);
            CREATE INDEX images_by_day ON images(day DESC,mtime DESC,id);
            CREATE INDEX images_by_folder ON images(folder);
            CREATE INDEX images_legacy ON images(legacy,mtime DESC,id);
        """)
        self.dirs={}
        self.roots=None
        self.version=0
        self.checked=0.0

    def refresh(self,roots,force=False):
        same=roots==self.roots
        if same and not force and time.monotonic()-self.checked<12: return
        fresh={}
        boundaries={p for p,_ in roots}
        changed=False
        with self.db:
            for root,kind in roots:
                pending=[root]
                while pending:
                    directory=pending.pop()
                    try:
                        if directory.is_symlink() or directory.resolve()!=directory: continue
                        mtime=directory.stat().st_mtime_ns
                    except OSError: continue
                    previous=self.dirs.get(directory)
                    if same and not force and previous and mtime==previous[0]:
                        fresh[directory]=previous;pending.extend(previous[1]);continue
                    children=[];rows=[]
                    day=folder_date(directory,root)
                    legacy=int(any(k in {"txt2img","img2img","Samples","Extras","Hires"} for k in kind.split(" / ")))
                    try:
                        with os.scandir(directory) as entries:
                            for e in entries:
                                try:
                                    if e.is_dir(follow_symlinks=False):
                                        p=Path(e.path)
                                        if p not in boundaries and not is_link(e):children.append(p)
                                    elif os.path.splitext(e.name)[1].lower() in EXTENSIONS and e.is_file(follow_symlinks=False) and not is_link(e):
                                        ns=e.stat(follow_symlinks=False).st_mtime_ns
                                        grouped=day or datetime.fromtimestamp(ns//10**9).date().isoformat()
                                        key=hashlib.sha256(os.fsencode(e.path)).hexdigest()[:32]
                                        rows.append((key,e.name,str(directory),str(root),kind,grouped,"folder" if day else "modified",ns,legacy))
                                except (ValueError,OSError,OverflowError): continue
                    except OSError: continue
                    stored=self.db.execute("SELECT * FROM images WHERE folder=?",(str(directory),))
                    if set(stored)!=set(rows):
                        self.db.execute("DELETE FROM images WHERE folder=?",(str(directory),))
                        self.db.executemany("INSERT OR REPLACE INTO images VALUES(?,?,?,?,?,?,?,?,?)",rows)
                        changed=True
                    fresh[directory]=(mtime,children);pending.extend(children)
            for old in self.dirs.keys()-fresh.keys():
                removed=self.db.execute("DELETE FROM images WHERE folder=?",(str(old),)).rowcount
                changed=changed or bool(removed)
        self.roots=roots;self.dirs=fresh;self.checked=time.monotonic()
        if changed or not self.version:self.version+=1

    def dates(self):
        days=[{"day":d,"count":n} for d,n in self.db.execute("SELECT day,count(*) FROM images GROUP BY day ORDER BY day DESC")]
        return {"version":self.version,"total":sum(d["count"] for d in days),"days":days}

    def page(self,day,offset,limit,version=None):
        if version is not None and version!=self.version: raise HTTPException(409,"Gallery changed. Refresh dates.")
        if day is not None:
            try:
                if date.fromisoformat(day).isoformat()!=day:raise ValueError()
            except ValueError:raise HTTPException(422,"Invalid date")
        where,args=(" WHERE day=?",(day,)) if day else ("",())
        total=self.db.execute("SELECT count(*) FROM images"+where,args).fetchone()[0]
        query="SELECT id,name,kind,day,basis,mtime FROM images"+where+" ORDER BY day DESC,mtime DESC,id LIMIT ? OFFSET ?"
        results=[]
        for key,name,kind,group,basis,mtime in self.db.execute(query,(*args,limit,offset)):
            stamp=datetime.fromtimestamp(mtime//10**9).astimezone().isoformat(timespec="seconds")
            results.append({"id":key,"name":name,"source":kind,"day":group,"day_source":basis,"modified":stamp})
        return {"version":self.version,"total":total,"offset":offset,"items":results}


LORA_PICTURES = ("png", "jpg", "jpeg", "webp", "gif", "avif", "heif", "jxl")


def standalone_lora_roots(root, configured):
    """First folder corresponds to TagComplete's primary LoRA root; remaining folders are additional Forge roots."""
    return [root / value for value in configured] if configured else [root/"models/Lora"]


class ExtensionFolders:
    """Resolve supported installed extensions without importing or executing their code."""
    def __init__(self, forge, data=None, tag_extension=None, ar_extension=None):
        self.forge = Path(forge).resolve()
        self.data = Path(data or forge).resolve()
        self.explicit = {"tags": tag_extension, "ratios": ar_extension}

    def active(self, feature, options):
        disabled = {str(name).casefold() for name in options.get("disabled_extensions") or []}
        disable_all = str(options.get("disable_all_extensions", "none")).lower()
        if disable_all == "all" or options.get("_cli_disable_all"):
            return None
        candidates = []
        for parent, builtin in ((self.data/"extensions", False), (self.forge/"extensions-builtin", True)):
            if not builtin and (disable_all == "extra" or options.get("_cli_disable_extra")):
                continue
            if not parent.is_dir():
                continue
            for folder in parent.iterdir():
                if not folder.is_dir() or folder.name.casefold() in disabled:
                    continue
                if feature == "tags":
                    signature = (folder/"scripts/tag_autocomplete_helper.py").is_file()
                    known = folder.name.casefold() in {"sd-webui-tagcomplete-neo","a1111-sd-webui-tagcomplete","tag-autocomplete"}
                    match = (known or signature) and (folder/"tags").is_dir()
                else:
                    match = (folder/"aspect_ratios.txt").is_file() and (folder/"resolutions.txt").is_file()
                if match:
                    candidates.append(folder.resolve())
        explicit = self.explicit[feature]
        if explicit:
            p = Path(explicit)
            selected = [folder for folder in candidates if folder.name.casefold() == str(explicit).casefold()
                        or folder == (self.forge/p).resolve()]
            if len(selected) == 1:
                return selected[0]
            raise HTTPException(409, "Selected extension is unavailable or disabled")
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1 and feature == "tags":
            marker = self.forge/"tmp/tagAutocompletePath.txt"
            try:
                if marker.stat().st_size <= 8192:
                    picked = (self.forge/marker.read_text(encoding="utf-8-sig").strip()).resolve()
                    for folder in candidates:
                        if (folder/"tags").resolve() == picked:
                            return folder
            except (ValueError, OSError, RuntimeError):
                pass
        if candidates:
            arg = "--tag-extension" if feature == "tags" else "--ar-extension"
            raise HTTPException(409, f"Multiple active compatible folders. Select one with {arg}")
        return None


def file_stamp(p):
    s = p.stat()
    return (s.st_size, s.st_mtime_ns)


def lora_scan(roots, include_hidden):
    """Each directory is read once, including custom roots. Never descend through nested links."""
    models = {}
    for number, root in enumerate(roots):
        if not root.is_dir():
            continue
        todo = [root]
        while todo:
            directory = todo.pop()
            files = {}
            try:
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if not include_hidden and entry.name.startswith("."):
                            continue
                        if is_link(entry):
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            todo.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            files[os.path.normcase(entry.name)] = Path(entry.path)
            except OSError:
                continue
            for model in files.values():
                if model.suffix.lower() != ".safetensors":
                    continue
                covers = []
                for ext in LORA_PICTURES:
                    for suffix in (f".{ext}", f".preview.{ext}", f".thumb.{ext}", f".preview.0.{ext}"):
                        cover = files.get(os.path.normcase(model.stem+suffix))
                        if cover is not None:
                            covers.append(cover)
                try:
                    ref = (root, model, file_stamp(model), tuple((p,file_stamp(p)) for p in covers))
                except OSError:
                    continue
                models[(number, os.path.normcase(str(model.relative_to(root))))] = ref
    return models


def safe_local(p, root):
    return p.resolve(strict=True) == p and p.is_relative_to(root) and p.is_file()


def gallery_jpeg(path, edge):
    try:
        with Image.open(path) as im:
            if im.width*im.height>64_000_000:
                raise HTTPException(413,"Image too large for preview")
            im.draft("RGB",(edge,edge))
            im.thumbnail((edge,edge),Image.Resampling.LANCZOS)
            flat=ImageOps.exif_transpose(im).convert("RGB")
            flat.info.clear()
            out=io.BytesIO()
            flat.save(out,"JPEG",quality=88 if edge>256 else 82)
            return Response(out.getvalue(),media_type="image/jpeg",
                            headers={"Cache-Control":"private, no-store","X-Content-Type-Options":"nosniff"})
    except Image.DecompressionBombError:
        raise HTTPException(413,"Image too large for preview")
    except (ValueError,OSError):
        raise HTTPException(415,"Image could not be opened")


def small_jpeg(stream):
    with Image.open(stream) as im:
        if im.format not in {"PNG","JPEG","WEBP","GIF","AVIF","HEIF","JPEGXL"} or im.width*im.height > 16_000_000:
            raise ValueError()
        im.draft("RGB", (256,256))
        im.thumbnail((256,256), Image.Resampling.LANCZOS)
        flat = ImageOps.exif_transpose(im).convert("RGB")
        # Re-encoding pixels drops text, source metadata and any secondary frames.
        flat.info.clear()
        out = io.BytesIO()
        flat.save(out, "JPEG", quality=82)
        return out.getvalue()


def embedded_lora_cover(model):
    with model.open("rb") as f:
        size = int.from_bytes(f.read(8), "little")
        if not 2 <= size <= 8*1024*1024:
            return None
        header = f.read(size)
    if len(header) != size or b'"ssmd_cover_images"' not in header:
        return None
    covers = json.loads(header).get("__metadata__",{}).get("ssmd_cover_images")
    if isinstance(covers,str):
        covers = json.loads(covers)
    if isinstance(covers,list):
        for value in covers[:8]:
            if isinstance(value,str) and 0<len(value)<=6*1024*1024:
                try:
                    return small_jpeg(io.BytesIO(base64.b64decode(value,validate=True)))
                except (ValueError, OSError, binascii.Error, Image.DecompressionBombError):
                    continue
    return None


class NativeFiles:
    def __init__(self, root, options, lora_roots=None, extensions=None):
        self.root, self.options = root.resolve(), options
        self.extensions = extensions or ExtensionFolders(root)
        self.lock = threading.RLock()
        self.output_index = OutputIndex()
        self.tag_key = None
        self.tags = []
        self.ranked_tags = []
        self.lora_roots = lora_roots or (lambda: [self.root/"models/Lora"])
        self.lora_index = {}
        self.lora_cache = OrderedDict()
        self.lora_cache_bytes = 0
        self.lora_slots = threading.BoundedSemaphore(2)

    def gallery(self):
        index=self.output_index
        with index.lock:
            index.refresh(gallery_roots(self.root,self.options()))
            return [{"id":k,"name":n} for k,n in index.db.execute(
                "SELECT id,name FROM images WHERE legacy=1 ORDER BY mtime DESC,id LIMIT 100")]

    def gallery_dates(self,refresh=False):
        index=self.output_index
        with index.lock:
            index.refresh(gallery_roots(self.root,self.options()),refresh)
            return index.dates()

    def gallery_page(self,day,offset,limit,version):
        index=self.output_index
        with index.lock:
            roots=gallery_roots(self.root,self.options())
            if version is None or roots!=index.roots:index.refresh(roots)
            return index.page(day,offset,limit,version)

    def path(self, key):
        index=self.output_index
        with index.lock:
            value=index.db.execute("SELECT folder,name,root FROM images WHERE id=?",(key,)).fetchone()
        if value:
            folder,name,base=value
            path=Path(folder)/name
            try:
                if Path(base) in {root for root,_ in gallery_roots(self.root,self.options())} and safe_local(path,Path(base)):
                    return path
            except (RuntimeError,OSError):pass
        raise HTTPException(404, "Image unavailable. Refresh gallery.")

    def loras(self):
        opts = self.options()
        folder = self.extensions.active("tags",opts)
        source = folder/"tags/temp/lora.txt" if folder else None
        roots = tuple(dict.fromkeys(Path(r).resolve() for r in self.lora_roots()))
        models = lora_scan(roots, opts.get("list_hidden_files", True))
        items, refs = [], []
        seen = set()
        with (source.open(encoding="utf-8-sig", newline="") if source and source.is_file() else io.StringIO("")) as f:
            for row in csv.reader(f):
                if not row or not row[0].strip():
                    continue
                relative = Path(row[0].strip().replace(chr(92), "/"))
                if relative.is_absolute() or ".." in relative.parts or any(":" in s for s in relative.parts):
                    continue
                name = relative.stem
                alias = row[3].strip() if len(row) >= 4 else ""
                if opts.get("lora_preferred_name") == "Filename" or not alias:
                    alias = name
                arch = row[4].split("|") if len(row) >= 5 else []
                items.append({"name": name, "path": relative.as_posix(), "alias": alias,
                              "arch": [a.strip().lower() for a in arch if a.strip()]})
                ref = models.get((0,os.path.normcase(str(relative))))
                refs.append(ref)
                if ref: seen.add(ref[1])
        # TagComplete scans its primary LoRA root. Forge can also load additional named roots.
        for (number,_), ref in models.items():
            model = ref[1]
            if model not in seen:
                seen.add(model)
                relative = model.relative_to(roots[number]).as_posix()
                display_path=f"[{number+1}]/{relative}" if number else relative
                items.append({"name":model.stem,"alias":model.stem,"path":display_path,"arch":[]})
                refs.append(ref)
        stem_counts = Counter(i["name"].casefold() for i in items)
        owners = Counter(k for i in items for k in {i["name"].casefold(),i["alias"].casefold()})
        index = {}
        for item, ref in zip(items, refs):
            if owners[item["alias"].casefold()] > 1:
                item["ambiguous"] = True
            duplicate = stem_counts[item["name"].casefold()] > 1
            if duplicate:
                item["duplicate_name"] = True
            if not ref:
                continue
            root, model, stamp, covers = ref
            version = (str(model),stamp,[(str(p),s) for p,s in covers])
            key = hashlib.sha256(repr(version).encode("utf-8","surrogatepass")).hexdigest()[:32]
            item["thumbnail_id"] = key
            index[key] = ref
            if duplicate:
                metadata = model.with_suffix(".json")
                try:
                    if safe_local(metadata,root) and metadata.stat().st_size <= 262144:
                        weight = json.loads(metadata.read_text(encoding="utf-8-sig")).get("preferred weight")
                        if type(weight) in (int,float) and math.isfinite(weight):
                            item["preferred_weight"] = weight
                except (OSError, ValueError, AttributeError):
                    pass
        with self.lock:
            self.lora_index = index
        return {"items": items, "auto_complete": bool(opts.get("tac_active", True) and opts.get("tac_useLoras", True)), "preset": str(opts.get("forge_preset", "")).lower(),
                "filter": opts.get("tac_loraArchFilter", "Off"),
                "weight": opts.get("extra_networks_default_multiplier", 1)}

    def lora_thumb(self, key):
        if not re.fullmatch(r"[0-9a-f]{32}",key):
            raise HTTPException(404,"Preview unavailable")
        with self.lock:
            ref = self.lora_index.get(key)
        if ref is None:
            raise HTTPException(404,"Refresh LoRA catalog")
        root, model, stamp, covers = ref
        try:
            if (not safe_local(model,root) or file_stamp(model)!=stamp or
                any(not safe_local(p,root) or file_stamp(p)!=s for p,s in covers)):
                raise HTTPException(404,"Refresh LoRA catalog")
            with self.lora_slots:
                with self.lock:
                    if key in self.lora_cache:
                        self.lora_cache.move_to_end(key)
                        data=self.lora_cache[key]
                        if data: return data
                        raise HTTPException(404,"Preview unavailable")
                data=None
                for p,s in covers:
                    if s[0]>24*1024*1024: continue
                    try:
                        with p.open("rb") as stream: data=small_jpeg(stream)
                        break
                    except (ValueError,OSError,Image.DecompressionBombError):
                        continue
                if data is None:
                    data=embedded_lora_cover(model)
                with self.lock:
                    self.lora_cache_bytes -= len(self.lora_cache.pop(key,None) or b"")
                    self.lora_cache[key]=data
                    self.lora_cache_bytes += len(data or b"")
                    while len(self.lora_cache)>192 or self.lora_cache_bytes>8*1024*1024:
                        self.lora_cache_bytes-=len(self.lora_cache.popitem(last=False)[1] or b"")
                if data: return data
        except (ValueError,OSError,TypeError,AttributeError,Image.DecompressionBombError,RecursionError):
            pass
        raise HTTPException(404,"Preview unavailable")

    def tag_inputs(self, opts):
        folder = self.extensions.active("tags",opts)
        if not opts.get("tac_active", True) or folder is None:
            return []
        directory = folder/"tags"
        extra = ("extra", opts.get("tac_extra.extraFile","extra-quality-tags.csv"))
        main = ("main", opts.get("tac_tagFile", "danbooru.csv"))
        order = [extra, main] if opts.get("tac_extra.addMode", "Insert before") == "Insert before" else [main, extra]
        if opts.get("tac_translation.searchByTranslation", True):
            order.insert(0, ("translations", opts.get("tac_translation.translationFile")))
        inputs = []
        for kind, name in order:
            if not isinstance(name, str) or Path(name).name != name or not name.lower().endswith(".csv"):
                continue
            p = directory / name
            if p.is_file() and p.resolve().is_relative_to(directory.resolve()):
                inputs.append((kind, p, p.stat().st_mtime_ns))
        return inputs

    def suggestions(self, q, formatted=False):
        opts = self.options()
        inputs = self.tag_inputs(opts)
        if not inputs:
            return []
        use_alias = bool(opts.get("tac_alias.searchByAlias", True) or opts.get("tac_alias.onlyShowAlias"))
        use_japanese = opts.get("tac_translation.searchByTranslation", True)
        old_format = opts.get("tac_translation.oldFormat", False)
        use_kana = opts.get("tacplus_enabled", True) and opts.get("tacplus_kana", True)
        key = (inputs, use_alias, use_kana, use_japanese, old_format)
        def fold(word):
            word = word.casefold().replace(" ", "_").strip()
            if use_kana and not word.isascii():
                word = unicodedata.normalize("NFKC", word)
                word = "".join(chr(ord(c) - 96) if "ァ" <= c <= "ヶ" else c for c in word)
            return word
        needle = fold(q)
        if not needle:
            return []
        with self.lock:
            if self.tag_key != key:
                records, translations = {}, {}
                for kind, path, _ in inputs:
                    with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
                        for row in csv.reader(stream):
                            if not row or not row[0]:
                                continue
                            if kind == "translations":
                                if len(row)>1:
                                    col = 2 if old_format and len(row)>2 and row[2] else 1
                                    translations[row[0]] = row[col]
                                continue
                            alternatives = [row[0]]
                            if use_alias and len(row)>3:
                                alternatives.extend(row[3].split(","))
                            if use_japanese:
                                value = row[4] if kind == "extra" and len(row)>4 and row[4] else translations.get(row[0], "")
                                alternatives.extend(value.split(","))
                            # TAC's additional CSV uses column three for a free-form annotation.
                            count = tag_post_count(row[2]) if kind == "main" and len(row)>2 else None
                            words = tuple(fold(a) for a in alternatives if a)
                            existing = records.get(row[0])
                            if existing is None:
                                records[row[0]] = TagCandidate(row[0], kind, words, count)
                            else:
                                if existing.count is None:
                                    existing.count = count
                                if existing.kind != kind:
                                    existing.ranked_words = tuple(dict.fromkeys(existing.ranked_words + words))
                self.tags = list(records.values())
                # Python's stable sort preserves configured dictionary order for ties and missing counts.
                self.ranked_tags = sorted(self.tags, key=lambda tag: tag.count if tag.count is not None else -1, reverse=True)
                self.tag_key = key
            result = []
            exclusions = set()
            if formatted:
                csv_exclusions = opts.get("tac_undersocreReplacementExclusionList", UNDERSCORE_EXCLUSIONS) or ""
                exclusions = {value.strip() for value in csv_exclusions.split(",")}
            boundary = "_" + needle
            substring = not needle.isascii()
            for tag in self.ranked_tags if formatted else self.tags:
                words = tag.ranked_words if formatted else tag.words
                if any(w.startswith(needle) or boundary in w or (substring and needle in w) for w in words):
                    result.append({"name":tag.name,"insert":format_tag(tag.name,tag.kind,opts,exclusions),
                                   "count":tag.count} if formatted else tag.name)
                    if len(result) == 32:
                        break
            return result


def local_json(base, route):
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.get(base + route, timeout=15, allow_redirects=False)
        if response.status_code != 200:
            raise HTTPException(503, f"Local Forge did not provide {route} (HTTP {response.status_code})")
        return response.json()
    except (ValueError, requests.RequestException) as exc:
        raise HTTPException(503, f"Cannot read local Forge {route}") from exc
    finally:
        session.close()


def ar_presets(root, options, extensions=None):
    directory = (extensions or ExtensionFolders(root)).active("ratios",options())
    if directory is None:
        raise HTTPException(404, "No enabled AR-PlusPlus compatible preset files")
    result = {"ratios": [], "resolutions": []}
    for filename, key in (("aspect_ratios.txt", "ratios"), ("resolutions.txt", "resolutions")):
        file = directory / filename
        if not file.is_file():
            raise HTTPException(404, f"AR-PlusPlus: {filename} unavailable")
        for i, line in enumerate(file.read_text(encoding="utf-8-sig").splitlines(), 1):
            body, _, comment = line.partition("#")
            if not body.strip():
                continue
            try:
                if key == "ratios":
                    left, right = (s.strip() for s in body.split(":"))
                    if not (0 < Fraction(left) and 0 < Fraction(right)):
                        raise ValueError()
                    row = {"label": body.strip(), "reverse": f"{right}:{left}"}
                else:
                    label, width, height = (s.strip() for s in body.split(","))
                    row = {"label": label, "width": int(width), "height": int(height)}
                    if min(row["width"], row["height"]) <= 0:
                        raise ValueError()
                result[key].append(dict(row, comment=comment.strip()))
            except (ValueError, ZeroDivisionError) as exc:
                raise HTTPException(422, f"AR-PlusPlus: {filename}, line {i}") from exc
    return result


def ar_dimensions(ratio, width, height, average, precision, reverse, one_dimension):
    left, right = (Fraction(part.strip()) for part in ratio.split(":"))
    if not one_dimension:
        # AR-PlusPlus rounds the provisional pixel values before scaling their area.
        w, h = (round(side / (left + right) * (2 * average)) for side in (left, right))
        if min(w, h) == 0:
            raise ValueError("Aspect ratio creates a zero-sized side")
        scale = math.sqrt(average ** 2 / (w * h))
        dims = tuple(round(side * scale / precision) * precision for side in (w, h))
        return dims[::-1] if reverse else dims
    ar = round(left / right, 2)
    if ar == 0:
        raise ValueError("Aspect ratio rounds to zero")
    if ar == 1:
        width = height = (max if reverse else min)(width, height)
    elif (ar > 1 and reverse) or (ar < 1 and not reverse):
        height = width / ar
    else:
        width = height * ar
    # The installed one-dimension mode includes a half-precision offset before round().
    return tuple(round((side + precision / 2) / precision) * precision for side in (width, height))


def install(app, root, options, forge_url, lora_roots=None, extensions=None):
    files = NativeFiles(root, options, lora_roots, extensions)
    api = APIRouter(prefix=PREFIX, dependencies=[Depends(authorize)])

    @api.post("/job-control")
    async def job_control(request: Request):
        body = bytearray()
        async for part in request.stream():
            if len(body) + len(part) > 4096:
                raise HTTPException(413, "Command too large")
            body.extend(part)
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeError, RecursionError) as exc:
            raise HTTPException(422, "Invalid JSON") from exc
        if not isinstance(payload, dict) or payload.get("action") not in ("poll", "skip", "interrupt", "receipt"):
            raise HTTPException(422, "Unknown control operation")

        def relay():
            target = urlsplit(forge_url)
            if target.scheme not in ("http", "https") or target.hostname not in ("localhost", "127.0.0.1", "::1"):
                raise HTTPException(503, "Scoped control requires a local Forge instance")
            try:
                with requests.Session() as session:
                    session.trust_env = False
                    reply = session.post(forge_url.rstrip("/") + "/forge-remote-control/v1/command",
                                         json=payload, headers={"X-Forge-Remote-Internal": "1"},
                                         timeout=(2, 4), allow_redirects=False)
                    if reply.status_code == 404 and payload["action"] == "poll":
                        return {"available": False, "active": False}
                    if reply.status_code in (409, 422):
                        raise HTTPException(reply.status_code, "Sampling ticket expired or invalid")
                    if reply.status_code != 200:
                        raise HTTPException(503, "Forge scoped job control unavailable")
                    return reply.json()
            except (requests.RequestException, ValueError) as exc:
                raise HTTPException(503, "Forge scoped job control unavailable") from exc
        return await run_in_threadpool(relay)

    @api.get("/size-presets")
    def size_presets():
        return ar_presets(root, options, extensions)

    @api.post("/size-presets/ratio")
    def size_ratio(payload: dict):
        ratios = ar_presets(root, options, extensions)["ratios"]
        label = payload.get("ratio")
        if not any(item["label"] == label for item in ratios):
            raise HTTPException(409, "Ratio unavailable. Refresh presets.")
        try:
            w, h, avg, prec = (payload[key] for key in ("width", "height", "average", "precision"))
            if any(type(v) is not int for v in (w, h, avg, prec)) or not (1 <= min(w, h, avg) <= max(w, h, avg) <= 65536):
                raise ValueError("Invalid base dimensions")
            if not 4 <= prec <= 128 or prec % 4:
                raise ValueError("Precision must be 4 through 128, in steps of 4")
            dims = ar_dimensions(label, w, h, avg, prec, payload.get("reverse") is True, payload.get("one_dimension") is True)
            if min(dims) <= 0:
                raise ValueError("Aspect ratio creates a zero-sized side")
            return {"width": dims[0], "height": dims[1], "average": avg}
        except (KeyError, ValueError, OverflowError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @api.get("/generation-defaults")
    def generation_defaults():
        config = local_json(forge_url, "/config")
        opts = local_json(forge_url, "/sdapi/v1/options")
        components = {p.get("elem_id"): p.get("value") for node in config["components"]
                      if (p := node.get("props", {})).get("elem_id") and p.get("value") is not None}
        profiles = {}
        preset = str(opts.get("forge_preset") or components.get("forge_ui_preset") or "")
        for tab, short in (("txt2img","t2i"),("img2img","i2i")):
            prefix = f"{preset}_{short}_"
            def value(key, default=None):
                return components.get(f"{tab}_{key}", default)
            def positive(key, component, default):
                option = opts.get(prefix + key)
                return option if isinstance(option, (int, float)) and option > 0 else value(component, default)
            profiles[tab] = {
                "prompt": value("prompt", ""), "negative_prompt": value("neg_prompt", ""),
                "width": positive("width", "width", 1024), "height": positive("height", "height", 1024),
                "cfg_scale": positive("cfg", "cfg_scale", 1), "steps": positive("step", "steps", 20),
                "sampler_name": opts.get(prefix + "sampler") or value("sampling", "Euler"),
                "scheduler": opts.get(prefix + "scheduler") or value("scheduler", "Simple"),
                "seed": value("seed", -1), "distilled_cfg_scale": opts.get(prefix + "dcfg", value("distilled_cfg_scale",3)),
                "batch_size": positive("batch_size","batch_size",1), "n_iter": value("batch_count", 1),
                "denoising_strength": value("denoising_strength", 0.45),
            }
        t2i = profiles["txt2img"]
        # An explicit empty list disables the Hires VAE/TE override models; the
        # UI sentinel inherits the first pass. Missing/None is invalid in Forge.
        hires_modules = components.get("hr_vae_te")
        if not isinstance(hires_modules, list) or not all(isinstance(name, str) for name in hires_modules):
            hires_modules = ["Use same choices"]
        hires_control = next((node["props"] for node in config["components"]
                              if node.get("props", {}).get("elem_id") == "hr_vae_te"), {})
        hires_choices = []
        for item in hires_control.get("choices") or []:
            # Gradio serializes (label, value) choices. The API accepts the value.
            value = item[1] if isinstance(item, (list, tuple)) and len(item) == 2 else item
            if isinstance(value, str):
                hires_choices.append(value)
        def hires_positive(key, component, fallback):
            selected=opts.get(f"{preset}_t2i_{key}")
            return selected if isinstance(selected,(int,float)) and selected>0 else components.get(component,fallback)
        t2i.update(hr_additional_modules=hires_modules, hr_additional_module_choices=hires_choices,
                   enable_hr=bool(components.get("txt2img_hr-checkbox",False)),
                   hr_second_pass_steps=hires_positive("hr_step","txt2img_hires_steps",0),
                   hr_cfg=hires_positive("hr_cfg","txt2img_hr_cfg",1),
                   hr_distilled_cfg=opts.get(f"{preset}_t2i_hr_dcfg",components.get("txt2img_hr_distilled_cfg",3)),
                   hr_scale=components.get("txt2img_hr_scale", 2.0),
                   hr_upscaler=components.get("txt2img_hr_upscaler", "Latent"))
        return {"preset":preset, "active":preset,
                "checkpoint": opts.get(f"forge_checkpoint_{preset}") or opts.get("sd_model_checkpoint") or "", **profiles}

    @api.get("/ad-controls")
    def ad_controls():
        # Only UI constraints are shared here. Per-mode slot values and the
        # canonical payload types come from Forge script-info and ADetailer.
        controls = {}
        profiles = {"txt2img": {}, "img2img": {}}
        for node in local_json(forge_url, "/config")["components"]:
            props = node.get("props", {})
            ident = props.get("elem_id") or ""
            mode = next((m for m in profiles if ident.startswith("script_"+m+"_adetailer_")), None)
            if not mode or node.get("type") not in ("textbox", "dropdown", "checkbox", "slider", "radio"):
                continue
            key = ident.removeprefix("script_"+mode+"_adetailer_")
            # Gradio widget ids use an empty suffix for the first tab and
            # ordinal suffixes such as _2nd and _3rd for subsequent tabs.
            suffix = key.rsplit("_", 1)[-1]
            index = 0
            if suffix[:-2].isdigit() and suffix[-2:] in ("st", "nd", "rd", "th"):
                index = int(suffix[:-2]) - 1
                key = key[:-(len(suffix)+1)]
            if not key.isidentifier():
                continue
            profiles[mode].setdefault(str(index), {})[key] = props.get("value")
            if mode != "txt2img" or index or key in controls:
                continue
            meta = {k: props[k] for k in ("minimum", "maximum", "step") if k in props}
            if "choices" in props:
                values = []
                for item in props["choices"]:
                    value = item[1] if isinstance(item, (list, tuple)) and len(item) == 2 else item
                    if isinstance(value, str):
                        values.append(value)
                meta["choices"] = values
            controls[key] = meta
        return {"controls":controls, "profiles":profiles}

    @api.get("/capabilities")
    def capabilities():
        try:
            dictionaries = files.tag_inputs(options())
            issue = None
        except HTTPException as exc:
            dictionaries, issue = [], exc.detail
        return {"version":1, "gallery":True, "tags":any(kind!="translations" for kind,_,_ in dictionaries),
                "tag_dictionary":" / ".join(dict.fromkeys(p.name for _,p,_ in dictionaries)), "tag_status":issue}

    @api.get("/gallery")
    def gallery():
        return files.gallery()

    @api.get("/gallery-index/dates")
    def gallery_dates(refresh:bool=False):
        return files.gallery_dates(refresh)

    @api.get("/gallery-index/page")
    def gallery_page(day:str|None=None,offset:int=Query(0,ge=0),limit:int=Query(24,ge=1,le=100),version:int|None=None):
        return files.gallery_page(day,offset,limit,version)

    @api.get("/gallery/{key}")
    def original(key:str):
        p = files.path(key)
        return FileResponse(p, media_type=EXTENSIONS[p.suffix.lower()], headers={"Cache-Control":"private, no-store"})

    @api.get("/gallery/{key}/thumb")
    def thumb(key:str):
        return gallery_jpeg(files.path(key),256)

    @api.get("/gallery/{key}/display")
    def display(key:str,large:bool=False):
        return gallery_jpeg(files.path(key),3072 if large else 1600)

    @api.get("/gallery/{key}/info")
    def info(key:str):
        with Image.open(files.path(key)) as im:
            return {"parameters":str(im.info.get("parameters", ""))}

    @api.get("/loras")
    def loras():
        return files.loras()

    @api.get("/loras/{key}/thumb")
    def lora_thumbnail(key:str):
        return Response(files.lora_thumb(key), media_type="image/jpeg",
                        headers={"Cache-Control":"private, no-store","X-Content-Type-Options":"nosniff"})

    @api.get("/tags")
    def tags(q:str=Query("",max_length=128),formatted:bool=False):
        return files.suggestions(q,formatted)

    app.include_router(api)


def runtime_layout(root, data_root=None, configured_loras=()):
    data=(root/(data_root or root)).resolve()
    loras=standalone_lora_roots(root,configured_loras) if configured_loras else [data/"models/Lora"]
    return data,tuple(dict.fromkeys(p.resolve() for p in loras))


def selected_options(values):
    strings={"forge_preset","lora_preferred_name","tac_tagFile","tac_extra.extraFile","tac_extra.addMode",
             "tac_translation.translationFile","tac_loraArchFilter","tac_undersocreReplacementExclusionList"}
    booleans={"list_hidden_files","tac_active","tac_useLoras","tac_replaceUnderscores","tac_escapeParentheses",
              "tac_alias.searchByAlias","tac_alias.onlyShowAlias","tac_translation.oldFormat",
              "tac_translation.searchByTranslation","tacplus_enabled","tacplus_kana"}
    result={}
    for key,value in values.items():
        if (key in strings or key.startswith("outdir_")) and isinstance(value,str):
            result[key]=value
        elif key in booleans and isinstance(value,bool):
            result[key]=value
        elif key=="disabled_extensions" and isinstance(value,list) and all(isinstance(item,str) for item in value):
            result[key]=value
        elif key=="disable_all_extensions" and value in ("none","all","extra"):
            result[key]=value
        elif key=="extra_networks_default_multiplier" and type(value) in (float,int) and math.isfinite(value):
            result[key]=value
    return result


class RuntimeOptions:
    def __init__(self, forge, config):
        self.forge=forge
        self.config=Path(config)
        self.lock=threading.Lock()
        self.cached={}
        self.expires=0.0

    def __call__(self):
        with self.lock:
            if time.monotonic() >= self.expires:
                live=local_json(self.forge,"/sdapi/v1/options")
                if not isinstance(live,dict):
                    raise HTTPException(503,"Forge options are unavailable")
                try:
                    configured=json.loads(self.config.read_text(encoding="utf-8-sig"))
                except FileNotFoundError:
                    configured={}
                except (OSError, ValueError, UnicodeError) as exc:
                    raise HTTPException(503,"Cannot read the configured Forge settings file") from exc
                if not isinstance(configured,dict):
                    raise HTTPException(503,"Forge settings file must be a JSON object")
                self.cached=selected_options(configured)
                self.cached.update(selected_options(live))
                self.expires=time.monotonic()+4
            return self.cached


def main():
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=17872)
    parser.add_argument("--forge-port", type=int, default=7860)
    parser.add_argument("--data-root",type=Path,help="Forge --data-dir (defaults to the Forge root).")
    parser.add_argument("--config",type=Path,help="Forge settings file if different from <data-root>/config.json.")
    parser.add_argument("--tag-extension",help="Choose one installed Tag Autocomplete folder if several are active.")
    parser.add_argument("--ar-extension",help="Choose one installed AR-PlusPlus compatible folder.")
    parser.add_argument("--lora-root", action="append", type=Path, default=[],
                        help="Override Forge LoRA paths. Repeat in Forge order, primary directory first.")
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root/"modules").is_dir() or not (root/"launch.py").is_file():
        parser.error("--root must be the Forge source directory containing modules and launch.py")
    if not (1<=args.port<=65535 and 1<=args.forge_port<=65535 and args.port!=args.forge_port):
        parser.error("Forge and companion ports must be different numbers in 1..65535")
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    forge_url = f"http://127.0.0.1:{args.forge_port}"
    data,roots=runtime_layout(root, args.data_root, args.lora_root)
    config=(data/(args.config or "config.json")).resolve()
    options=RuntimeOptions(forge_url,config)
    try:
        options()
    except HTTPException as exc:
        parser.exit(2, f"{exc.detail}. Start Forge with --api and check --forge-port.\n")
    extensions=ExtensionFolders(root,data,args.tag_extension,args.ar_extension)
    install(app, root, options, forge_url, lambda: roots, extensions)
    uvicorn.run(app, host="127.0.0.1", port=args.port, proxy_headers=False, access_log=False)


if __name__ == "__main__":
    main()
