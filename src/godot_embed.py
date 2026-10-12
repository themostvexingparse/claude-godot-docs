"""Optional embedding (semantic) search for godot_docs.py.

Only imported when a search asks for `--mode semantic` or `--mode hybrid`, so the plain lexical tool keeps
working with the standard library alone. Needs numpy, onnxruntime and tokenizers, and a model folder under
models/ holding `tokenizer.json` plus an ONNX export (prefers `model_int8.onnx`).

Vectors are cached per model in `<index>_vectors.<model>.npz`, keyed by a hash of each passage's text, so an
index rebuild only re-embeds passages whose text actually changed.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
MODELS_DIR = SCRIPT_DIR.parent / "models"
# First existing folder wins; override with GODOT_DOCS_MODEL (folder name under models/ or a path).
DEFAULT_MODELS = ("gdocs-bge-small",)
MODEL_FILES = ("model_int8.onnx", "model.onnx", "onnx/model.onnx")
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TEXT_VERSION = "1"  # bump when passage_texts() changes
DOC_MAX_TOKENS = 256
QUERY_MAX_TOKENS = 64
CHUNK_CHARS = 1000
MAX_CHUNKS = 6
# A search may embed at most this many new passages inline (a few seconds); bigger jobs need `index --vectors`.
INLINE_BUILD_LIMIT = 1500
CHECKPOINT_EVERY = 2048


class VectorsMissing(Exception):
    pass

FENCE_RE = re.compile(r"^\s*```")


def find_model(name: str | None = None) -> Path:
    name = name or os.environ.get("GODOT_DOCS_MODEL")
    cands = [Path(name) if Path(name).is_dir() else MODELS_DIR / name] if name else [MODELS_DIR / m for m in DEFAULT_MODELS]
    for d in cands:
        if (d / "tokenizer.json").is_file() and any((d / f).is_file() for f in MODEL_FILES):
            return d
    raise RuntimeError(f"No embedding model found (looked for {', '.join(str(c) for c in cands)}).")


def model_file(model_dir: Path) -> Path:
    return next(model_dir / f for f in MODEL_FILES if (model_dir / f).is_file())


# ---------------------------------------------------------------------------
# Passage texts: what gets embedded for each index entry
# ---------------------------------------------------------------------------


def _prose(text: str) -> str:
    """Drop fenced code and markdown noise; keep the prose an embedding model can use."""
    out, in_code = [], False
    for line in (text or "").split("\n"):
        if FENCE_RE.match(line):
            in_code = not in_code
            continue
        if not in_code:
            out.append(line)
    t = "\n".join(out).replace("**", "").replace("`", "")
    return re.sub(r"\n{2,}", "\n", t).strip()


def _chunks(text: str, size: int = CHUNK_CHARS, limit: int = MAX_CHUNKS) -> list[str]:
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        while len(p) > size:  # a single huge paragraph: hard split on a sentence boundary when possible
            cut = p.rfind(". ", 0, size)
            cut = cut + 1 if cut > size // 2 else size
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(p[:cut].strip())
            p = p[cut:].strip()
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur)
            cur = ""
        cur = f"{cur}\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks[:limit] or [""]


def passage_texts(row) -> list[str]:
    kind, name, cls = row["kind"], row["name"] or "", row["cls"] or ""
    body = _prose(row["body"])
    if kind in ("page", "section"):
        return [f"{row['title']}\n{c}".strip() for c in _chunks(body)]
    if kind == "class":
        sig = row["signature"] or name
        return [f"{name} class ({sig})\n{c}".strip() for c in _chunks(body, limit=3)]
    head = f"{cls}.{name} ({kind.replace('_', ' ')})"
    sig = (row["signature"] or "").split("\n")[0]
    if kind == "enum":
        try:
            values = json.loads(row["extra"] or "{}").get("values", [])
        except ValueError:
            values = []
        vals = "; ".join(f"{v[0]}: {v[1]}" if v[1] else v[0] for v in values[:12])
        return [f"{head} {sig}\n{vals}".strip()]
    return [f"{head} {sig}\n{_chunks(body, limit=1)[0]}".strip()]


def all_passages(conn) -> tuple[list[int], list[str]]:
    ns, texts = [], []
    for row in conn.execute("SELECT n, id, kind, name, cls, title, signature, body, extra FROM entries ORDER BY n"):
        for t in passage_texts(row):
            ns.append(row["n"])
            texts.append(t)
    return ns, texts


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Encoder:
    def __init__(self, model_dir: Path, threads: int | None = None):
        # Narrow imports: the tokenizers Rust core and the onnxruntime session class only.
        from tokenizers import Tokenizer
        from onnxruntime import InferenceSession, SessionOptions
        self.model_dir = model_dir
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tok.no_padding()
        self.tok.no_truncation()
        opts = SessionOptions()
        # Hyperthreads hurt: on an 8-thread CPU, 6 threads ran ~1.8x faster than 8.
        threads = threads or int(os.environ.get("GODOT_DOCS_THREADS", "0"))
        opts.intra_op_num_threads = threads or max(1, min(6, (os.cpu_count() or 4) - 2))
        opts.log_severity_level = 3
        self.sess = InferenceSession(str(model_file(model_dir)), opts, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}

    def encode(self, texts: list[str], query: bool = False, batch: int = 32, log=None):
        import numpy as np
        if query:
            texts = [QUERY_PREFIX + t for t in texts]
        max_len = QUERY_MAX_TOKENS if query else DOC_MAX_TOKENS
        self.tok.enable_truncation(max_len)
        encs = self.tok.encode_batch(texts)
        order = sorted(range(len(encs)), key=lambda i: len(encs[i].ids))  # length-bucketed batches: little padding
        out = np.zeros((len(texts), 384), dtype=np.float32)
        t0, done = time.time(), 0
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            width = max(len(encs[i].ids) for i in idx)
            ids = np.zeros((len(idx), width), dtype=np.int64)
            mask = np.zeros_like(ids)
            for r, i in enumerate(idx):
                e = encs[i].ids
                ids[r, :len(e)] = e
                mask[r, :len(e)] = 1
            feed = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feed["token_type_ids"] = np.zeros_like(ids)
            cls = self.sess.run(None, feed)[0][:, 0]  # CLS pooling (bge)
            if out.shape[1] != cls.shape[1]:
                out = np.zeros((len(texts), cls.shape[1]), dtype=np.float32)
            out[idx] = cls
            done += len(idx)
            if log and (s // batch) % 50 == 0:
                log(f"[godot_docs]   {done}/{len(texts)} ({time.time() - t0:.0f}s)")
        out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
        return out


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


def _hash(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8")).digest()[:12]


def model_signature(model_dir: Path) -> str:
    f = model_file(model_dir)
    st = f.stat()
    return f"{model_dir.name}:{f.name}:{st.st_size}:{int(st.st_mtime)}:{TEXT_VERSION}"


class VectorStore:
    """Passage vectors for one model, aligned to the current index's entry numbers."""

    def __init__(self, ix, model_dir: Path, build_limit: int | None = INLINE_BUILD_LIMIT,
                 log=lambda m: print(m, file=sys.stderr)):
        import numpy as np
        self.ix, self.model_dir, self.log = ix, model_dir, log
        self._encoder = None
        self.path = Path(ix.path).with_name(f"{Path(ix.path).stem}_vectors.{model_dir.name}.npz")
        sig, built = model_signature(model_dir), index_signature(ix)
        cached = None
        if self.path.is_file():
            try:
                # read everything and close the file: Windows can't replace an .npz that is still open
                with np.load(self.path) as z:
                    cached = {k: z[k] for k in z.files}
                meta = json.loads(str(cached["meta"]))
            except (OSError, ValueError, KeyError):
                cached, meta = None, {}
            if cached is not None and meta.get("model") == sig and meta.get("index") == built:
                self.vecs = cached["vecs"].astype(np.float32)
                self.pn = cached["pn"]
                return
            if cached is not None and meta.get("model") != sig:
                cached = None
        self._build(cached, sig, built, build_limit)

    @property
    def encoder(self) -> Encoder:
        if self._encoder is None:
            self._encoder = Encoder(self.model_dir)
        return self._encoder

    def _build(self, cached, sig: str, built: str, build_limit: int | None):
        import numpy as np
        t0 = time.time()
        ns, texts = all_passages(self.ix.conn)
        xn, xt, _ = extra_passages(self.ix)
        ns, texts = ns + xn, texts + xt
        hashes = [_hash(t) for t in texts]
        reuse = {}
        if cached is not None:  # same model: keep vectors whose passage text is unchanged (also resumes a partial build)
            reuse = {h.tobytes(): k for k, h in enumerate(cached["hashes"])}
        old = cached["vecs"] if cached is not None else None
        todo = [k for k, h in enumerate(hashes) if h not in reuse]
        if build_limit is not None and len({hashes[k] for k in todo}) > build_limit:
            raise VectorsMissing(f"{len(todo)} of {len(texts)} passages need embedding for model {self.model_dir.name}")
        self.log(f"[godot_docs] Embedding with {self.model_dir.name}: {len(texts)} passages, "
                 f"{len(texts) - len(todo)} cached, {len(todo)} to embed ...")
        vecs = np.zeros((len(texts), 384), dtype=np.float16)
        for k, h in enumerate(hashes):
            if h in reuse:
                vecs[k] = old[reuse[h]]
        done = [k for k, h in enumerate(hashes) if h in reuse]
        todo.sort(key=lambda k: len(texts[k]))  # similar lengths per checkpoint keep batches tight
        first: dict = {}  # identical texts (a concept phrase shared by several targets) are embedded once
        for k in todo:
            first.setdefault(hashes[k], k)
        for c in range(0, len(todo), CHECKPOINT_EVERY):
            part = todo[c:c + CHECKPOINT_EVERY]
            uniq = [k for k in part if first[hashes[k]] == k]
            vecs[uniq] = self.encoder.encode([texts[k] for k in uniq]).astype(np.float16)
            for k in part:
                vecs[k] = vecs[first[hashes[k]]]
            done += part
            rate = len(done) - (len(texts) - len(todo))
            self.log(f"[godot_docs]   {c + len(part)}/{len(todo)} embedded ({time.time() - t0:.0f}s, "
                     f"{rate / max(time.time() - t0, 1e-9):.0f}/s)")
            if c + len(part) < len(todo):  # checkpoint: an interrupted build resumes from here
                self._save(vecs[done], np.asarray([ns[k] for k in done], dtype=np.int32),
                           [hashes[k] for k in done], sig, "partial")
        self.vecs = vecs.astype(np.float32)
        self.pn = np.asarray(ns, dtype=np.int32)
        self._save(vecs, self.pn, hashes, sig, built)
        self.log(f"[godot_docs] Embedded {len(todo)} passages in {time.time() - t0:.1f}s -> {self.path.name}")

    def _save(self, vecs, pn, hashes, sig, built):
        import numpy as np
        meta = json.dumps({"model": sig, "index": built, "passages": len(hashes)})
        tmp = self.path.with_name(self.path.name + f".tmp{os.getpid()}.npz")
        np.savez(tmp, vecs=vecs, pn=pn, hashes=np.frombuffer(b"".join(hashes), dtype="S12"), meta=np.array(meta))
        os.replace(tmp, self.path)

    def entry_scores(self, query: str):
        """Cosine similarity of the query to every entry (max over its passages), indexed by entry number n."""
        import numpy as np
        q = self.encoder.encode([query], query=True)[0]
        sims = self.vecs @ q
        scores = np.full(int(self.pn.max()) + 1, -1.0, dtype=np.float32)
        np.maximum.at(scores, self.pn, sims)
        return scores


def extra_passages(ix) -> tuple[list[int], list[str], str]:
    """Passages that aren't index entries (concept phrases), if the index provides them."""
    fn = getattr(ix, "extra_passages", None)
    return fn() if fn else ([], [], "")


def index_signature(ix) -> str:
    sig = extra_passages(ix)[2]
    return ix.meta.get("built_at", "") + ix.meta.get("fingerprint", "") + (f":concepts={sig}" if sig else "")


_STORES: dict = {}


def store_for(ix, model: str | None = None, build_limit: int | None = INLINE_BUILD_LIMIT) -> VectorStore:
    """Vectors for the current index. Raises VectorsMissing rather than embedding more than build_limit passages."""
    model_dir = find_model(model)
    key = (str(ix.path), str(model_dir))
    if key not in _STORES:
        _STORES[key] = VectorStore(ix, model_dir, build_limit)
    return _STORES[key]


def vector_status(ix, model: str | None = None) -> dict:
    model_dir = find_model(model)
    path = Path(ix.path).with_name(f"{Path(ix.path).stem}_vectors.{model_dir.name}.npz")
    out = {"model": model_dir.name, "model_file": model_file(model_dir).name, "vectors_path": str(path)}
    if not path.is_file():
        return dict(out, vectors="missing")
    import numpy as np
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
    fresh = meta.get("model") == model_signature(model_dir) and meta.get("index") == index_signature(ix)
    return dict(out, vectors="ready" if fresh else ("partial" if meta.get("index") == "partial" else "stale"),
                passages=meta.get("passages"))
