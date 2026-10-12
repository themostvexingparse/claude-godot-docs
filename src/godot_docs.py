#!/usr/bin/env python3
"""Offline search engine over the Godot documentation, built for AI coding agents.

Indexes the godot-docs reStructuredText sources (class reference + manual) into a
local SQLite FTS5 database and answers queries with compact, agent-friendly text.
Standard library only (Python 3.9+).

Commands (run `python godot_docs.py <command> -h` for details):
  search    ranked search over the API reference and the manual
  read      full text of one entry (class, member, page or page section)
  lookup    resolve a symbol name: which classes define it, Godot 3 renames, typos
  members   list a class's members, optionally including inherited ones
  hierarchy show a class's ancestors and subclasses
  examples  search code examples only
  toc       browse the manual's page tree
  index     build / rebuild / inspect the index

Every result carries an ID that can be passed to `read`.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import fnmatch
import json
import math
import os
import posixpath
import re
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from itertools import zip_longest
from pathlib import Path

SCHEMA_VERSION = "1"
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent  # repository root: src/, data/, models/
DATA_DIR = ROOT_DIR / "data"
TOOL_HASH = hashlib.sha1(Path(__file__).read_bytes()).hexdigest()[:12]
DEFAULT_MAX_CHARS = 10000
LANGS = ("gdscript", "csharp", "all")
MODES = ("lexical", "semantic", "hybrid")

# ---------------------------------------------------------------------------
# Locating the docs and the index
# ---------------------------------------------------------------------------


def is_docs_root(path: Path) -> bool:
    return (path / "classes").is_dir() and (path / "conf.py").is_file()


def find_docs_root(explicit: str | None) -> Path | None:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.environ.get("GODOT_DOCS_PATH"):
        candidates.append(Path(os.environ["GODOT_DOCS_PATH"]))
    candidates += [ROOT_DIR.parent, ROOT_DIR / "godot-docs", Path.cwd() / "godot-docs", Path.cwd()]
    for c in candidates:
        if is_docs_root(c):
            return c.resolve()
    return None


def default_index_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if os.environ.get("GODOT_DOCS_INDEX"):
        return Path(os.environ["GODOT_DOCS_INDEX"])
    return DATA_DIR / "godot_docs_index.db"


def docs_fingerprint(root: Path) -> str:
    """Identify the docs revision: the git commit if available, else file mtimes."""
    git = root / ".git"
    try:
        head = (git / "HEAD").read_text().strip()
        if not head.startswith("ref: "):
            return head
        ref = head[5:]
        if (git / ref).is_file():
            return (git / ref).read_text().strip()
        for line in (git / "packed-refs").read_text().splitlines():
            if line.endswith(" " + ref):
                return line.split()[0]
    except OSError:
        pass
    latest, count = 0.0, 0
    for p in root.rglob("*.rst"):
        count += 1
        latest = max(latest, p.stat().st_mtime)
    return f"mtime:{count}:{int(latest)}"


def read_godot_version(root: Path) -> str:
    try:
        m = re.search(r'"godot_version":\s*"([^"]+)"', (root / "conf.py").read_text(encoding="utf-8"))
        return m.group(1) if m else "?"
    except OSError:
        return "?"


# ---------------------------------------------------------------------------
# reStructuredText -> compact Markdown
# ---------------------------------------------------------------------------

UNDERLINE_RE = re.compile(r"^([=\-~^\"'`#*+.:_])\1{2,}\s*$")
DIRECTIVE_RE = re.compile(r"^\.\.\s+([A-Za-z][\w:-]*)::(?:\s+(.*))?$")
LABEL_RE = re.compile(r"^\.\.\s+_([^:]+):\s*$")
OPTION_RE = re.compile(r"^:([\w-]+):(?:\s+(.*))?$")
ROLE_RE = re.compile(r":(?P<role>[a-zA-Z][\w-]*):`(?P<body>[^`]*)`")
LITERAL_RE = re.compile(r"``(.+?)``")
EXTLINK_RE = re.compile(r"`([^`<]*?)\s*<([^`<>]+)>`__?")
NAMEDREF_RE = re.compile(r"`([^`]+)`__?")
SUBST_RE = re.compile(r"\|([\w-]+)\|")
SIMPLE_TABLE_RE = re.compile(r"^=+(\s+=+)+\s*$")
FENCE_RE = re.compile(r"^(\s*)```(\S*)\s*$")
PLACEHOLDER_RE = re.compile("\x00(\\d+)\x00")

SUBSTITUTIONS = {
    "void": "void", "const": "const", "virtual": "virtual", "required": "required",
    "vararg": "vararg", "constructor": "constructor", "static": "static",
    "operator": "operator", "bitfield": "BitField",
}
QUALIFIERS = ("const", "virtual", "required", "vararg", "static")
ADMONITIONS = {
    "note": "Note", "warning": "Warning", "tip": "Tip", "important": "Important",
    "caution": "Caution", "danger": "Danger", "attention": "Attention", "hint": "Hint",
    "error": "Error", "seealso": "See also",
}
SKIP_DIRECTIVES = {
    "image", "figure", "rst-class", "meta", "raw", "video", "contents", "index", "include",
    "cssclass", "literalinclude", "sectionauthor", "codeauthor", "role", "default-role",
}
LANG_NORMALIZE = {
    "c#": "csharp", "cs": "csharp", "gd": "gdscript", "sh": "shell", "bash": "shell",
    "console": "shell", "none": "text", "": "text", "c++": "cpp",
}


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def dedent(lines: list[str]) -> list[str]:
    ind = min((indent_of(l) for l in lines if l.strip()), default=0)
    return [l[ind:] if l.strip() else "" for l in lines]


def indent_lines(lines: list[str], n: int) -> list[str]:
    pad = " " * n
    return [pad + l if l.strip() else "" for l in lines]


def trim_blank(lines: list[str]) -> list[str]:
    a, b = 0, len(lines)
    while a < b and not lines[a].strip():
        a += 1
    while b > a and not lines[b - 1].strip():
        b -= 1
    return lines[a:b]


def take_block(lines: list[str], i: int, base_indent: int) -> tuple[list[str], int]:
    """Lines from i on that are blank or indented deeper than base_indent."""
    j = i
    while j < len(lines) and (not lines[j].strip() or indent_of(lines[j]) > base_indent):
        j += 1
    return lines[i:j], j


def split_options(block: list[str]) -> tuple[dict, list[str]]:
    lines = dedent(block)
    opts, k = {}, 0
    while k < len(lines) and OPTION_RE.match(lines[k]):
        m = OPTION_RE.match(lines[k])
        opts[m.group(1)] = m.group(2) or ""
        k += 1
    return opts, dedent(trim_blank(lines[k:]))


def tidy(lines: list[str]) -> str:
    text = "\n".join(l.rstrip() for l in lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def strip_code(text: str) -> str:
    out, in_code = [], False
    for line in text.split("\n"):
        if FENCE_RE.match(line):
            in_code = not in_code
            continue
        if not in_code:
            out.append(line)
    return "\n".join(out)


def first_sentence(text: str, limit: int = 200) -> str:
    t = re.sub(r"\s+", " ", strip_code(text)).strip()
    m = re.search(r"^(.{20,}?[.!?])(\s|$)", t)
    s = m.group(1) if m else t
    return s if len(s) <= limit else s[: limit - 3].rstrip() + "..."


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def slugify(title: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return s or "section"


def split_ident(name: str) -> list[str]:
    words = []
    for part in re.split(r"[_.@\s]+", name):
        words += re.findall(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+[A-Za-z]?", part)
    return [w.lower() for w in words]


def filter_code(text: str, lang: str) -> str:
    """Drop the other scripting language's tab of every GDScript/C# tab group."""
    lines = text.split("\n")
    blocks, groups, i = [], {}, 0
    while i < len(lines):
        m = FENCE_RE.match(lines[i])
        if m and m.group(2):
            tag = m.group(2)
            blang, _, g = tag.partition("|g")
            j = i + 1
            while j < len(lines) and not (FENCE_RE.match(lines[j]) and not FENCE_RE.match(lines[j]).group(2)):
                j += 1
            blocks.append((i, j, blang, int(g) if g.isdigit() else 0))
            if g.isdigit():
                groups.setdefault(int(g), set()).add(blang)
            i = j + 1
            continue
        i += 1
    drop = set()
    for start, end, blang, g in blocks:
        if lang != "all" and g and blang != lang and lang in groups.get(g, ()):
            drop.update(range(start, end + 1))
            if start > 0 and not lines[start - 1].strip():
                drop.add(start - 1)
        else:
            lines[start] = re.sub(r"\|g\d+$", "", lines[start])
    return re.sub(r"\n{3,}", "\n\n", "\n".join(l for k, l in enumerate(lines) if k not in drop))


class Resolver:
    """Global cross-reference tables, filled in pass 1 and used while converting text."""

    def __init__(self):
        self.anchors: dict[str, tuple[str, str, str]] = {}  # label -> (entry id, title, kind)
        self.pages: dict[str, tuple[str, str]] = {}  # path without .rst -> (page id, title)
        self.group = 0
        self.refcount: dict[str, int] = {}  # entry id -> number of cross-references pointing at it

    def resolve_doc(self, target: str, curdir: str):
        t = target.strip()
        t = t[:-4] if t.endswith(".rst") else t
        p = t[1:] if t.startswith("/") else posixpath.normpath(posixpath.join(curdir, t))
        return self.pages.get(p, (None, None))

    def glob_docs(self, pattern: str, curdir: str):
        p = pattern[1:] if pattern.startswith("/") else posixpath.normpath(posixpath.join(curdir, pattern))
        return [self.pages[k] for k in sorted(self.pages) if fnmatch.fnmatch(k, p)]


class RstConverter:
    def __init__(self, resolver: Resolver, relpath: str, cls: str | None = None):
        self.r = resolver
        self.dir = posixpath.dirname(relpath)
        self.cls = cls
        self.highlight = "gdscript"
        self.group = 0
        self.codes: list[tuple[str, str, int]] = []

    def take_codes(self):
        codes, self.codes = self.codes, []
        return codes

    def text(self, lines: list[str]) -> str:
        return tidy(self.convert(lines))

    # -- inline markup -----------------------------------------------------

    def inline(self, text: str) -> str:
        held: list[str] = []

        def hold(s: str) -> str:
            held.append(s)
            return f"\x00{len(held) - 1}\x00"

        text = LITERAL_RE.sub(lambda m: hold(f"`{m.group(1)}`"), text)
        text = ROLE_RE.sub(lambda m: hold(self.role(m.group("role"), m.group("body"))), text)
        text = EXTLINK_RE.sub(lambda m: hold(f"[{unescape(m.group(1)) or m.group(2)}]({m.group(2)})"), text)
        text = NAMEDREF_RE.sub(lambda m: unescape(m.group(1)), text)
        text = SUBST_RE.sub(lambda m: SUBSTITUTIONS.get(m.group(1), m.group(0)), text)
        text = unescape(text)
        for _ in range(3):  # placeholders can nest
            if "\x00" not in text:
                break
            text = PLACEHOLDER_RE.sub(lambda m: held[int(m.group(1))], text)
        return text

    def role(self, role: str, body: str) -> str:
        if role in ("ref", "doc", "download"):
            m = re.match(r"^(.*?)\s*<([^<>]+)>$", body, re.S)
            text, target = (unescape(m.group(1).strip()), m.group(2).strip()) if m else (None, body.strip())
            if text == "🔗":
                return ""
            if role == "download":
                return text or target
            if role == "doc":
                pid, title = self.r.resolve_doc(target, self.dir)
                return f"{text or title or target} [{pid}]" if pid else (text or target)
            hit = self.r.anchors.get(target.lower())
            if hit is None:
                return text or target
            eid, title, kind = hit
            self.r.refcount[eid] = self.r.refcount.get(eid, 0) + 1
            disp = text or title
            if eid.startswith("doc:"):
                return f"{disp} [{eid}]"
            if kind == "class":
                return disp
            cls = eid.split(".", 1)[0]
            if "." not in disp and cls != self.cls and cls not in ("@GlobalScope", "@GDScript"):
                disp = f"{cls}.{disp}"
            return f"`{disp}`"
        if role == "abbr":
            return unescape(re.sub(r"\s*\(.*\)\s*$", "", body, flags=re.S))
        if role in ("code", "samp", "file"):
            return f"`{body}`"
        return unescape(body)  # kbd, button, menu, ui, inspector, attr, meth, class, sub, sup, ...

    # -- block structure ---------------------------------------------------

    def convert(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        i, n = 0, len(lines)
        while i < n:
            line = lines[i]
            s = line.strip()
            if not s:
                out.append("")
                i += 1
                continue
            ind = indent_of(line)
            if s.startswith(".. ") or s == "..":
                m = DIRECTIVE_RE.match(s)
                block, i = take_block(lines, i + 1, ind)
                if m:
                    out.extend(indent_lines(self.directive(m.group(1).lower(), (m.group(2) or "").strip(), block), ind))
                continue  # labels, substitution definitions and comments are dropped
            if UNDERLINE_RE.match(s) or SIMPLE_TABLE_RE.match(s):
                i += 1  # transitions, overlines, simple-table borders
                continue
            if i + 1 < n and UNDERLINE_RE.match(lines[i + 1].strip()) and indent_of(lines[i + 1]) == ind:
                out += ["", f"**{self.inline(s)}**", ""]  # heading inside a block
                i += 2
                continue
            # Inline markup (roles, literals) may wrap onto the next line: join until backticks balance.
            while s.count("`") % 2 and i + 1 < n and lines[i + 1].strip() and not lines[i + 1].strip().startswith(".. "):
                i += 1
                s += " " + lines[i].strip()
            if s.startswith(("+-", "+=")) and s.endswith("+"):
                j = i
                while j < n and lines[j].strip()[:1] in ("+", "|"):
                    j += 1
                out.extend(indent_lines(self.grid_table(lines[i:j]), ind))
                i = j
                continue
            if s.endswith("::"):
                text = s[:-2].rstrip()
                if text:
                    out.append(" " * ind + self.inline(text if s.endswith(" ::") else text + ":"))
                j = i + 1
                while j < n and not lines[j].strip():
                    j += 1
                if j < n and indent_of(lines[j]) > ind:
                    block, i = take_block(lines, j, ind)
                    out.extend(indent_lines(self.code(self.highlight, dedent(trim_blank(block))), ind))
                else:
                    i += 1
                continue
            out.append(" " * ind + self.inline(s))
            i += 1
        return out

    def directive(self, name: str, arg: str, block: list[str]) -> list[str]:
        if name in SKIP_DIRECTIVES:
            return []
        opts, content = split_options(block)
        if name in ADMONITIONS or name == "admonition":
            head = ADMONITIONS.get(name) or self.inline(arg) or "Note"
            body = trim_blank(self.convert(([arg] if arg and name != "admonition" else []) + content))
            if not body:
                return []
            body[0] = f"**{head}:** {body[0].lstrip()}"
            return ["", *body, ""]
        if name in ("code-block", "code", "sourcecode"):
            return self.code(arg or self.highlight, content)
        if name == "highlight":
            self.highlight = arg or self.highlight
            return []
        if name == "tabs":
            self.r.group += 1
            saved, self.group = self.group, self.r.group
            body = self.convert(content)
            self.group = saved
            return body
        if name == "code-tab":
            return self.code(arg.split()[0] if arg else self.highlight, content, self.group)
        if name in ("tab", "group-tab"):
            return ["", f"*{self.inline(arg)}:*", "", *self.convert(content)] if arg else self.convert(content)
        if name == "toctree":
            return self.toctree(content, opts)
        if name == "rubric":
            return ["", f"**{self.inline(arg)}**", ""]
        if name == "math":
            return ["```math", *content, "```"]
        if name == "container" and "contribute" in arg:
            return []
        if name in ("list-table", "table", "csv-table"):
            return ([f"**{self.inline(arg)}**"] if arg else []) + self.convert(content)
        return self.convert(content)

    def code(self, lang: str, block: list[str], group: int = 0) -> list[str]:
        lang = (lang or "text").split()[0].lower()
        lang = LANG_NORMALIZE.get(lang, lang)
        block = trim_blank(block)
        if not block:
            return []
        self.codes.append((lang, "\n".join(block), group))
        return ["", f"```{lang}|g{group}" if group else f"```{lang}", *block, "```", ""]

    def toctree(self, content: list[str], opts: dict) -> list[str]:
        items = []
        for line in content:
            s = line.strip()
            if not s:
                continue
            m = re.match(r"^(.*?)\s*<([^>]+)>$", s)
            title, target = (m.group(1), m.group(2)) if m else (None, s)
            hits = self.r.glob_docs(target, self.dir) if "*" in target else [self.r.resolve_doc(target, self.dir)]
            for pid, ptitle in hits:
                if pid:
                    items.append(f"- {title or ptitle} [{pid}]")
        if not items:
            return []
        caption = opts.get("caption")
        return ["", f"**{caption}:**" if caption else "Contents:", *items, ""]

    def grid_table(self, lines: list[str]) -> list[str]:
        rows, cur = [], None
        for line in lines:
            s = line.strip()
            if s.startswith("+"):
                if cur:
                    rows.append(cur)
                cur = None
                continue
            cells = [c.strip() for c in s.strip("|").split("|")]
            cur = cells if cur is None else [f"{a} {b}".strip() for a, b in zip_longest(cur, cells, fillvalue="")]
        if cur:
            rows.append(cur)
        return ["", *("| " + " | ".join(self.inline(c) for c in row) + " |" for row in rows), ""]


def unescape(text: str) -> str:
    return re.sub(r"\\(.)", r"\1", text.replace("\\ ", ""))


def grid_table_cells(lines: list[str]) -> list[list[str]]:
    """Raw cell text of a grid table (multi-line cells merged)."""
    rows, cur = [], None
    for line in lines:
        s = line.strip()
        if not s:
            break
        if s.startswith("+"):
            if cur:
                rows.append(cur)
            cur = None
            continue
        if not s.startswith("|"):
            break
        cells = [c.strip() for c in s.strip("|").split("|")]
        cur = cells if cur is None else [f"{a} {b}".strip() for a, b in zip_longest(cur, cells, fillvalue="")]
    if cur:
        rows.append(cur)
    return rows


# ---------------------------------------------------------------------------
# Pass 1: document structure
# ---------------------------------------------------------------------------

MEMBER_RST = {
    "method": "method", "property": "property", "signal": "signal", "constant": "constant",
    "enumeration": "enum", "enumeration-constant": "constant", "themeproperty": "theme_item",
    "operator": "operator", "constructor": "constructor", "annotation": "annotation",
}
KIND_ORDER = ["property", "method", "signal", "enum", "constant", "annotation", "constructor", "operator", "theme_item"]
KIND_PLURAL = {
    "property": "Properties", "method": "Methods", "signal": "Signals", "enum": "Enumerations",
    "constant": "Constants", "annotation": "Annotations", "constructor": "Constructors",
    "operator": "Operators", "theme_item": "Theme properties",
}
MEMBER_KINDS = set(KIND_ORDER)


@dataclass
class Heading:
    start: int
    end: int
    level: int
    title: str
    id: str = ""


@dataclass
class PageDoc:
    relpath: str
    id: str
    title: str
    lines: list
    headings: list
    flags: set
    lang: str


@dataclass
class MemberPart:
    sig: str
    desc: list
    setget: list
    line: int


@dataclass
class Member:
    kind: str
    name: str
    id: str
    parts: list = field(default_factory=list)
    anchors: list = field(default_factory=list)
    enum: str = ""


@dataclass
class ClassDoc:
    relpath: str
    name: str
    line: int
    inherits: list
    inherited_by: list
    header: list
    description: list
    tutorials: list
    members: list


def read_lines(path: Path) -> list[str]:
    return [l.expandtabs(8).rstrip() for l in path.read_text(encoding="utf-8-sig").splitlines()]


def plain_title(s: str) -> str:
    s = re.sub(r":[\w-]+:`([^`<]*?)\s*<[^>]*>`", r"\1", s)
    s = re.sub(r":[\w-]+:`([^`]*)`", r"\1", s)
    s = re.sub(r"`([^`<]*?)\s*<[^>]*>`_*", r"\1", s)
    return unescape(s.replace("``", "")).strip()


def find_headings(lines: list[str]) -> list[Heading]:
    styles: dict = {}
    heads = []
    i = 0
    while i < len(lines) - 1:
        s, u = lines[i], lines[i + 1]
        if s and not s[0].isspace() and not s.startswith("..") and not UNDERLINE_RE.match(s) and UNDERLINE_RE.match(u):
            over = i > 0 and UNDERLINE_RE.match(lines[i - 1]) is not None and lines[i - 1][0] == u[0]
            if (i == 0 or not lines[i - 1].strip() or over) and len(u.strip()) >= len(s.strip()) - 1:
                level = styles.setdefault((u[0], over), len(styles) + 1)
                heads.append(Heading(start=i - 1 if over else i, end=i + 2, level=level, title=plain_title(s)))
                i += 2
                continue
        i += 1
    return heads


def page_lang(relpath: str) -> str:
    if "/c_sharp/" in relpath:
        return "csharp"
    if "/cpp/" in relpath or "/gdextension/" in relpath:
        return "cpp"
    if "/gdscript/" in relpath:
        return "gdscript"
    return ""


def parse_page(relpath: str, lines: list[str], resolver: Resolver) -> PageDoc:
    page_id = "doc:" + relpath[:-4]
    flags = set()
    for line in lines[:5]:
        if line.startswith(":article_outdated: True"):
            flags.add("outdated")
    heads = find_headings(lines)
    title = heads[0].title if heads else posixpath.basename(relpath[:-4])
    used: set = set()
    for k, h in enumerate(heads):
        if k == 0:
            h.id = page_id
            continue
        slug = base = slugify(h.title)
        n = 2
        while slug in used:
            slug, n = f"{base}-{n}", n + 1
        used.add(slug)
        h.id = f"{page_id}#{slug}"
    resolver.pages[relpath[:-4]] = (page_id, title)
    # Labels point at the next heading, or else at the section containing them.
    starts = {h.start: h for h in heads}
    for i, line in enumerate(lines):
        m = LABEL_RE.match(line)
        if not m:
            continue
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or LABEL_RE.match(lines[j])):
            j += 1
        target = starts.get(j)
        if target is None:
            target = next((h for h in reversed(heads) if h.start < i), None)
        tid, ttitle = (target.id, target.title) if target else (page_id, title)
        resolver.anchors[m.group(1).lower()] = (tid, ttitle, "doc")
    return PageDoc(relpath, page_id, title, lines, heads, flags, page_lang(relpath))


def parse_class(relpath: str, lines: list[str]) -> ClassDoc | None:
    # Underline is as long as the title, so two-letter classes (OS, IP) only get "==".
    title_at = next((i for i in range(len(lines) - 1) if lines[i].strip() and lines[i + 1].startswith("==")
                     and not lines[i + 1].strip("=")), None)
    if title_at is None:
        return None
    name = lines[title_at].strip()
    groups = [k for k, l in enumerate(lines) if l.startswith(".. rst-class:: classref-") and l.endswith("-group")]
    header_end = groups[0] if groups else len(lines)
    inherits, inherited_by, header = [], [], []
    for line in lines[title_at + 2: header_end]:
        if line.startswith("**Inherits:**"):
            inherits = re.findall(r"<class_([^>]+)>", line)
        elif line.startswith("**Inherited By:**"):
            inherited_by = re.findall(r"<class_([^>]+)>", line)
        else:
            header.append(line)
    cd = ClassDoc(relpath, name, title_at, inherits, inherited_by, trim_blank(header), [], [], [])
    used: set = set()
    for gi, start in enumerate(groups):
        end = groups[gi + 1] if gi + 1 < len(groups) else len(lines)
        heads = [k for k in range(start, min(end, start + 6)) if k + 1 < end and UNDERLINE_RE.match(lines[k + 1]) and lines[k].strip()]
        if not heads:
            continue
        gname = lines[heads[0]].strip()
        body = lines[heads[0] + 2: end]
        if gname == "Description":
            cd.description = trim_blank(body)
        elif gname == "Tutorials":
            cd.tutorials = trim_blank(body)
        elif "descriptions-group" in lines[start]:
            parse_members(cd, body, heads[0] + 2, used)
    return cd


def parse_members(cd: ClassDoc, body: list[str], offset: int, used: set):
    by_key: dict = {}
    pending_anchor, enum, cur, mode = None, "", None, "desc"
    i = 0
    while i < len(body):
        line = body[i]
        s = line.strip()
        m = LABEL_RE.match(s) if s.startswith(".. _") else None
        if m:
            pending_anchor = m.group(1)
            i += 1
            continue
        if s.startswith(".. rst-class:: classref-"):
            rc = s.split("classref-", 1)[1]
            if rc in MEMBER_RST:
                j = i + 1
                while j < len(body) and not body[j].strip():
                    j += 1
                sig = body[j].strip() if j < len(body) else ""
                nm = re.search(r"\*\*(.+?)\*\*(?!\*)", sig)
                name = unescape(nm.group(1)) if nm else sig
                kind = MEMBER_RST[rc]
                cur = MemberPart(sig, [], [], offset + j)
                key = (kind, name)
                member = by_key.get(key)
                if member is None:
                    mid = f"{cd.name}.{name.replace(' ', '')}"
                    if mid in used:
                        mid = f"{mid}#{kind}"
                    used.add(mid)
                    member = Member(kind, name, mid, enum=enum if rc == "enumeration-constant" else "")
                    by_key[key] = member
                    cd.members.append(member)
                member.parts.append(cur)
                if pending_anchor:
                    member.anchors.append(pending_anchor)
                pending_anchor = None
                if rc == "enumeration":
                    enum = name
                mode = "desc"
                i = j + 1
                continue
            if rc == "property-setget":
                mode = "setget"
            elif rc == "item-separator":
                cur, enum = None, ""
            i += 1
            continue
        if s == "----":
            i += 1
            continue
        if cur is not None:
            if mode == "setget":
                if s.startswith("- "):
                    cur.setget.append(s[2:])
                    i += 1
                    continue
                if s:
                    mode = "desc"
            if mode == "desc":
                cur.desc.append(line)
        i += 1


# ---------------------------------------------------------------------------
# Godot 3 -> 4 aliases
# ---------------------------------------------------------------------------

# (old, new, note, target id or rst label, trigger regex or "", extra search terms)
CURATED_ALIASES = [
    ("yield", "await", "GDScript 2.0 replaced `yield()` with the `await` keyword, e.g. `await get_tree().create_timer(1.0).timeout` or `await some_signal`.",
     "doc:tutorials/scripting/gdscript/gdscript_basics#awaiting-signals-or-coroutines", r"\byield\b", "await signal coroutine"),
    ("export", "@export", "Godot 4 uses annotations: `@export var speed: float = 10.0` (also `@export_range`, `@export_enum`, ...).",
     "doc_gdscript_exports", r"(?<![@\w])export\s*(\(|var\b)", "@export"),
    ("onready", "@onready", "Godot 4 uses the `@onready` annotation: `@onready var label = $Label`.",
     "@GDScript.@onready", r"(?<![@\w])onready\b", "@onready"),
    ("tool", "@tool", "Godot 4 uses the `@tool` annotation at the top of the script.",
     "@GDScript.@tool", r"(?<![@\w])tool(?=\s*(extends|$))", "@tool"),
    ("setget", "set/get property syntax", "`setget` was removed. Use `var x: int: set = _set_x, get = _get_x` or inline `set(value):` / `get:` blocks.",
     "doc_gdscript_basics_setters_getters", r"\bsetget\b", "setter getter property"),
    ("remote/master/puppet", "@rpc", "RPC keywords (`remote`, `master`, `puppet`, `remotesync`, ...) were replaced by the `@rpc` annotation, e.g. `@rpc(\"any_peer\", \"call_local\")`.",
     "@GDScript.@rpc", r"\b(master|puppet|remotesync|puppetsync|mastersync)\b|\bremote\s+func\b", "@rpc"),
    ("connect(\"signal\", obj, \"method\")", "signal.connect(callable)", "Godot 4 connects signals with Callables: `button.pressed.connect(_on_pressed)`.",
     "doc_signals", r"connect\(\s*['\"]\w+['\"]\s*,", "signal connect callable"),
    ("move_and_slide(velocity)", "move_and_slide()", "In Godot 4 `move_and_slide()` takes no arguments: set the `velocity` property first; `up_direction` and the floor options are properties too.",
     "CharacterBody2D.move_and_slide", r"move_and_slide\s*\(\s*[^)\s]", "move_and_slide velocity"),
    ("intersect_ray(from, to)", "intersect_ray(PhysicsRayQueryParameters2D)", "Godot 4 raycasts take a query object: `space_state.intersect_ray(PhysicsRayQueryParameters2D.create(from, to))`.",
     "PhysicsDirectSpaceState2D.intersect_ray", r"intersect_ray\s*\(\s*[^)\s]+\s*,", "intersect_ray PhysicsRayQueryParameters2D"),
]
# (old, new) renames of global functions, classes and members that are missing from the migration guide.
CURATED_RENAMES = [
    ("deg2rad", "@GlobalScope.deg_to_rad"), ("rad2deg", "@GlobalScope.rad_to_deg"),
    ("rand_range", "@GlobalScope.randf_range"), ("stepify", "@GlobalScope.snapped"),
    ("range_lerp", "@GlobalScope.remap"), ("linear2db", "@GlobalScope.linear_to_db"),
    ("db2linear", "@GlobalScope.db_to_linear"), ("str2var", "@GlobalScope.str_to_var"),
    ("var2str", "@GlobalScope.var_to_str"), ("bytes2var", "@GlobalScope.bytes_to_var"),
    ("var2bytes", "@GlobalScope.var_to_bytes"), ("to_json", "JSON.stringify"),
    ("parse_json", "JSON.parse_string"), ("funcref", "Callable"), ("convert", "@GlobalScope.type_convert"),
    ("instance", "PackedScene.instantiate"), ("get_world", "Node3D.get_world_3d"),
    ("is_network_master", "Node.is_multiplayer_authority"), ("get_network_master", "Node.get_multiplayer_authority"),
    ("set_network_master", "Node.set_multiplayer_authority"), ("editor_hint", "Engine.is_editor_hint"),
    ("OS.get_ticks_msec", "Time.get_ticks_msec"), ("OS.get_ticks_usec", "Time.get_ticks_usec"),
    ("OS.get_unix_time", "Time.get_unix_time_from_system"), ("OS.get_screen_size", "DisplayServer.screen_get_size"),
    ("OS.window_size", "DisplayServer.window_get_size"), ("OS.window_fullscreen", "DisplayServer.window_set_mode"),
    ("rect_position", "Control.position"), ("rect_size", "Control.size"), ("rect_min_size", "Control.custom_minimum_size"),
    ("rect_global_position", "Control.global_position"), ("rect_rotation", "Control.rotation"),
    ("rect_scale", "Control.scale"), ("rect_pivot_offset", "Control.pivot_offset"),
    ("BUTTON_LEFT", "@GlobalScope.MOUSE_BUTTON_LEFT"), ("BUTTON_RIGHT", "@GlobalScope.MOUSE_BUTTON_RIGHT"),
    ("BUTTON_MIDDLE", "@GlobalScope.MOUSE_BUTTON_MIDDLE"), ("BUTTON_WHEEL_UP", "@GlobalScope.MOUSE_BUTTON_WHEEL_UP"),
    ("BUTTON_WHEEL_DOWN", "@GlobalScope.MOUSE_BUTTON_WHEEL_DOWN"), ("KEY_CONTROL", "@GlobalScope.KEY_CTRL"),
    ("interpolate_property", "Tween.tween_property"), ("File", "FileAccess"), ("Directory", "DirAccess"),
    ("Reference", "RefCounted"), ("PoolByteArray", "PackedByteArray"), ("PoolIntArray", "PackedInt32Array"),
    ("PoolRealArray", "PackedFloat32Array"), ("PoolStringArray", "PackedStringArray"),
    ("PoolVector2Array", "PackedVector2Array"), ("PoolVector3Array", "PackedVector3Array"),
    ("PoolColorArray", "PackedColorArray"), ("KinematicCollision", "KinematicCollision3D"),
]


@dataclass
class Alias:
    old: str
    new: str
    target: str
    note: str
    kind: str  # class | member | syntax
    strict: bool
    trigger: str = ""
    terms: str = ""


def is_plain_word(s: str) -> bool:
    return re.fullmatch(r"[A-Za-z][a-z]*", s) is not None


def build_aliases(docs_root: Path, class_names: set, ids: set, resolver: Resolver) -> list[Alias]:
    aliases: list[Alias] = []
    seen: set = set()
    lower_classes = {c.lower(): c for c in class_names}

    def add(a: Alias):
        key = (a.old.lower(), a.target or a.new)
        if key not in seen and a.old.lower() != a.new.lower():
            seen.add(key)
            aliases.append(a)

    mig = docs_root / "tutorials/migrating/upgrading_to_godot_4.rst"
    lines = read_lines(mig) if mig.is_file() else []
    # Node/resource rename tables: | Old name | New name | [comment]
    for i, line in enumerate(lines):
        if not line.startswith("+-") or (i > 0 and lines[i - 1].startswith(("+", "|"))):
            continue
        rows = grid_table_cells(lines[i:])
        if not rows or not re.search(r"old name|removed node", rows[0][0], re.I):
            continue
        for row in rows[1:]:
            if len(row) < 2 or not re.fullmatch(r"\w+", row[0]):
                continue
            new_names = [w for w in re.findall(r"\w+", row[1]) if w in class_names]
            note = unescape(re.sub(r":ref:`([^`<]*?)\s*<[^>]+>`|:ref:'[^']*'", r"\1", row[2])) if len(row) > 2 else ""
            note = f"`{row[0]}` (Godot 3) -> `{row[1]}` in Godot 4." + (f" {note}" if note else "")
            add(Alias(row[0], row[1], new_names[0] if new_names else "", note, "class", is_plain_word(row[0])))
    # "- Class's ``old()`` is now ``new()``" lists
    rename_re = re.compile(r"^-\s+(?P<cls>[\w ,]+?)'s?\s+``(?P<old>[\w.]+)(?:\(\))?``\s+is now\s+``(?P<new>[\w.]+)(?:\(\))?``")
    for line in lines:
        m = rename_re.match(line)
        if not m:
            continue
        for c in re.split(r",\s*|\s+and\s+", m.group("cls")):
            cname = lower_classes.get(c.strip().lower(), c.strip())
            target = f"{cname}.{m.group('new')}"
            add(Alias(m.group("old"), target, target if target in ids else "",
                      f"`{c.strip()}.{m.group('old')}` (Godot 3) is `{target}` in Godot 4.", "member", is_plain_word(m.group("old"))))
    # Nodes that only gained a "3D" suffix.
    for c in sorted(class_names):
        if c.endswith("3D") and c[:-2] and c[:-2] not in class_names and c[:-2].lower() not in seen_olds(aliases):
            add(Alias(c[:-2], c, c, f"`{c[:-2]}` (Godot 3) is `{c}` in Godot 4.", "class", is_plain_word(c[:-2])))
    for old, new in CURATED_RENAMES:
        target = new if new in ids else ""
        kind = "class" if new in class_names else "member"
        add(Alias(old, new, target, f"`{old}` (Godot 3) is `{new}` in Godot 4.", kind, is_plain_word(old.split(".")[-1])))
    for old, new, note, target, trigger, terms in CURATED_ALIASES:
        if target not in ids:
            hit = resolver.anchors.get(target.lower())
            target = hit[0] if hit else ""
        add(Alias(old, new, target, note, "syntax", False, trigger, terms))
    return aliases


def seen_olds(aliases: list[Alias]) -> set:
    return {a.old.lower() for a in aliases}


# ---------------------------------------------------------------------------
# Pass 2: entries and index build
# ---------------------------------------------------------------------------


@dataclass
class Entry:
    id: str
    kind: str
    name: str
    title: str
    cls: str = ""
    parent: str = ""
    signature: str = ""
    summary: str = ""
    body: str = ""
    source: str = ""
    flags: str = ""
    lang: str = ""
    level: int = 0
    keywords: str = ""
    fts_title: str = ""
    extra: dict = field(default_factory=dict)
    codes: list = field(default_factory=list)


def format_signature(conv: RstConverter, raw: str, kind: str) -> tuple[str, list[str]]:
    quals = [q for q in QUALIFIERS if f"|{q}|" in raw]
    for q in QUALIFIERS:
        raw = raw.replace(f"|{q}|", "")
    t = conv.inline(raw).replace("`", "")
    t = re.sub(r"\s+", " ", t).strip()
    m = re.match(r"^(?P<ret>.*?)\s*\*\*(?P<name>.+?)\*\*(?!\*)\s*(?P<rest>.*)$", t)
    if not m:
        return t, quals
    ret, name, rest = m.group("ret").strip(), m.group("name"), m.group("rest").strip()
    if kind in ("property", "theme_item"):
        sig = f"{name}: {ret}" + (f" {rest}" if rest else "")
    elif kind == "constant":
        sig = f"{name} {rest}".strip()
    elif kind == "enum":
        sig = f"{ret} {name}".strip()
    elif kind == "constructor":
        sig = f"{name}{rest}"
    else:
        sig = f"{name}{rest}" + (f" -> {ret}" if ret else "")
    if quals:
        sig += " [" + ", ".join(quals) + "]"
    return sig, quals


def class_entries(cd: ClassDoc, resolver: Resolver, alias_kw: dict) -> list[Entry]:
    conv = RstConverter(resolver, cd.relpath, cd.name)
    header = conv.text(cd.header)
    flags = []
    if "**Deprecated:**" in header:
        flags.append("deprecated")
    if "**Experimental:**" in header:
        flags.append("experimental")
    brief_lines = [l for l in header.split("\n") if not l.startswith(("**Deprecated:**", "**Experimental:**"))]
    brief = tidy(brief_lines)
    desc = conv.text(cd.description)
    body = header
    if desc:
        body += "\n\n## Description\n\n" + desc
    tuts = conv.text(cd.tutorials)
    if tuts:
        body += "\n\n## Tutorials\n\n" + tuts
    entries = [Entry(
        id=cd.name, kind="class", name=cd.name, title=cd.name, cls=cd.name,
        signature=" < ".join([cd.name] + cd.inherits), summary=first_sentence(brief or desc),
        body=body, source=f"{cd.relpath}:{cd.line + 1}", flags=",".join(flags),
        keywords=" ".join(split_ident(cd.name) + alias_kw.get(cd.name, [])), fts_title=cd.name,
        extra={"inherits": cd.inherits, "inherited_by": cd.inherited_by}, codes=conv.take_codes(),
    )]
    enum_values: dict = {}
    for mem in cd.members:
        sigs, bodies, setget, quals_all = [], [], [], []
        for part in mem.parts:
            sig, quals = format_signature(conv, part.sig, mem.kind)
            sigs.append(sig)
            quals_all += quals
            text = conv.text(part.desc)
            bodies.append(text if len(mem.parts) == 1 else f"`{sig}`\n\n{text}".rstrip())
            setget += [format_signature(conv, sg, "method")[0] for sg in part.setget]
        body = "\n\n".join(b for b in bodies if b)
        mflags = sorted(set(quals_all))
        if "**Deprecated:**" in body:
            mflags.append("deprecated")
        if "**Experimental:**" in body:
            mflags.append("experimental")
        kw = split_ident(mem.name) + [re.match(r"[\w@]+", s).group(0) for s in setget if re.match(r"[\w@]+", s)]
        kw += alias_kw.get(mem.id, [])
        extra = {}
        if setget:
            extra["setget"] = setget
        if mem.enum:
            extra["enum"] = mem.enum
            enum_values.setdefault(mem.enum, []).append((sigs[0], first_sentence(body, 120)))
            kw += split_ident(mem.enum)
        entries.append(Entry(
            id=mem.id, kind=mem.kind, name=mem.name, title=mem.id, cls=cd.name, parent=cd.name,
            signature="\n".join(sigs), summary=first_sentence(body), body=body,
            source=f"{cd.relpath}:{mem.parts[0].line + 1}", flags=",".join(mflags),
            keywords=" ".join(kw), fts_title=f"{cd.name} {mem.name}", extra=extra, codes=conv.take_codes(),
        ))
    for e in entries:
        if e.kind == "enum" and e.name in enum_values:
            e.extra["values"] = enum_values[e.name]
    return entries


def page_entries(pd: PageDoc, resolver: Resolver) -> list[Entry]:
    conv = RstConverter(resolver, pd.relpath)
    words = [w for w in re.split(r"[/_\-]+", pd.relpath[:-4]) if w not in ("tutorials", "index")]
    heads = pd.headings
    entries = []
    if not heads:
        body = conv.text(pd.lines)
        return [Entry(id=pd.id, kind="page", name=pd.title, title=pd.title, body=body,
                      summary=first_sentence(body), source=f"{pd.relpath}:1", flags=",".join(sorted(pd.flags)),
                      lang=pd.lang, level=1, keywords=" ".join(words), fts_title=pd.title, codes=conv.take_codes())]
    crumbs: dict = {}
    for k, h in enumerate(heads):
        end = heads[k + 1].start if k + 1 < len(heads) else len(pd.lines)
        body = conv.text(pd.lines[h.end:end] if k else pd.lines[:h.start] + pd.lines[h.end:end])
        crumbs[h.level] = h.title
        crumbs = {lv: t for lv, t in crumbs.items() if lv <= h.level}
        is_page = k == 0
        title = pd.title if is_page else " > ".join(crumbs[lv] for lv in sorted(crumbs))
        entries.append(Entry(
            id=h.id, kind="page" if is_page else "section", name=h.title, title=title,
            parent="" if is_page else pd.id, body=body, summary=first_sentence(body),
            source=f"{pd.relpath}:{h.start + 1}", flags=",".join(sorted(pd.flags)), lang=pd.lang,
            level=h.level, keywords=" ".join(words) if is_page else "", fts_title=title,
            codes=conv.take_codes(),
        ))
    if not entries[0].summary:  # page that starts directly with a section
        entries[0].summary = next((e.summary for e in entries[1:] if e.summary), "")
    return entries


SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE entries(
    n INTEGER PRIMARY KEY, id TEXT NOT NULL, kind TEXT NOT NULL, name TEXT, cls TEXT, parent TEXT,
    title TEXT, signature TEXT, summary TEXT, body TEXT, source TEXT, flags TEXT, lang TEXT,
    level INTEGER, pop INTEGER, extra TEXT);
CREATE UNIQUE INDEX entries_id ON entries(id);
CREATE INDEX entries_id_nc ON entries(id COLLATE NOCASE);
CREATE INDEX entries_name ON entries(name COLLATE NOCASE);
CREATE INDEX entries_cls ON entries(cls COLLATE NOCASE, n);
CREATE INDEX entries_parent ON entries(parent, n);
CREATE TABLE classes(name TEXT PRIMARY KEY COLLATE NOCASE, inherits TEXT, inherited_by TEXT, brief TEXT);
CREATE TABLE anchors(label TEXT PRIMARY KEY, id TEXT);
CREATE TABLE aliases(old TEXT, new TEXT, target TEXT, note TEXT, kind TEXT, strict INTEGER, trigger TEXT, terms TEXT);
CREATE TABLE code(n INTEGER PRIMARY KEY, entry INTEGER, lang TEXT, grp INTEGER, code TEXT);
CREATE VIRTUAL TABLE fts USING fts5(name, keywords, title, signature, summary, body, code, code_cs,
    tokenize='porter unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE code_fts USING fts5(code, context, tokenize='porter unicode61 remove_diacritics 2');
"""


def collect_files(root: Path) -> list[str]:
    rels = []
    for p in root.rglob("*.rst"):
        rel = p.relative_to(root).as_posix()
        if any(part.startswith((".", "_")) for part in rel.split("/")) or rel == "404.rst":
            continue
        rels.append(rel)
    return sorted(rels)


def build_index(docs_root: Path, index_path: Path, log=lambda m: print(m, file=sys.stderr)) -> dict:
    t0 = time.time()
    log(f"[godot_docs] Building index from {docs_root} ...")
    resolver = Resolver()
    classes, pages = [], []
    for rel in collect_files(docs_root):
        lines = read_lines(docs_root / rel)
        if rel.startswith("classes/class_"):
            cd = parse_class(rel, lines)
            if cd:
                classes.append(cd)
                continue
        pages.append(parse_page(rel, lines, resolver))
    class_names = {c.name for c in classes}
    ids = set(class_names)
    for cd in classes:
        resolver.anchors[f"class_{cd.name}".lower()] = (cd.name, cd.name, "class")
        for mem in cd.members:
            ids.add(mem.id)
            for a in mem.anchors:
                resolver.anchors[a.lower()] = (mem.id, mem.name, mem.kind)
    for pd in pages:
        ids.update(h.id for h in pd.headings)

    aliases = build_aliases(docs_root, class_names, ids, resolver)
    alias_kw: dict = {}
    for a in aliases:
        if a.target and a.kind in ("class", "member"):
            alias_kw.setdefault(a.target, []).append(a.old.split(".")[-1])

    entries: list[Entry] = []
    for cd in classes:
        entries += class_entries(cd, resolver, alias_kw)
    for pd in pages:
        entries += page_entries(pd, resolver)

    tmp = index_path.with_name(f"{index_path.name}.tmp{os.getpid()}")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if tmp.exists():
        tmp.unlink()
    conn = sqlite3.connect(tmp)
    conn.executescript(SCHEMA)
    code_n = 0
    for n, e in enumerate(entries, 1):
        conn.execute(
            "INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n, e.id, e.kind, e.name, e.cls, e.parent, e.title, e.signature, e.summary, e.body,
             e.source, e.flags, e.lang, e.level, resolver.refcount.get(e.id, 0), json.dumps(e.extra) if e.extra else ""),
        )
        code_main = "\n".join(c for lang, c, _ in e.codes if lang != "csharp")
        code_cs = "\n".join(c for lang, c, _ in e.codes if lang == "csharp")
        conn.execute(
            "INSERT INTO fts(rowid, name, keywords, title, signature, summary, body, code, code_cs) VALUES (?,?,?,?,?,?,?,?,?)",
            (n, e.name, e.keywords, e.fts_title, e.signature, e.summary, strip_code(e.body), code_main, code_cs),
        )
        for lang, code, grp in e.codes:
            code_n += 1
            conn.execute("INSERT INTO code VALUES (?,?,?,?,?)", (code_n, n, lang, grp, code))
            conn.execute("INSERT INTO code_fts(rowid, code, context) VALUES (?,?,?)", (code_n, code, f"{e.title} {e.name}"))
    for cd in classes:
        brief = next((e.summary for e in entries if e.id == cd.name), "")
        conn.execute("INSERT INTO classes VALUES (?,?,?,?)", (cd.name, json.dumps(cd.inherits), json.dumps(cd.inherited_by), brief))
    conn.executemany("INSERT OR REPLACE INTO anchors VALUES (?,?)", [(k, v[0]) for k, v in resolver.anchors.items()])
    conn.executemany("INSERT INTO aliases VALUES (?,?,?,?,?,?,?,?)",
                     [(a.old, a.new, a.target, a.note, a.kind, int(a.strict), a.trigger, a.terms) for a in aliases])
    conn.execute("INSERT INTO fts(fts) VALUES ('optimize')")
    conn.execute("INSERT INTO code_fts(code_fts) VALUES ('optimize')")
    meta = {
        "schema": SCHEMA_VERSION, "tool": TOOL_HASH, "fingerprint": docs_fingerprint(docs_root),
        "godot_version": read_godot_version(docs_root), "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "docs_root": str(docs_root), "entries": str(len(entries)), "classes": str(len(classes)),
        "pages": str(len(pages)), "code_blocks": str(code_n), "aliases": str(len(aliases)),
    }
    conn.executemany("INSERT INTO meta VALUES (?,?)", meta.items())
    conn.commit()
    conn.close()
    os.replace(tmp, index_path)
    log(f"[godot_docs] Indexed {len(entries)} entries ({len(classes)} classes, {len(pages)} pages) in {time.time() - t0:.1f}s")
    return meta


# ---------------------------------------------------------------------------
# Index access
# ---------------------------------------------------------------------------


class DocsError(Exception):
    pass


class Index:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.meta = {r["key"]: r["value"] for r in self.conn.execute("SELECT key, value FROM meta")}
        self.total = int(self.meta.get("entries", "1"))
        self._aliases = None
        self._concepts = None
        # Deprecated classes -> their deprecation note. Their members aren't flagged in the index itself, so
        # flags_of() and deprecation_note() pass the class's status on to them at query time.
        self.deprecated: dict = {}
        for r in self.conn.execute("SELECT id, body FROM entries WHERE kind = 'class' AND flags LIKE '%deprecated%'"):
            line = next((l for l in (r["body"] or "").split("\n") if l.startswith("**Deprecated:**")), "")
            self.deprecated[r["id"]] = line.removeprefix("**Deprecated:**").strip()

    def flags_of(self, row) -> list[str]:
        """The entry's flags, plus `deprecated` for members of a deprecated class."""
        flags = [f for f in (row["flags"] or "").split(",") if f]
        if "deprecated" not in flags and row["kind"] not in ("class", "page", "section") and row["cls"] in self.deprecated:
            flags.append("deprecated")
        return flags

    def deprecation_note(self, row) -> str:
        """For a member of a deprecated class: one line naming the class and its replacement, else ''."""
        if row["kind"] in ("class", "page", "section") or row["cls"] not in self.deprecated:
            return ""
        return f"{row['cls']} is deprecated, so this member is too. {first_sentence(self.deprecated[row['cls']])} (details: read {row['cls']})"

    @property
    def version(self) -> str:
        return self.meta.get("godot_version", "?")

    def get(self, eid: str):
        return self.conn.execute("SELECT * FROM entries WHERE id = ?", (eid,)).fetchone()

    def get_nocase(self, eid: str):
        return self.conn.execute("SELECT * FROM entries WHERE id = ? COLLATE NOCASE", (eid,)).fetchone()

    def by_rowids(self, ns) -> dict:
        ns = list(ns)
        if not ns:
            return {}
        rows = self.conn.execute(f"SELECT * FROM entries WHERE n IN ({','.join('?' * len(ns))})", ns)
        return {r["n"]: r for r in rows}

    def class_row(self, name: str):
        return self.conn.execute("SELECT * FROM classes WHERE name = ?", (name,)).fetchone()

    def ancestors(self, name: str) -> list[str]:
        row = self.class_row(name)
        return json.loads(row["inherits"]) if row else []

    def aliases(self) -> list[sqlite3.Row]:
        if self._aliases is None:
            self._aliases = self.conn.execute("SELECT * FROM aliases").fetchall()
        return self._aliases

    @property
    def concepts(self) -> "Concepts | None":
        if self._concepts is None:
            self._concepts = Concepts.load(self, CONCEPTS_FILE) or False
        return self._concepts or None

    def extra_passages(self) -> tuple[list[int], list[str], str]:
        """Concept phrases as extra embedding passages of their targets: (entry numbers, texts, signature)."""
        cx = self.concepts
        return (cx.passages() + (cx.signature,)) if cx else ([], [], "")

    def name_exists(self, name: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM entries WHERE name = ? COLLATE NOCASE AND kind NOT IN ('page','section') LIMIT 1", (name,)
        ).fetchone() is not None


def read_index_meta(index_path: Path) -> dict | None:
    """The index's meta table, or None if there is no usable index."""
    if not index_path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
        try:
            return dict(conn.execute("SELECT key, value FROM meta").fetchall())
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _warned_path(index_path: Path) -> Path:
    return index_path.with_name(index_path.name + ".warned")


def rebuild_or_keep(docs_root: Path | None, index_path: Path, meta: dict | None, force: bool = False):
    """Rebuild the index. If that is impossible or fails while an index already exists, keep using the existing
    one (it is complete for the docs it was built from) and print a warning instead of failing.

    Automatic rebuilds warn once per tool/docs version and don't retry a rebuild that already failed; the state
    is remembered in `<index>.warned`. `force` (index --rebuild) always tries and always reports."""
    if meta is None and docs_root is None:
        raise DocsError("No index found and no godot-docs checkout located. Pass --docs PATH or set GODOT_DOCS_PATH.")
    state = f"{TOOL_HASH}|{docs_fingerprint(docs_root) if docs_root else 'no-docs'}"
    try:
        known = _warned_path(index_path).read_text(encoding="utf-8") == state
    except OSError:
        known = False
    if meta is not None and known and not force:
        return
    reason = "no godot-docs checkout found (pass --docs PATH or set GODOT_DOCS_PATH to rebuild)"
    if docs_root is not None:
        try:
            build_index(docs_root, index_path)
            return
        except Exception as exc:  # noqa: BLE001 - any build failure falls back to the existing index
            try:
                index_path.with_name(f"{index_path.name}.tmp{os.getpid()}").unlink(missing_ok=True)
            except OSError:
                pass
            if meta is None:
                raise DocsError(f"Building the index failed: {type(exc).__name__}: {exc}") from exc
            reason = f"reindexing failed ({type(exc).__name__}: {exc})"
    print(f"[godot_docs] WARNING: {reason}. Using the existing index, built {meta.get('built_at', '?')} from the "
          f"Godot {meta.get('godot_version', '?')} docs; searches work normally.", file=sys.stderr)
    try:
        _warned_path(index_path).write_text(state, encoding="utf-8")
    except OSError:
        pass


def open_index(args) -> Index:
    index_path = default_index_path(getattr(args, "index", None))
    docs_root = find_docs_root(getattr(args, "docs", None))
    meta = read_index_meta(index_path)
    stale = meta is None or (meta.get("schema"), meta.get("tool")) != (SCHEMA_VERSION, TOOL_HASH) or (
        docs_root is not None and meta.get("fingerprint") != docs_fingerprint(docs_root))
    if stale:
        rebuild_or_keep(docs_root, index_path, meta)
    return Index(index_path)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

STOPWORDS = set("""
a an the to of in on for with and or how do does did doing done i im my me we our us you your is are was were be been
being can could should would will shall what whats when where which who whom why it its this that these those there
then than from by as into about via vs using use uses want need way ways some any all also just like so if not no
godot engine gdscript
""".split()) - {"gdscript"}
TOKEN_RE = re.compile(r"@?[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_@][A-Za-z0-9_]*)*(?:\(\))?|\d+[A-Za-z]+")
KIND_GROUPS = {
    "api": ["class", *KIND_ORDER], "member": KIND_ORDER, "members": KIND_ORDER,
    "docs": ["page", "section"], "doc": ["page", "section"], "manual": ["page", "section"],
    "methods": ["method"], "properties": ["property"], "signals": ["signal"], "constants": ["constant"],
    "classes": ["class"], "theme": ["theme_item"], "pages": ["page"], "sections": ["section"],
}
KIND_PRIOR = {
    "class": 1.0, "method": 1.0, "property": 0.97, "signal": 0.95, "annotation": 1.0, "enum": 0.85,
    "constant": 0.7, "constructor": 0.8, "operator": 0.5, "theme_item": 0.65, "page": 1.0, "section": 1.0,
}
# Big families of niche classes that otherwise flood results: (prefix, multiplier, query words lifting it).
FAMILY_PENALTIES = [
    ("VisualShaderNode", 0.55, {"visual", "visualshader"}),
    ("Editor", 0.7, {"editor", "plugin", "plugins", "tool", "inspector", "gizmo", "importer", "dock"}),
    ("OpenXR", 0.6, {"xr", "openxr", "vr", "ar"}),
    ("XR", 0.8, {"xr", "openxr", "vr", "ar"}),
    ("GLTF", 0.75, {"gltf", "glb"}),
    ("TextServer", 0.6, {"textserver", "shaping"}),
    ("RenderingDevice", 0.75, {"renderingdevice", "rd", "compute"}),
    ("AudioEffect", 0.7, {"bus", "reverb", "delay", "filter", "eq", "compressor", "distortion", "chorus", "limiter", "spectrum", "record", "capture", "pitch"}),
    ("AnimationNode", 0.85, {"animationtree", "blend", "blendspace", "statemachine"}),
    ("GDExtension", 0.7, {"gdextension", "extension"}),
]
PATH_PENALTIES = [
    ("doc:engine_details/", 0.5, {"engine", "c++", "cpp", "compile", "compiling", "scons", "module", "modules", "contribute", "contributing", "source"}),
    ("doc:about/", 0.7, {"license", "faq", "release", "about", "system", "requirements"}),
    ("doc:community/", 0.6, {"community", "contribute", "contributing"}),
]
LANG_PENALTY = {"csharp": 0.45, "gdscript": 0.6, "cpp": 0.6}
# Everyday game-dev words -> the vocabulary the docs actually use. Alternatives count as one query concept.
SYNONYMS = {
    "keyboard": ["key", "keys", "input"], "keypress": ["key", "input"], "player": ["character"],
    "shoot": ["bullet", "projectile"], "shooting": ["bullet", "projectile"], "bullet": ["projectile"],
    "sound": ["audio"], "music": ["audio"], "sfx": ["audio"], "click": ["mouse", "pressed"],
    "gamepad": ["joypad", "controller"], "controller": ["joypad"], "joystick": ["joypad"],
    "collide": ["collision"], "colliding": ["collision"], "spawn": ["instantiate", "instance"],
    "spawning": ["instantiate", "instance"], "destroy": ["queue_free", "free"], "delete": ["queue_free", "free"],
    "wait": ["await", "timer"], "delay": ["timer", "await"], "sleep": ["timer", "await"],
    "hud": ["ui", "control", "canvaslayer"], "gui": ["ui", "control"], "menu": ["ui", "control"],
    "global": ["autoload", "singleton"], "singleton": ["autoload"], "autoload": ["singleton"],
    "tilemap": ["tilemaplayer"], "tiles": ["tilemaplayer", "tileset"], "spritesheet": ["spriteframes", "hframes"],
    "network": ["multiplayer"], "online": ["multiplayer", "network"], "pathfinding": ["navigation", "astar"],
    "navmesh": ["navigation"], "fps": ["framerate", "frames"], "framerate": ["fps"], "lag": ["performance"],
    "optimize": ["optimization", "performance"], "random": ["randf", "randi", "randomnumbergenerator"],
    "phone": ["mobile", "android"], "browser": ["web", "html5"], "dialog": ["dialogue"], "dialogue": ["dialog"],
    "raycast": ["raycast2d", "intersect_ray", "ray"], "healthbar": ["progressbar"],
    "move": ["movement"], "movement": ["move"], "moving": ["movement"],
}
# Everyday game words -> the node types they usually mean in Godot. "2d"/"3d" in the query keeps one dimension.
_CHAR = ["CharacterBody2D", "CharacterBody3D"]
_AREA = ["Area2D", "Area3D"]
TYPE_HINTS = {
    **dict.fromkeys(["sprite", "sprites"], ["Sprite2D", "AnimatedSprite2D", "Sprite3D", "AnimatedSprite3D"]),
    **dict.fromkeys(["character", "characters", "player", "players", "enemy", "enemies", "npc", "npcs",
                     "mob", "mobs"], _CHAR),
    **dict.fromkeys(["item", "items", "pickup", "pickups", "coin", "coins", "collectible", "collectibles",
                     "trigger", "zone", "hitbox", "hurtbox"], _AREA),
    **dict.fromkeys(["bullet", "bullets", "projectile", "projectiles"], _AREA),
    **dict.fromkeys(["wall", "walls", "floor", "ground"], ["StaticBody2D", "StaticBody3D"]),
}
# "object" alone is too vague; with a physics word it usually means a rigid body.
PHYSICS_WORDS = {"fall", "falls", "falling", "bounce", "bounces", "push", "pushed", "force", "physics", "gravity",
                 "roll", "rolling", "throw", "thrown", "impulse", "realistic", "realistically"}
OBJECT_PHYSICS = ["RigidBody2D", "RigidBody3D"]
# "off", "syn" (class name counts as matching the word) or "hint" (class searched + boosted like a named class)
TYPE_HINT_MODE = os.environ.get("GODOT_DOCS_TYPE_HINTS", "members")  # chosen on dev+val 2026-10-07
TYPE_HINT_BOOST = 0.1


def type_hints(words: set) -> list[str]:
    classes = [c for w in words for c in TYPE_HINTS.get(w, [])]
    if words & {"object", "objects", "thing", "things"} and words & PHYSICS_WORDS:
        classes += OBJECT_PHYSICS
    if "3d" in words and "2d" not in words:
        classes = [c for c in classes if not c.endswith("2D")]
    elif "2d" in words and "3d" not in words:
        classes = [c for c in classes if not c.endswith("3D")]
    return list(dict.fromkeys(classes))
# Multi-word phrases: alternatives are attached to the phrase's last word.
PHRASE_SYNONYMS = {
    "health bar": ["progressbar", "textureprogressbar"], "progress bar": ["progressbar"],
    "sprite sheet": ["spriteframes", "hframes", "animatedsprite2d"], "screen shake": ["camera2d", "offset"],
    "main menu": ["control", "ui"], "game over": ["reload_current_scene", "change_scene_to_file"],
}


# Concept glossary (concepts.json): other engines' words and everyday gameplay phrasing -> the Godot entries they
# mean ("prefab" -> PackedScene.instantiate, "how far away" -> Vector2.distance_to). Each phrase acts as extra text of
# its targets: keyword-like for lexical ranking, an extra embedded passage for semantic/hybrid.
CONCEPTS_FILE = Path(os.environ.get("GODOT_DOCS_CONCEPTS", DATA_DIR / "concepts.json"))
CONCEPT_WEIGHT = 0.5  # bonus for a target of a fully matched phrase (train/results/sweep_concepts.log: flat 0.3-0.8)
CONCEPT_DECAY = 0.2  # the bonus shrinks by this much per position in the target list
CONCEPT_RARE_DF = 30  # a one-word phrase counts as a full match only if the docs barely use the word


class Concepts:
    def __init__(self, ix: "Index", data: dict, signature: str):
        self.signature = signature
        self.concepts = []  # [{"targets": [(n, id)], "note": str}]
        self.phrases = []  # [(concept index, text, content word count, single rare word?)]
        self.conn = sqlite3.connect(":memory:")
        self.conn.execute("CREATE VIRTUAL TABLE cfts USING fts5(phrase, tokenize='porter unicode61 remove_diacritics 2')")
        for c in data.get("concepts", []):
            targets = [(r["n"], r["id"]) for t in c["targets"] for r in [ix.get(t)] if r is not None]
            if not targets:
                continue
            ci = len(self.concepts)
            self.concepts.append({"targets": targets, "note": c.get("note", "")})
            for ph in c["phrases"]:
                words = [w.lower() for t in TOKEN_RE.findall(ph) for w in re.split(r"[._()]", t)
                         if len(w) >= 2 and w.lower() not in STOPWORDS]
                rare = len(words) == 1 and ix.conn.execute(
                    "SELECT count(*) FROM (SELECT 1 FROM fts WHERE fts MATCH ? LIMIT ?)",
                    [fts_phrase(words[0]), CONCEPT_RARE_DF]).fetchone()[0] < CONCEPT_RARE_DF
                self.conn.execute("INSERT INTO cfts(rowid, phrase) VALUES (?, ?)", (len(self.phrases), ph))
                self.phrases.append((ci, ph, len(words), rare))

    @classmethod
    def load(cls, ix: "Index", path: Path) -> "Concepts | None":
        if not path.is_file():
            return None
        raw = path.read_bytes()
        return cls(ix, json.loads(raw.decode("utf-8")), hashlib.sha1(raw).hexdigest()[:12])

    def passages(self) -> tuple[list[int], list[str]]:
        ns, texts = [], []
        for ci, ph, _, _ in self.phrases:
            for n, _ in self.concepts[ci]["targets"]:
                ns.append(n)
                texts.append(ph)
        return ns, texts

    def match(self, groups: list) -> tuple[dict, dict, list]:
        """Query concept groups -> (entry n -> set of group indexes its phrases cover,
        entry n -> full-phrase bonus (decays with the target's position), notes of fully matched concepts)."""
        hits: dict = {}  # phrase index -> groups matched
        for gi, g in enumerate(groups):
            alts = " OR ".join(fts_phrase(t) for t in g)
            for (pi,) in self.conn.execute("SELECT rowid FROM cfts WHERE cfts MATCH ?", [alts]):
                hits.setdefault(pi, set()).add(gi)
        covered: dict = {}
        full: dict = {}
        notes = []
        for pi, gs in sorted(hits.items(), key=lambda kv: -len(kv[1])):
            ci, _, nwords, rare = self.phrases[pi]
            c = self.concepts[ci]
            is_full = len(gs) >= nwords and (nwords >= 2 or rare)
            if not (is_full or len(gs) >= 2):  # one common word shared with a longer phrase says little
                continue
            for k, (n, _) in enumerate(c["targets"]):
                covered.setdefault(n, set()).update(gs)
                if is_full:
                    full[n] = max(full.get(n, 0.0), 1.0 - CONCEPT_DECAY * min(k, 5))
            if is_full and c["note"] and c["note"] not in notes:
                notes.append(c["note"])
        return covered, full, notes[:2]


@dataclass
class QueryInfo:
    raw: str
    groups: list  # list of term alternatives; each group is one concept of the query
    exact: list
    words: set
    notes: list
    alias_targets: list
    identifiers: bool = False
    natural: bool = False
    ngrams: list = field(default_factory=list)
    class_hints: list = field(default_factory=list)  # classes named anywhere in the query (any case)
    extra_terms: list = field(default_factory=list)  # weak expansions: searched, but not counted as concepts
    type_hints: list = field(default_factory=list)  # node types implied by everyday words (members get a small boost)

    @property
    def terms(self) -> list:
        return list(dict.fromkeys(t for g in self.groups for t in g))


def analyze_query(ix: Index, query: str) -> QueryInfo:
    toks = [(m.group(0)[:-2] if m.group(0).endswith("()") else m.group(0), m.group(0).endswith("()"))
            for m in TOKEN_RE.finditer(query)]
    words = {w.lower() for t, _ in toks for w in re.split(r"[._]", t) if w}
    words |= {t.lower() for t, _ in toks}
    notes, terms, exact, targets = [], [], [], []
    qn = query.strip().rstrip("?").strip()
    qn = qn[:-2] if qn.endswith("()") else qn
    if qn and " " not in qn:
        exact.append(qn.lower())

    replaced: set = set()
    for a in ix.aliases():
        hit = False
        if a["trigger"]:
            hit = re.search(a["trigger"], query) is not None
        else:
            old = a["old"]
            for t, parens in toks:
                last = t.split(".")[-1]
                if a["kind"] == "class":
                    first = t.split(".")[0]
                    if a["strict"]:
                        hit = (len(toks) == 1 and t == old) or ("." in t and first == old)
                    else:
                        hit = first.lower() == old.lower()
                else:
                    full_match = t.lower() == old.lower()
                    last_match = last.lower() == old.split(".")[-1].lower() and "." not in old
                    if a["strict"]:
                        hit = (full_match or last_match) and (parens or "." in t)
                    else:
                        hit = full_match or last_match
                if hit:
                    if not ix.name_exists(t.split(".")[-1] if a["kind"] == "member" else t.split(".")[0]):
                        replaced.add(t)
                    break
        if hit:
            notes.append(f"Godot 3 -> 4: {a['note']}" + (f" See: read {a['target']}" if a["target"] else ""))
            new_terms = a["terms"] or a["new"]
            for nt in re.findall(r"@?[A-Za-z_][\w.]*", new_terms):
                terms += [p for p in nt.split(".") if p]
            if a["target"]:
                targets.append(a["target"])
                exact.append(a["target"].lower())

    # "Vector2 normalized" -> Vector2.normalized ; "get mouse position" -> get_mouse_position
    plain = [t for t, _ in toks]
    class_hints = []
    for k, a in enumerate(plain):
        crow = ix.class_row(a) if "." not in a else None
        if crow is None:
            continue
        class_hints.append(crow["name"].lower())
        for size in (1, 2, 3):
            if k + size < len(plain):
                exact.append(f"{a}.{'_'.join(plain[k + 1:k + 1 + size])}".lower())
    words_only = [t.lower() for t in plain if re.fullmatch(r"[A-Za-z0-9]+", t)]
    ngrams = []
    for size in (2, 3, 4):
        for k in range(len(words_only) - size + 1):
            ngrams += ["_".join(words_only[k:k + size]), "".join(words_only[k:k + size])]
    identifiers = False
    for t, parens in toks:
        if t in replaced:
            continue
        identish = "_" in t or "." in t or "@" in t or re.search(r"[a-z][A-Z]|[A-Za-z]\d", t) is not None
        if not identish and t[:1].isupper():
            crow = ix.class_row(t)
            identish = crow is not None and crow["name"] == t
        if identish or parens:
            exact.append(t.lower())
            identifiers = True
        for p in t.split("."):
            pl = p.lower().lstrip("@")
            if len(pl) < 2 or (pl in STOPWORDS and "_" not in pl):
                continue
            terms.append(pl)
    groups, seen = [], set()
    for t in terms:
        t = t.lower().lstrip("@")
        if t and t not in seen:
            seen.add(t)
            groups.append([t] + [s for s in SYNONYMS.get(t, []) if s not in seen])
    lowq = " ".join(re.findall(r"[a-z0-9]+", query.lower()))
    for phrase, alts in PHRASE_SYNONYMS.items():
        if f" {phrase} " in f" {lowq} ":
            for g in groups:
                if g[0] == phrase.split()[-1]:
                    g += [a for a in alts if a not in g]
    natural = not identifiers and len(plain) >= 2
    # Everyday words ("enemy", "sprite", "pickup") -> the node types they usually mean (TYPE_HINTS).
    extra = []
    hinted = type_hints(words) if TYPE_HINT_MODE != "off" and not identifiers else []
    hinted = [h for h in hinted if h.lower() not in class_hints]
    if TYPE_HINT_MODE == "syn":
        for h in hinted:
            word = next(w for w in words if h in TYPE_HINTS.get(w, []) + OBJECT_PHYSICS)
            for g in groups:
                if g[0] == word or word in g:
                    g.append(h.lower())
                    break
    elif TYPE_HINT_MODE == "hint":  # search the class and boost it like a class named in the query
        extra += [h.lower() for h in hinted]
        class_hints += [h.lower() for h in hinted]
    elif TYPE_HINT_MODE == "members":  # search the class, boost only its members
        extra += [h.lower() for h in hinted]
    q = QueryInfo(query, groups, list(dict.fromkeys(exact)), words, list(dict.fromkeys(notes)), targets,
                  identifiers, natural, ngrams, class_hints, extra)
    q.type_hints = [h.lower() for h in hinted] if TYPE_HINT_MODE in ("members", "boost") else []
    return q


def fts_phrase(term: str) -> str:
    return '"' + term.replace('"', "") + '"'


def entry_prior(ix: Index, row, q: QueryInfo, lang: str) -> float:
    p = KIND_PRIOR.get(row["kind"], 1.0)
    flags = ix.flags_of(row)
    if "deprecated" in flags:
        p *= 0.5
    if "outdated" in flags:
        p *= 0.85
    if row["lang"] and lang != "all" and row["lang"] != lang:
        lifted = row["lang"] == "cpp" and q.words & {"gdextension", "cpp", "c++", "godot_cpp"}
        lifted = lifted or (row["lang"] == "csharp" and q.words & {"csharp", "c#", "dotnet", "mono"})
        if not lifted:
            p *= LANG_PENALTY.get(row["lang"], 1.0)
    cls = row["cls"] or ""
    for prefix, mult, lift in FAMILY_PENALTIES:
        if cls.startswith(prefix) and not (q.words & lift):
            p *= mult
            break
    if row["kind"] in ("page", "section"):
        for prefix, mult, lift in PATH_PENALTIES:
            if row["id"].startswith(prefix) and not (q.words & lift):
                p *= mult
        if row["kind"] == "section" and len(row["body"] or "") < 120:
            p *= 0.7
    return p


def load_embed():
    try:
        import godot_embed
    except ImportError as exc:
        raise DocsError(f"Semantic search needs numpy, onnxruntime and tokenizers ({exc}). "
                        "Install them or use --mode lexical.") from exc
    return godot_embed


def semantic_scores(ix: Index, query: str, model=None):
    """Per-entry cosine similarities from the optional embedding layer (imported only here).
    Returns (scores, None), or (None, note) when vectors aren't built yet - search then falls back to lexical."""
    ge = load_embed()
    try:
        return ge.store_for(ix, model).entry_scores(query), None
    except ge.VectorsMissing as exc:
        return None, (f"Embedding vectors not built ({exc}); showing lexical results. "
                      f"Build once with: index --vectors (takes a few minutes on CPU).")
    except RuntimeError as exc:
        raise DocsError(str(exc)) from exc


def semantic_candidates(ix: Index, sims, filt: str, params: list, top: int) -> dict:
    """Top entries by similarity (respecting the search filters), scaled to 0..1 against the top-`top` floor."""
    import numpy as np
    if filt:
        allowed = np.fromiter((r[0] for r in ix.conn.execute(f"SELECT e.n FROM entries e WHERE 1{filt}", params)),
                              dtype=np.int64)
        allowed = allowed[allowed < len(sims)]
    else:
        allowed = np.arange(1, len(sims))
    if not len(allowed):
        return {}, 0.0, 1.0
    sub = sims[allowed]
    k = min(top, len(sub))
    best = allowed[np.argpartition(-sub, k - 1)[:k]]
    hi, lo = float(sims[best].max()), float(sims[best].min())
    span = (hi - lo) or 1.0
    return {int(n): max(0.0, (float(sims[n]) - lo) / span) for n in best}, lo, span


SEMANTIC_TOP = 100
HYBRID_WEIGHT = 0.8  # tuned on core + dev split for models/gdocs-bge-small (train/results/final_bench.log)


def run_search(ix: Index, query: str, kinds=None, cls=None, path=None, limit=8, lang="gdscript", diversify=True,
               mode="lexical", model=None) -> dict:
    q = analyze_query(ix, query)
    out = {"query": query, "godot_version": ix.version, "lang": lang, "mode": mode, "notes": q.notes,
           "results": [], "more": []}
    if mode == "lexical" and not q.terms:
        return out
    cols = "name keywords title signature summary body code" + (" code_cs" if lang != "gdscript" else "")
    weights = "10.0, 6.0, 5.0, 3.0, 3.0, 1.0, 1.2, " + ("0.0" if lang == "gdscript" else "1.2")
    match = "{%s} : (%s)" % (cols, " OR ".join(fts_phrase(t) for t in q.terms + q.extra_terms)) if q.terms else ""
    where, params = [], []
    if kinds:
        where.append(f"e.kind IN ({','.join('?' * len(kinds))})")
        params += kinds
    if cls:
        where.append("e.cls = ? COLLATE NOCASE")
        params.append(cls)
    if path:
        p = path.strip().removeprefix("doc:").strip("/")
        where.append("e.id LIKE ?")
        params.append(f"doc:{p}%")
    filt = "".join(f" AND {w}" for w in where)
    bm: dict = {}
    if match and mode != "semantic":
        bm = {r[0]: -r[1] for r in ix.conn.execute(
            f"SELECT e.n, bm25(fts, {weights}) AS r FROM fts JOIN entries e ON e.n = fts.rowid "
            f"WHERE fts MATCH ?{filt} ORDER BY r LIMIT 400", [match, *params])}
        # Always consider exact name/id matches, even if BM25 ranked them low.
        for s in q.exact:
            for r in ix.conn.execute(
                    f"SELECT e.n FROM entries e WHERE (e.name = ? COLLATE NOCASE OR e.id = ? COLLATE NOCASE){filt} LIMIT 50",
                    [s, s, *params]):
                bm.setdefault(r[0], 0.0)
    c_cov, c_full = {}, {}
    cx = ix.concepts if q.groups else None
    if cx:
        c_cov, c_full, c_notes = cx.match(q.groups)
        out["notes"] += [n for n in c_notes if n not in out["notes"]]
        if c_cov and mode != "semantic":
            allowed = [r[0] for r in ix.conn.execute(
                f"SELECT e.n FROM entries e WHERE e.n IN ({','.join('?' * len(c_cov))}){filt}", [*c_cov, *params])]
            for n in allowed:
                bm.setdefault(n, 0.0)
    sem: dict = {}
    if mode != "lexical":
        sims, note = semantic_scores(ix, query, model)
        if sims is None:
            out["notes"].append(note)
            out["mode"] = mode = "lexical"
            return run_search(ix, query, kinds, cls, path, limit, lang, diversify) | {"notes": out["notes"], "mode": mode}
        sem, sem_lo, sem_span = semantic_candidates(ix, sims, filt, params, SEMANTIC_TOP)
        if mode == "semantic":
            exact = set(q.exact)
            return finish_search(ix, out, [
                (s * entry_prior(ix, row, q, lang), n, row,
                 {"exact": float(row["id"].lower() in exact or (row["name"] or "").lower() in exact), "cos": float(sims[n]),
                  "ident": float(q.identifiers)})
                for n, row in ix.by_rowids(sem).items() for s in [sem[n]]], match, limit, diversify)
        for n in sem:
            bm.setdefault(n, 0.0)
    if not bm:
        return out
    missing = [n for n, v in bm.items() if v == 0.0]
    if missing and match:
        for r in ix.conn.execute(
                f"SELECT rowid, bm25(fts, {weights}) FROM fts WHERE fts MATCH ? AND rowid IN ({','.join('?' * len(missing))})",
                [match, *missing]):
            bm[r[0]] = -r[1]
    max_bm = max(bm.values()) or 1.0

    # Per-concept coverage, weighted by rarity (IDF): entries matching more of the query's concepts win.
    idf, any_rows, title_rows = [], [], []
    for g in q.groups:
        alts = " OR ".join(fts_phrase(t) for t in g)
        any_rows.append({r[0] for r in ix.conn.execute("SELECT rowid FROM fts WHERE fts MATCH ?", ["{%s} : (%s)" % (cols, alts)])})
        title_rows.append({r[0] for r in ix.conn.execute("SELECT rowid FROM fts WHERE fts MATCH ?", ["{name keywords title} : (%s)" % alts])})
        df = len(any_rows[-1])
        idf.append(math.log((ix.total - df + 0.5) / (df + 0.5) + 1.0))
    for n, gs in c_cov.items():  # concept phrases count like keywords of their targets
        for gi in gs:
            any_rows[gi].add(n)
            title_rows[gi].add(n)
    idf_sum = sum(idf) or 1.0
    # Conceptual questions ("how to ...") are best answered by the manual; identifiers by the API reference.
    docs_boost = 1.25 if q.natural else 1.0
    ngrams = set(q.ngrams)
    hints = set(q.class_hints)
    type_hinted = set(q.type_hints)

    rows = ix.by_rowids(bm)
    exact = set(q.exact)
    targets = {t.lower() for t in q.alias_targets}
    model = RANK_MODELS.get("hybrid" if sem else "lexical") if RANK_MODE == "learned" else None
    scored = []
    for n, row in rows.items():
        cov = sum(w for w, rs in zip(idf, any_rows) if n in rs) / idf_sum
        tcov = sum(w for w, rs in zip(idf, title_rows) if n in rs) / idf_sum
        rid, name = row["id"].lower(), (row["name"] or "").lower()
        is_doc = row["kind"] in ("page", "section")
        member_of = (row["cls"] or "").lower() if row["kind"] not in ("class", "page", "section") else ""
        # Ranking signals (also the inputs of the learned ranker; see train/fit_ranker.py)
        f = {"bm": max(bm[n], 0.0) / max_bm, "cov": cov, "cov2": cov * cov, "tcov": tcov, "doc": float(is_doc),
             "doc_nat": float(is_doc and q.natural), "ngram": float(name in ngrams and not is_doc),
             "chint": float(row["kind"] == "class" and rid in hints), "mhint": float(member_of in hints),
             "thint": float(member_of in type_hinted), "ident": float(q.identifiers),
             "lprior": math.log(max(entry_prior(ix, row, q, lang), 1e-6)), "lpop": math.log1p(row["pop"] or 0)}
        if sem:
            f["sem"] = max(0.0, (float(sims[n]) - sem_lo) / sem_span)
            f["cos"] = float(sims[n])
        # Hard rules: exact names and Godot 3 alias targets (identifier queries) always win.
        rule = 0.0
        if rid in targets:
            rule += 1.5
        if rid in exact:
            rule += 2.0
        elif name in exact or name.lstrip("@") in exact:
            rule += 1.0 if not is_doc else 0.4
            if row["cls"] in ("@GlobalScope", "@GDScript"):
                rule += 0.15
            if row["kind"] == "class":
                rule += 0.1
        prior = math.exp(f["lprior"]) * (1.0 + 0.03 * f["lpop"])
        # hand-tuned formula: soft part (before rules and the prior)
        soft = 0.40 * f["bm"] + 0.40 * f["cov2"] + 0.20 * tcov
        if is_doc:
            soft *= docs_boost
        if not rule:
            soft += 0.5 * f["ngram"] + 0.3 * f["chint"]
        soft += 0.1 * f["mhint"] or TYPE_HINT_BOOST * f["thint"]
        f["concept"] = c_full.get(n, 0.0)
        soft += CONCEPT_WEIGHT * f["concept"]
        if sem:
            soft += HYBRID_WEIGHT * f["sem"]
        f["lhand"] = math.log(soft + 1e-4)
        if model:  # learned corrections around the hand score (identical to it when lhand=1, others 0)
            z = model["bias"] + sum(w * f.get(k, 0.0) for k, w in model["w"].items())
            score = (math.exp(min(z, 20.0)) + rule) * prior
        else:
            score = (soft + rule) * prior
        if _TRACE is not None:
            _TRACE.append({"id": row["id"], "f": f, "rule": rule, "prior": prior, "score": score})
        feat = {"cov": cov, "tcov": tcov, "exact": float(rid in targets or rid in exact or (
            (name in exact or name.lstrip("@") in exact) and not is_doc)), "ident": float(q.identifiers)}
        if sem:
            feat["cos"] = float(sims[n])
        scored.append((score, n, row, feat))
    return finish_search(ix, out, scored, match, limit, diversify)


# Ranking: "hand" (the tuned formula above) or "learned" (pairwise logistic weights over the same signals,
# fitted on judged benchmark queries by train/fit_ranker.py). Exact-name/alias rules apply in both.
RANK_MODE = os.environ.get("GODOT_DOCS_RANKER", "hand")
RANK_MODELS: dict = {}
_TRACE = None  # set to a list to record every candidate's signals (used by train/fit_ranker.py)


# Logistic models giving P(result answers the query) from ranking signals, one per signal set. Fitted on the
# pooled-label benchmark (dev + val splits) by train/fit_confidence.py; tiers checked on the held-out test split.
CONF_MODELS: dict = {"lexical": {"bias": -2.3164, "w": {"cov": 1.5405, "tcov": 2.199, "rel": 0.2452, "rank": -0.1549, "doc": -0.113}}, "semantic": {"bias": -7.7254, "w": {"cos": 9.1001, "rel": 0.9923, "rank": -0.1323, "doc": 0.3443}}, "hybrid": {"bias": -7.5789, "w": {"cov": 1.0222, "tcov": 0.3311, "cos": 8.7415, "rel": 0.5975, "rank": -0.1063, "doc": 0.2329}}}
TIER_STRONG, TIER_PARTIAL = 0.5, 0.2


def confidence(sig: dict) -> tuple:
    kind = "semantic" if "cov" not in sig else ("hybrid" if "cos" in sig else "lexical")
    model = CONF_MODELS.get(kind)
    if sig.get("exact"):
        p = 0.9  # exact name or Godot 3 alias target: almost always what was asked for
    elif not model:
        return None, None
    else:
        z = model["bias"] + sum(w * sig.get(k, 0.0) for k, w in model["w"].items())
        p = 1.0 / (1.0 + math.exp(-z))
        if sig.get("ident"):  # identifier queries: only exact name matches count as strong (models fit on prose)
            p = min(p, TIER_STRONG - 0.001)
    return round(p, 3), ("strong" if p >= TIER_STRONG else "partial" if p >= TIER_PARTIAL else "weak")


def finish_search(ix: Index, out: dict, scored: list, match: str, limit: int, diversify: bool) -> dict:
    """Sort, cap results per class / page so one family doesn't flood the list, and render result dicts."""
    scored.sort(key=lambda x: -x[0])
    caps: dict = {}
    more: dict = {}
    picked = []
    for score, n, row, feat in scored:
        key = (row["parent"] or row["id"]) if row["kind"] in ("page", "section") else row["cls"]
        if diversify and caps.get(key, 0) >= 3:
            if key in caps and len(picked) < limit:
                more[key] = more.get(key, 0) + 1
            continue
        if len(picked) >= limit:
            break
        caps[key] = caps.get(key, 0) + 1
        picked.append((score, n, row, feat))
    top = picked[0][0] if picked and picked[0][0] > 0 else 1.0
    for rank, (score, n, row, feat) in enumerate(picked):
        snippet = ""
        if row["kind"] in ("page", "section") and match:
            r = ix.conn.execute("SELECT snippet(fts, 5, '', '', ' ... ', 24) FROM fts WHERE fts MATCH ? AND rowid = ?", [match, n]).fetchone()
            snippet = re.sub(r"\s+", " ", r[0]).strip() if r and r[0] else ""
        out["results"].append({
            "id": row["id"], "kind": row["kind"], "title": row["title"], "signature": row["signature"] or "",
            "summary": row["summary"] or "", "snippet": snippet or (row["summary"] or ""),
            "flags": ix.flags_of(row), "source": row["source"],
            "score": round(score, 4), "_group": (row["parent"] or row["id"]) if row["kind"] in ("page", "section") else row["cls"],
            "_signals": dict(feat, rel=score / top, rank=rank, doc=float(row["kind"] in ("page", "section"))),
        })
    for r in out["results"]:
        r["confidence"], r["tier"] = confidence(r["_signals"])
    out["more"] = [{"group": k, "count": v} for k, v in more.items()]
    return out


# Multi-phrasing search (`search "<question>" --also "<Godot terms>"`): each phrasing is searched separately and the
# ranked lists are fused. Tuned on the dev/val splits with eval/also_phrasings.json (eval_queries.py --also).
MULTI_POOL = 30      # candidates taken from each phrasing
MULTI_FUSION = "conf_or"  # dev/val sweep: train/results/sweep_also.log
RRF_K = 10


def fuse_score(ranks_rels: list, fusion: str) -> float:
    """ranks_rels: (rank, rel, confidence) of one result in each phrasing's list where it appears."""
    if fusion == "rrf":
        return sum(1.0 / (RRF_K + rank + 1) for rank, _, _ in ranks_rels)
    if fusion == "rel_sum":
        return sum(rel for _, rel, _ in ranks_rels)
    if fusion == "rel_max":
        return max(rel for _, rel, _ in ranks_rels) + 1e-3 * len(ranks_rels)
    if fusion == "conf_or":  # P(at least one phrasing's verdict is right), confidences are calibrated per query
        p = 1.0
        for _, _, c in ranks_rels:
            p *= 1.0 - (c or 0.0)
        return 1.0 - p + 1e-3 * fuse_score(ranks_rels, "rrf")  # ties (e.g. all-weak tails): reciprocal rank
    raise ValueError(fusion)


def multi_search(ix: Index, queries: list, kinds=None, cls=None, path=None, limit=8, lang="gdscript", diversify=True,
                 mode="lexical", model=None, fusion=None) -> dict:
    """run_search over several phrasings of one question, fused into one list. One phrasing = plain run_search."""
    queries = [q for q in dict.fromkeys(q.strip() for q in queries) if q]
    if len(queries) <= 1:
        return run_search(ix, queries[0] if queries else "", kinds, cls, path, limit, lang, diversify, mode, model)
    runs = [run_search(ix, q, kinds, cls, path, max(MULTI_POOL, limit), lang, False, mode, model) for q in queries]
    out = dict(runs[0], also=queries[1:], results=[], more=[],
               notes=list(dict.fromkeys(n for r in runs for n in r["notes"])))
    if any(r["mode"] != mode for r in runs):
        out["mode"] = "lexical"
    hits: dict = {}
    for res in runs:
        for rank, r in enumerate(res["results"]):
            h = hits.setdefault(r["id"], {"best": r, "rr": []})
            h["rr"].append((rank, r["_signals"]["rel"], r["confidence"]))
            if (r["confidence"] or 0) > (h["best"]["confidence"] or 0):
                h["best"] = r
    fused = sorted(((fuse_score(h["rr"], fusion or MULTI_FUSION), h["best"]) for h in hits.values()),
                   key=lambda x: -x[0])
    caps: dict = {}
    more: dict = {}
    for score, r in fused:
        key = r["_group"]
        if diversify and caps.get(key, 0) >= 3:
            if len(out["results"]) < limit:
                more[key] = more.get(key, 0) + 1
            continue
        if len(out["results"]) >= limit:
            break
        caps[key] = caps.get(key, 0) + 1
        out["results"].append(dict(r, score=round(score, 4)))
    out["more"] = [{"group": k, "count": v} for k, v in more.items()]
    return out


def search_label(res: dict) -> str:
    return f'"{res["query"]}"' + "".join(f' --also "{a}"' for a in res.get("also", []))


MIN_FULL_RESULTS = 3


def _result_detail(r: dict) -> str:
    """One line under the result header: signature + summary for API entries, breadcrumb + snippet for docs."""
    if r["kind"] in ("page", "section"):
        snip = re.sub(r"^(\.\.\.\s*)+|(\s*\.\.\.)+$", "", r["snippet"]).strip()
        snip = f"...{snip}" if r["snippet"].lstrip().startswith("...") else snip
        return f"{r['title']} - {shorten(snip, 200)}"
    summary = shorten(r["summary"], 170) if r["summary"] else ""
    if r["kind"] == "class":
        parents = r["signature"].split(" < ")[1:2]
        head = f"extends {parents[0]}" if parents else ""
    else:
        head = r["signature"].split("\n")[0]
    return " - ".join(x for x in (head, summary) if x)


def render_search(res: dict) -> str:
    """Compact text for AI readers: tiered results, weak tail collapsed to ids, concrete next commands."""
    extra = "".join(f" | {x}" for x in (res.get("mode", "lexical"), f"lang {res['lang']}" if res["lang"] != "gdscript" else "") if x)
    lines = [f'search {search_label(res)} | Godot {res["godot_version"]}{extra}']
    for note in res["notes"]:
        lines.append(f"NOTE {note}")
    results = res["results"]
    if not results:
        lines.append("No results. Try fewer or different words, `lookup <name>` for symbols, or `toc` to browse.")
        return "\n".join(lines)
    # full entries up to the last non-weak result (at least MIN_FULL_RESULTS), the weak tail as ids only
    last_good = max((i for i, r in enumerate(results) if r.get("tier") in ("strong", "partial")), default=-1)
    n_full = min(len(results), max(MIN_FULL_RESULTS, last_good + 1))
    lines.append("")
    for i, r in enumerate(results[:n_full], 1):
        tag = f"[{r['tier']}] " if r.get("tier") else ""
        flags = "".join(f" [{f}]" for f in r["flags"] if f in ("deprecated", "experimental", "outdated"))
        lines.append(f"{i}. {tag}{r['id']} ({r['kind']}){flags}")
        lines.append(f"   {_result_detail(r)}")
    if n_full < len(results):
        lines.append(f"+{len(results) - n_full} weaker: " + ", ".join(r["id"] for r in results[n_full:]))
    for m in res["more"]:
        hint = f"--class {m['group']}" if not m["group"].startswith("doc:") else f"read {m['group']}"
        lines.append(f"(+{m['count']} more in {m['group']}: {hint})")
    lines.append("")
    tiers = [r.get("tier") for r in results]
    if tiers[0] is not None and "strong" not in tiers:
        hint = ("No strong match. Try another --also guess (class, method or node names)" if res.get("also") else
                'No strong match. Add --also "<Godot class/method names you expect>"') + ", narrow with --kind api or --kind docs"
        lines.append(hint + (", or try --mode hybrid." if res.get("mode", "lexical") == "lexical" else "."))
    best = next((r for r in results if r.get("tier") == "strong"), None) or         next((r for r in results if r.get("tier") == "partial"), results[0])
    lines.append(f"Read: read {best['id']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resolving IDs
# ---------------------------------------------------------------------------


def resolve_id(ix: Index, ident: str) -> tuple:
    """Returns (row, note, candidates). row is None when ambiguous / not found."""
    s = ident.strip()
    s = s[:-2] if s.endswith("()") else s
    row = ix.get(s) or ix.get_nocase(s)
    if row:
        return row, "", []
    # Manual pages: labels, paths, basenames.
    path = s.removeprefix("doc:").removesuffix(".rst").replace("\\", "/")
    frag = ""
    if "#" in path:
        path, frag = path.split("#", 1)
    for cand in (f"doc:{path}", f"doc:{path}/index"):
        r = ix.get(cand + (f"#{frag}" if frag else ""))
        if r:
            return r, "", []
    hit = ix.conn.execute("SELECT id FROM anchors WHERE label = ?", (s.lower(),)).fetchone()
    if hit and ix.get(hit[0]):
        return ix.get(hit[0]), "", []
    base = path.split("/")[-1]
    pages = ix.conn.execute("SELECT id FROM entries WHERE kind = 'page' AND (id LIKE ? OR id = ?)", (f"%/{base}", f"doc:{base}")).fetchall()
    if len(pages) == 1:
        r = ix.get(pages[0][0] + (f"#{frag}" if frag else "")) or ix.get(pages[0][0])
        return r, "", []
    # Inherited member: Class.member defined on an ancestor.
    if "." in s and not s.startswith("doc:"):
        cls, mem = s.split(".", 1)
        crow = ix.class_row(cls)
        if crow:
            for anc in json.loads(crow["inherits"]):
                r = ix.get(f"{anc}.{mem}") or ix.get_nocase(f"{anc}.{mem}")
                if r:
                    return r, f"`{mem}` is inherited by {crow['name']} from {anc}.", []
    # Godot 3 names.
    for a in matching_aliases(ix, s):
        r = ix.get(a["target"]) if a["target"] else None
        if r:
            return r, f"Godot 3 -> 4: {a['note']}", []
    # Bare member name (annotations may be given without their "@").
    name = s.split(".")[-1]
    cands = ix.conn.execute("SELECT * FROM entries WHERE name IN (?, ?) COLLATE NOCASE AND kind NOT IN ('page','section') ORDER BY n",
                            (name, "@" + name)).fetchall()
    if len(cands) == 1:
        return cands[0], "", []
    return None, "", cands


def matching_aliases(ix: Index, symbol: str) -> list:
    """Godot 3 aliases for a symbol typed on its own (`read`/`lookup`)."""
    s = symbol.lower()
    out = []
    for a in ix.aliases():
        if a["trigger"]:
            if re.search(a["trigger"], symbol) or s in [o.strip().lower() for o in a["old"].split("/")]:
                out.append(a)
        elif a["old"].lower() in (s, s.split(".")[-1]):
            out.append(a)
    return out


def suggestions(ix: Index, ident: str, n: int = 8) -> list[str]:
    s = ident.strip().removesuffix("()")
    key = s.split(".")[-1].lower()
    names = [r[0] for r in ix.conn.execute("SELECT DISTINCT name FROM entries WHERE kind NOT IN ('page','section')")]
    lower = {}
    for nm in names:
        lower.setdefault(nm.lower(), nm)
    close = difflib.get_close_matches(key, list(lower), n=n, cutoff=0.72)
    out = []
    for c in close:
        for r in ix.conn.execute("SELECT id FROM entries WHERE name = ? COLLATE NOCASE AND kind NOT IN ('page','section') LIMIT 3", (lower[c],)):
            out.append(r[0])
    return out[:n]


# ---------------------------------------------------------------------------
# Rendering entries
# ---------------------------------------------------------------------------


def members_of(ix: Index, cls: str):
    return ix.conn.execute("SELECT * FROM entries WHERE cls = ? AND kind != 'class' ORDER BY n", (cls,)).fetchall()


def member_listing(ix: Index, cls: str, with_summary: bool, kinds=None, name_filter: str = "") -> list[str]:
    rows = members_of(ix, cls)
    if name_filter:
        rows = [r for r in rows if name_filter.lower() in r["name"].lower()]
    out = []
    for kind in KIND_ORDER:
        if kinds and kind not in kinds:
            continue
        group = [r for r in rows if r["kind"] == kind]
        if kind == "constant":
            group = [r for r in group if not json.loads(r["extra"] or "{}").get("enum")]
        if not group:
            continue
        out.append(f"### {KIND_PLURAL[kind]} ({len(group)})")
        for r in group:
            if kind == "enum":
                vals = json.loads(r["extra"] or "{}").get("values", [])
                names = ", ".join(v[0] for v in vals)
                out.append(f"- {r['signature']} {{ {names} }}")
                continue
            sig = r["signature"].split("\n")
            line = f"- {sig[0]}" + (f"  (+{len(sig) - 1} overloads)" if len(sig) > 1 else "")
            if "deprecated" in ix.flags_of(r):
                line += "  [deprecated]"
            if with_summary and r["summary"]:
                line += f" - {shorten(r['summary'], 110)}"
            out.append(line)
        out.append("")
    return out


def render_class(ix: Index, row, full: bool, lang: str) -> str:
    extra = json.loads(row["extra"] or "{}")
    lines = [f"# {row['id']}  (class, Godot {ix.version})"]
    if extra.get("inherits"):
        lines.append("Inherits: " + " < ".join(extra["inherits"]))
    kids = extra.get("inherited_by", [])
    if kids:
        shown = ", ".join(kids[:15])
        lines.append(f"Inherited by: {shown}" + (f" (+{len(kids) - 15} more; see `hierarchy {row['id']}`)" if len(kids) > 15 else ""))
    lines.append(f"Source: {row['source']}")
    lines += ["", filter_code(row["body"], lang), ""]
    if full:
        lines.append("## Members")
        for kind in KIND_ORDER:
            group = [r for r in members_of(ix, row["id"]) if r["kind"] == kind]
            if not group:
                continue
            lines += ["", f"### {KIND_PLURAL[kind]}", ""]
            for r in group:
                lines.append(f"#### {r['name']}  [{r['id']}]")
                lines.append("    " + "\n    ".join(r["signature"].split("\n")))
                body = filter_code(r["body"], lang).strip()
                if body:
                    lines += ["", body]
                lines.append("")
    else:
        listing = member_listing(ix, row["id"], with_summary=False)
        if listing:
            lines.append(f"## Members (details: `read {row['id']}.<name>`; inherited: `members {row['id']} --inherited`)")
            lines.append("")
            lines += listing
    return tidy(lines)


def render_member(ix: Index, row, lang: str) -> str:
    extra = json.loads(row["extra"] or "{}")
    anc = ix.ancestors(row["cls"])
    lines = [f"# {row['id']}  ({row['kind']}, Godot {ix.version})"]
    lines += [f"    {s}" for s in row["signature"].split("\n")]
    lines.append(f"Class: {row['cls']}" + (f" (inherits {' < '.join(anc)})" if anc else ""))
    if ix.deprecation_note(row):
        lines.append(f"DEPRECATED: {ix.deprecation_note(row)}")
    flags = ix.flags_of(row)
    if flags:
        lines.append("Flags: " + ", ".join(flags))
    if extra.get("enum"):
        lines.append(f"Enum: {row['cls']}.{extra['enum']}")
    if extra.get("setget"):
        lines.append("Accessors: " + "; ".join(extra["setget"]))
    body = filter_code(row["body"] or "", lang).strip()
    lines += ["", body or "(No description in the docs.)"]
    if extra.get("values"):
        lines += ["", "Values:"]
        lines += [f"- {sig}" + (f" - {summ}" if summ else "") for sig, summ in extra["values"]]
    lines += ["", f"Source: {row['source']}"]
    return tidy(lines)


def render_doc(ix: Index, row, lang: str) -> str:
    page_id = row["parent"] or row["id"]
    secs = ix.conn.execute("SELECT * FROM entries WHERE id = ? OR parent = ? ORDER BY n", (page_id, page_id)).fetchall()
    start = next(k for k, r in enumerate(secs) if r["id"] == row["id"])
    chunk = [secs[start]]
    for r in secs[start + 1:]:
        if row["kind"] == "section" and r["level"] <= row["level"]:
            break
        chunk.append(r)
    flags = [f for f in (row["flags"] or "").split(",") if f]
    lines = [f"# {row['title']}  ({row['kind']}, Godot {ix.version})", f"ID: {row['id']}", f"Source: {row['source']}"]
    if "outdated" in flags:
        lines.append("Warning: this page is marked as possibly outdated for the current Godot version.")
    if row["kind"] == "page" and len(chunk) > 4:
        lines += ["", "Sections (read one with `read <page id>#<slug>`):"]
        lines += ["  " * max(r["level"] - 2, 0) + f"- {r['name']}  #{r['id'].split('#', 1)[1]}" for r in chunk[1:]]
    for k, r in enumerate(chunk):
        body = filter_code(r["body"] or "", lang).strip()
        if k > 0 or row["kind"] == "section":
            slug = r["id"].split("#", 1)[1] if "#" in r["id"] else ""
            lines += ["", "#" * min(max(r["level"], 2), 6) + f" {r['name']}" + (f"  {{#{slug}}}" if slug else "")]
        if body:
            lines += ["", body]
    return tidy(lines)


def render_entry(ix: Index, row, full: bool, lang: str) -> str:
    if row["kind"] == "class":
        return render_class(ix, row, full, lang)
    if row["kind"] in ("page", "section"):
        return render_doc(ix, row, lang)
    return render_member(ix, row, lang)


def paginate(text: str, offset: int, max_chars: int) -> tuple[str, int | None]:
    if max_chars <= 0 or len(text) - offset <= max_chars:
        return text[offset:], None
    end = offset + max_chars
    cut = text.rfind("\n", offset, end)
    if cut > offset + max_chars // 2:
        end = cut
    return text[offset:end], end


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def parse_kinds(value: str | None) -> list[str] | None:
    if not value:
        return None
    kinds = []
    for k in value.split(","):
        k = k.strip().lower()
        if k in KIND_GROUPS:
            kinds += KIND_GROUPS[k]
        elif k in KIND_PRIOR:
            kinds.append(k)
        elif k.rstrip("s") in KIND_PRIOR:
            kinds.append(k.rstrip("s"))
        else:
            raise DocsError(f"Unknown kind '{k}'. Use: {', '.join(sorted(set(KIND_PRIOR) | set(KIND_GROUPS)))}")
    return list(dict.fromkeys(kinds))


def cmd_search(ix: Index, args) -> tuple:
    res = multi_search(ix, [args.query, *(getattr(args, "also", None) or [])], kinds=parse_kinds(args.kind), cls=args.cls, path=args.path,
                       limit=args.limit, lang=args.lang, diversify=not args.cls, mode=args.mode, model=args.model)
    log_query(args, res)
    text = render_search(res)
    for r in res["results"]:
        r.pop("_signals", None)
        r.pop("_group", None)
    return res, text, 0 if res["results"] else 1


def log_query(args, res: dict):
    """Opt-in: GODOT_DOCS_QUERY_LOG=<file> appends each search as a JSON line (future benchmark material)."""
    path = os.environ.get("GODOT_DOCS_QUERY_LOG")
    if not path:
        return
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "query": args.query, "also": getattr(args, "also", None) or [], "mode": res.get("mode"), "kind": args.kind,
           "class": args.cls, "path": args.path, "results": [r["id"] for r in res["results"]]}
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except OSError:
        pass  # logging must never break a search


def not_found(ix: Index, ident: str, cands) -> tuple:
    if cands:
        lines = [f"'{ident}' is ambiguous - {len(cands)} matches:"]
        lines += [f"- {r['id']}  ({r['kind']})  {r['signature'].split(chr(10))[0]}" for r in cands[:30]]
        if len(cands) > 30:
            lines.append(f"... +{len(cands) - 30} more (use `lookup {ident}`)")
        lines.append("Use `read <Class>.<name>` with one of the IDs above.")
        return {"error": "ambiguous", "candidates": [r["id"] for r in cands]}, "\n".join(lines), 1
    sugg = suggestions(ix, ident)
    lines = [f"No entry '{ident}'."]
    if sugg:
        lines.append("Did you mean: " + ", ".join(sugg))
    lines.append(f"Try: search \"{ident}\"")
    return {"error": "not_found", "suggestions": sugg}, "\n".join(lines), 1


def cmd_read(ix: Index, args) -> tuple:
    if args.max_chars is None:
        args.max_chars = 0 if getattr(args, "pretty", False) else DEFAULT_MAX_CHARS
    row, note, cands = resolve_id(ix, args.id)
    if row is None:
        return not_found(ix, args.id, cands)
    text = render_entry(ix, row, args.full, args.lang)
    if note:
        text = f"NOTE {note}\n\n{text}"
    page, next_off = paginate(text, args.offset, args.max_chars)
    if next_off is not None:
        page += f"\n\n[truncated: chars {args.offset}-{next_off} of {len(text)}. Continue with: read {row['id']} --offset {next_off}]"
    data = {"id": row["id"], "kind": row["kind"], "title": row["title"], "note": note, "text": page,
            "offset": args.offset, "next_offset": next_off, "total_chars": len(text)}
    return data, page, 0


def cmd_lookup(ix: Index, args) -> tuple:
    s = args.symbol.strip().removesuffix("()")
    out = {"symbol": s, "matches": [], "aliases": [], "suggestions": []}
    seen = set()

    def add(r, why):
        if r and r["id"] not in seen:
            seen.add(r["id"])
            out["matches"].append({"id": r["id"], "kind": r["kind"], "signature": (r["signature"] or "").split("\n")[0],
                                   "summary": r["summary"] or "", "why": why,
                                   "deprecated": "deprecated" in ix.flags_of(r)})

    add(ix.get(s) or ix.get_nocase(s), "exact id")
    row, note, _ = resolve_id(ix, s)
    if row is not None and note:
        add(row, note)
    name = s.split(".")[-1]
    for r in ix.conn.execute("SELECT * FROM entries WHERE name IN (?, ?) COLLATE NOCASE AND kind NOT IN ('page','section') "
                             "ORDER BY CASE WHEN cls IN ('@GlobalScope','@GDScript') THEN 0 ELSE 1 END, n", (name, "@" + name)):
        add(r, "same name")
    out["matches"].sort(key=lambda m: m["deprecated"] and m["why"] == "same name")  # deprecated namesakes last
    for a in matching_aliases(ix, s):
        out["aliases"].append({"old": a["old"], "new": a["new"], "target": a["target"], "note": a["note"]})
    if not out["matches"]:
        out["suggestions"] = suggestions(ix, s)
    lines = [f"lookup '{s}' (Godot {ix.version})"]
    for a in out["aliases"]:
        lines.append(f"NOTE Godot 3 -> 4: {a['note']}" + (f" See: read {a['target']}" if a["target"] else ""))
    for m in out["matches"][:60]:
        lines.append(f"- {m['id']}  ({m['kind']})  {m['signature']}" + ("  [deprecated]" if m["deprecated"] else "")
                     + (f"  <- {m['why']}" if m["why"] not in ("same name", "exact id") else ""))
        if m["summary"]:
            lines.append(f"    {shorten(m['summary'], 140)}")
    if len(out["matches"]) > 60:
        lines.append(f"... +{len(out['matches']) - 60} more")
    if not out["matches"]:
        lines.append("No symbol with that name.")
        if out["suggestions"]:
            lines.append("Did you mean: " + ", ".join(out["suggestions"]))
        lines.append(f'Try: search "{s}"')
    return out, "\n".join(lines), 0 if out["matches"] or out["aliases"] else 1


def cmd_members(ix: Index, args) -> tuple:
    crow = ix.class_row(args.cls)
    if crow is None:
        row, _, cands = resolve_id(ix, args.cls)
        if row is None or row["kind"] != "class":
            return not_found(ix, args.cls, cands)
        crow = ix.class_row(row["id"])
    kinds = parse_kinds(args.kind)
    chain = [crow["name"]] + (json.loads(crow["inherits"]) if args.inherited else [])
    lines, data = [f"# {crow['name']} members (Godot {ix.version})"], {"class": crow["name"], "groups": []}
    if crow["name"] in ix.deprecated:
        lines.append(f"DEPRECATED: {crow['name']} is deprecated (all its members too). {first_sentence(ix.deprecated[crow['name']])}")
        data["deprecated"] = ix.deprecated[crow["name"]]
    if not args.inherited and json.loads(crow["inherits"]):
        lines.append(f"Inherits: {' < '.join(json.loads(crow['inherits']))} (add --inherited to include their members)")
    for c in chain:
        listing = member_listing(ix, c, with_summary=not args.no_summary, kinds=kinds, name_filter=args.filter or "")
        if c != crow["name"]:
            if not listing:
                continue
            lines += ["", f"## Inherited from {c}"]
        lines += [""] + listing
        rows = [r for r in members_of(ix, c) if (not kinds or r["kind"] in kinds)
                and (not args.filter or args.filter.lower() in r["name"].lower())]
        data["groups"].append({"class": c, "members": [{"id": r["id"], "kind": r["kind"], "signature": r["signature"],
                                                         "summary": r["summary"]} for r in rows]})
    text = tidy(lines)
    page, next_off = paginate(text, args.offset, args.max_chars)
    if next_off is not None:
        page += f"\n\n[truncated: chars {args.offset}-{next_off} of {len(text)}. Continue with the same command plus --offset {next_off}]"
    return data, page, 0


def cmd_hierarchy(ix: Index, args) -> tuple:
    crow = ix.class_row(args.cls)
    if crow is None:
        row, _, cands = resolve_id(ix, args.cls)
        if row is None or row["kind"] != "class":
            return not_found(ix, args.cls, cands)
        crow = ix.class_row(row["id"])
    anc = json.loads(crow["inherits"])
    lines = [f"# {crow['name']}  - {crow['brief']}", "Ancestors: " + (" < ".join([crow["name"]] + anc) if anc else "(none)"), ""]
    data = {"class": crow["name"], "ancestors": anc, "descendants": []}

    def walk(name, depth, prefix):
        kids = json.loads(ix.class_row(name)["inherited_by"])
        for k in kids:
            kr = ix.class_row(k)
            lines.append(f"{prefix}- {k}" + (f": {shorten(kr['brief'], 90)}" if kr and kr["brief"] and args.brief else ""))
            data["descendants"].append({"name": k, "parent": name, "depth": depth})
            if depth < args.depth and kr:
                walk(k, depth + 1, prefix + "  ")

    lines.append(f"Subclasses (depth {args.depth}):")
    walk(crow["name"], 1, "")
    if not data["descendants"]:
        lines.append("(none)")
    return data, "\n".join(lines), 0


def cmd_examples(ix: Index, args) -> tuple:
    q = analyze_query(ix, args.query)
    out = {"query": args.query, "notes": q.notes, "results": []}
    lines = [f'Godot {ix.version} code examples: "{args.query}" (lang: {args.lang})']
    lines += [f"NOTE {n}" for n in q.notes]
    if q.terms:
        match = " OR ".join(fts_phrase(t) for t in q.terms)
        cand = ix.conn.execute(
            "SELECT c.n, c.entry, c.lang, c.grp, c.code, bm25(code_fts, 1.0, 3.0) AS r FROM code_fts "
            "JOIN code c ON c.n = code_fts.rowid WHERE code_fts MATCH ? ORDER BY r LIMIT 300", [match]).fetchall()
        rows = ix.by_rowids({c["entry"] for c in cand})
        groups: dict = {}
        if args.lang != "all":
            for c in ix.conn.execute(f"SELECT grp, lang FROM code WHERE grp IN ({','.join('?' * len(cand))})",
                                     [c["grp"] for c in cand]):
                groups.setdefault(c["grp"], set()).add(c["lang"])
        scored = []
        for c in cand:
            other = "csharp" if args.lang == "gdscript" else "gdscript"
            if args.lang != "all" and c["lang"] == other and (args.lang == "gdscript" or args.lang in groups.get(c["grp"], ())):
                continue
            row = rows[c["entry"]]
            if args.cls and (row["cls"] or "").lower() != args.cls.lower():
                continue
            hits = sum(1 for t in q.terms if t in c["code"].lower() or t in (row["title"] or "").lower())
            scored.append(((-c["r"]) * entry_prior(ix, row, q, args.lang) * (0.5 + hits / len(q.terms)), c, row))
        scored.sort(key=lambda x: -x[0])
        seen = set()
        for _, c, row in scored:
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            code_lines = c["code"].split("\n")
            code = "\n".join(code_lines[: args.max_lines])
            out["results"].append({"id": row["id"], "title": row["title"], "lang": c["lang"], "code": code,
                                   "truncated": len(code_lines) > args.max_lines})
            lines += ["", f"## {len(out['results'])}. {row['id']}"] + ([f"   {row['title']}"] if row["title"] != row["id"] else [])
            lines += [f"```{c['lang']}", code,
                      "```" + (f"  (+{len(code_lines) - args.max_lines} more lines: read {row['id']})" if len(code_lines) > args.max_lines else "")]
            if len(out["results"]) >= args.limit:
                break
    if not out["results"]:
        lines.append("No code examples found. Try `search` instead.")
    return out, "\n".join(lines), 0 if out["results"] else 1


def cmd_toc(ix: Index, args) -> tuple:
    prefix = (args.path or "").strip().removeprefix("doc:").strip("/")
    page = ix.get(f"doc:{prefix}") if prefix else None
    if page and page["kind"] == "page" and not prefix.endswith("index"):
        secs = ix.conn.execute("SELECT id, name, level FROM entries WHERE parent = ? ORDER BY n", (page["id"],)).fetchall()
        lines = [f"# {page['title']}  [{page['id']}]"] + ["  " * (r["level"] - 2) + f"- {r['name']}  [{r['id']}]" for r in secs]
        return {"page": page["id"], "sections": [dict(r) for r in secs]}, "\n".join(lines), 0
    rows = ix.conn.execute("SELECT id, title FROM entries WHERE kind = 'page' AND id LIKE ? ORDER BY id",
                           (f"doc:{prefix}{'/' if prefix else ''}%",)).fetchall()
    if not rows:
        return {"error": "not_found"}, f"No manual pages under '{prefix}'. Run `toc` without arguments to see the top level.", 1
    depth = prefix.count("/") + 1 if prefix else 0
    dirs: dict = {}
    pages = []
    for r in rows:
        rel = r["id"][4:]
        parts = rel.split("/")
        if len(parts) > depth + 1:
            d = "/".join(parts[: depth + 1])
            dirs[d] = dirs.get(d, 0) + 1
        else:
            pages.append(r)
    lines = [f"# Godot {ix.version} manual: /{prefix}"]
    if dirs:
        lines += ["", "Folders (browse with `toc <folder>`):"]
        for d, cnt in dirs.items():
            idx = ix.get(f"doc:{d}/index")
            lines.append(f"- {d}/  ({cnt} pages)" + (f" - {idx['title']}" if idx else ""))
    if pages:
        lines += ["", "Pages (`read <id>`; sections: `toc <id>`):"]
        lines += [f"- {r['id']}  - {r['title']}" for r in pages]
    return {"folders": dirs, "pages": [dict(r) for r in pages]}, "\n".join(lines), 0


def cmd_index(args) -> tuple:
    index_path = default_index_path(args.index)
    docs_root = find_docs_root(args.docs)
    if args.rebuild:
        rebuild_or_keep(docs_root, index_path, read_index_meta(index_path), force=True)
    ix = open_index(args)
    meta = dict(ix.meta, index_path=str(index_path))
    if args.vectors:
        ge = load_embed()
        try:
            ge.store_for(ix, args.model, build_limit=None)
        except RuntimeError as exc:
            raise DocsError(str(exc)) from exc
    if args.vectors or args.model:
        meta.update(load_embed().vector_status(ix, args.model))
    return meta, "\n".join(f"{k}: {v}" for k, v in meta.items()), 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

EPILOG = """examples:
  %(prog)s search "move character with keyboard"
  %(prog)s search "timer" --kind class
  %(prog)s search "apply force" --class RigidBody2D
  %(prog)s read CharacterBody2D.move_and_slide
  %(prog)s read doc:tutorials/2d/2d_movement
  %(prog)s lookup look_at
  %(prog)s members CharacterBody2D --inherited --filter velocity
  %(prog)s hierarchy Node2D
  %(prog)s examples "tween property"
  %(prog)s toc tutorials/2d

IDs: classes `Node2D`, members `Node2D.look_at`, pages `doc:tutorials/2d/2d_movement`,
sections `doc:tutorials/2d/2d_movement#8-way-movement`. Add --json for machine-readable output.
"""


def make_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="machine-readable JSON output")
    common.add_argument("--lang", choices=LANGS, default=os.environ.get("GODOT_DOCS_LANG", "gdscript"),
                        help="code language to show for GDScript/C# tab pairs (default: gdscript)")
    common.add_argument("--docs", help="path to the godot-docs checkout (default: auto-detect, or GODOT_DOCS_PATH)")
    common.add_argument("--index", help="path to the index database (default: next to this script, or GODOT_DOCS_INDEX)")

    p = argparse.ArgumentParser(prog="godot_docs.py", description=__doc__.split("\n")[0],
                                epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", parents=[common], help="ranked search over API reference and manual")
    s.add_argument("query")
    s.add_argument("--also", action="append", metavar="PHRASING",
                   help="another phrasing of the same question, ideally in Godot terms (class/method names); "
                        "repeatable; the result lists are merged")
    s.add_argument("-k", "--kind", help="filter: class, method, property, signal, constant, enum, annotation, "
                                         "theme_item, page, section, or groups api / member / docs (comma-separated)")
    s.add_argument("-c", "--class", dest="cls", help="only this class and its members")
    s.add_argument("-p", "--path", help="only manual pages under this path, e.g. tutorials/physics")
    s.add_argument("-n", "--limit", type=int, default=8)
    s.add_argument("-m", "--mode", choices=MODES, default=os.environ.get("GODOT_DOCS_MODE", "lexical"),
                   help="lexical (default, stdlib only); semantic (embeddings only) or hybrid (both) need numpy, "
                        "onnxruntime and tokenizers")
    s.add_argument("--model", help="embedding model folder under models/ (default: fine-tuned if present)")
    s.add_argument("--pretty", action="store_true", help="colourful, full output for humans (default output is for AI agents)")

    r = sub.add_parser("read", parents=[common], help="full text of a class, member, page or section")
    r.add_argument("id")
    r.add_argument("--full", action="store_true", help="for classes: include every member's full description")
    r.add_argument("--offset", type=int, default=0, help="continue truncated output from this character")
    r.add_argument("--max-chars", type=int, default=None,
                   help=f"output budget (0 = unlimited; default {DEFAULT_MAX_CHARS}, or unlimited with --pretty)")
    r.add_argument("--pretty", action="store_true", help="colourful, full output for humans (default output is for AI agents)")

    l = sub.add_parser("lookup", parents=[common], help="where is a symbol defined? (also Godot 3 names, typos)")
    l.add_argument("symbol")

    m = sub.add_parser("members", parents=[common], help="list a class's members")
    m.add_argument("cls", metavar="class")
    m.add_argument("--inherited", action="store_true", help="include members of all ancestor classes")
    m.add_argument("-k", "--kind", help="only these kinds (method, property, signal, ...)")
    m.add_argument("-f", "--filter", help="only members whose name contains this text")
    m.add_argument("--no-summary", action="store_true", help="signatures only")
    m.add_argument("--offset", type=int, default=0)
    m.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)

    h = sub.add_parser("hierarchy", parents=[common], help="ancestors and subclasses of a class")
    h.add_argument("cls", metavar="class")
    h.add_argument("--depth", type=int, default=1, help="subclass levels to show (default 1)")
    h.add_argument("--brief", action="store_true", help="show a one-line description per subclass")

    e = sub.add_parser("examples", parents=[common], help="search code examples only")
    e.add_argument("query")
    e.add_argument("-c", "--class", dest="cls", help="only examples from this class's reference")
    e.add_argument("-n", "--limit", type=int, default=5)
    e.add_argument("--max-lines", type=int, default=40)

    t = sub.add_parser("toc", parents=[common], help="browse the manual's page tree")
    t.add_argument("path", nargs="?", help="folder (e.g. tutorials/2d) or page id to list sections")

    i = sub.add_parser("index", parents=[common], help="show index status, or rebuild it")
    i.add_argument("--rebuild", action="store_true")
    i.add_argument("--vectors", action="store_true",
                   help="build/refresh embedding vectors for --mode semantic/hybrid (minutes on CPU; resumable)")
    i.add_argument("--model", help="embedding model folder under models/ (default: fine-tuned if present)")
    return p


COMMANDS = {
    "search": cmd_search, "read": cmd_read, "lookup": cmd_lookup, "members": cmd_members,
    "hierarchy": cmd_hierarchy, "examples": cmd_examples, "toc": cmd_toc,
}


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = make_parser().parse_args(argv)
    try:
        if args.cmd == "index":
            data, text, code = cmd_index(args)
        else:
            ix = open_index(args)
            data, text, code = COMMANDS[args.cmd](ix, args)
    except DocsError as exc:
        if getattr(args, "json", False):
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "pretty", False) and not args.json and "error" not in data:
        import godot_pretty  # human output only; never loaded for the default (AI) format
        text = godot_pretty.search(data) if args.cmd == "search" else godot_pretty.read(data)
    print(json.dumps(data, ensure_ascii=False, indent=1) if args.json else text)
    return code


if __name__ == "__main__":
    sys.exit(main())
