"""Human-oriented rendering for `search --pretty` and `read --pretty`: boxed tables, colours, highlighted code.

Stdlib only (ANSI escapes). Imported only when --pretty is used, so the default AI-oriented output path never
loads it. Set NO_COLOR=1 to keep the layout without colours.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap

STYLES = {
    "bold": "1", "dim": "2", "italic": "3", "under": "4",
    "red": "31", "green": "32", "yellow": "33", "blue": "34", "magenta": "35", "cyan": "36", "grey": "90",
    "bgreen": "92", "byellow": "93", "bblue": "94", "bmagenta": "95", "bcyan": "96",
}
TIER_STYLE = {"strong": ("bold", "bgreen"), "partial": ("bold", "byellow"), "weak": ("grey",)}
TIER_ICON = {"strong": "●", "partial": "◐", "weak": "○"}
KIND_STYLE = {"class": "bmagenta", "method": "bcyan", "property": "bblue", "signal": "byellow", "constant": "yellow",
              "enum": "yellow", "annotation": "magenta", "theme_item": "blue", "page": "bgreen", "section": "green"}
GD_KEYWORDS = set("""
if elif else for while match break continue pass return class class_name extends is in as self signal func static
const enum var await yield breakpoint preload super and or not true false null void PI TAU INF NAN when
""".split())
CODE_TOKEN = re.compile(r'(?P<comment>#.*$|//.*$)|(?P<string>"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')'
                        r'|(?P<annotation>@\w+)|(?P<number>\b\d+(?:\.\d+)?\b)|(?P<word>[A-Za-z_]\w*)(?P<call>\s*\()?')
ANSI = re.compile(r"\x1b\[[0-9;]*m")
MEMBER_GROUP = re.compile(r"^(Properties|Theme properties|Methods|Signals|Constants|Enumerations|Constructors|"
                          r"Operators|Annotations) \((\d+)\)$")
FLAG = re.compile(r"\s+\[(deprecated|experimental)\]$")
OVERLOADS = re.compile(r"\s+\(\+(\d+) overloads?\)$")


def _use_colour() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if sys.platform == "win32":  # turn on ANSI escape processing for classic consoles
        try:
            import ctypes
            k = ctypes.windll.kernel32
            h = k.GetStdHandle(-11)
            mode = ctypes.c_uint32()
            if k.GetConsoleMode(h, ctypes.byref(mode)):
                k.SetConsoleMode(h, mode.value | 0x0004)
        except (AttributeError, OSError):
            pass
    return True


COLOUR = _use_colour()


def c(text: str, *styles: str) -> str:
    if not COLOUR or not styles or not text:
        return text
    return f"\x1b[{';'.join(STYLES[s] for s in styles)}m{text}\x1b[0m"


def vlen(text: str) -> int:
    """Visible width (ANSI escapes excluded)."""
    return len(ANSI.sub("", text))


def width() -> int:
    return max(70, min(shutil.get_terminal_size((110, 24)).columns, 140))


def inline(text: str) -> str:
    """Inline markdown: `code`, **bold**, [doc:...] / [label](url) references."""
    spans: list[str] = []

    def stash(m):  # code spans first, kept out of the bold/link patterns (e.g. the `**` operator)
        spans.append(c(m.group(1), "bcyan"))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = re.sub(r"\*\*([^*]+)\*\*", lambda m: c(m.group(1), "bold"), text)
    text = re.sub(r"\[(doc:[^\]]+)\]", lambda m: c(f"[{m.group(1)}]", "blue"), text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", lambda m: c(m.group(1), "under") + c(f" <{m.group(2)}>", "grey"), text)
    return re.sub("\x00(\\d+)\x00", lambda m: spans[int(m.group(1))], text)


def wrap_plain(text: str, w: int) -> list[str]:
    lines = []
    for para in text.split("\n"):
        lines += textwrap.wrap(para, width=max(4, w), break_long_words=True, break_on_hyphens=False) or [""]
    return lines


def wrap_id(text: str, w: int) -> list[str]:
    """Wrap an entry id at its natural joints (/ # .) instead of mid-word."""
    lines, cur = [], ""
    for part in re.split(r"(?<=[/#.])", text):
        if cur and len(cur) + len(part) > w:
            lines.append(cur)
            cur = ""
        while len(part) > w:  # a single segment longer than the column
            lines.append(part[:w])
            part = part[w:]
        cur += part
    return lines + [cur] if cur else lines or [""]


def highlight_code(line: str) -> str:
    out, pos = [], 0
    for m in CODE_TOKEN.finditer(line):
        out.append(line[pos:m.start()])
        tok = m.group(0)
        if m.group("comment"):
            out.append(c(tok, "grey", "italic"))
        elif m.group("string"):
            out.append(c(tok, "green"))
        elif m.group("annotation"):
            out.append(c(tok, "magenta"))
        elif m.group("number"):
            out.append(c(tok, "byellow"))
        else:
            word, call = m.group("word"), m.group("call") or ""
            if word in GD_KEYWORDS:
                out.append(c(word, "bold", "red") + call)
            elif word[:1].isupper() and not word.isupper() or word in ("int", "float", "bool"):
                out.append(c(word, "bmagenta") + call)
            elif call:
                out.append(c(word, "bblue") + call)
            else:
                out.append(word)
        pos = m.end()
    out.append(line[pos:])
    return "".join(out)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------


class Col:
    """A table column. `style(line) -> str` styles each wrapped line; `flex` columns absorb the leftover width."""

    def __init__(self, title: str, min_w: int = 4, max_w: int = 40, flex: int = 0, style=None, align: str = "<",
                 wrap=wrap_plain):
        self.title, self.min_w, self.max_w, self.flex, self.style, self.align = title, min_w, max_w, flex, style, align
        self.wrap = wrap


def _fit(cols: list, rows: list, total: int) -> list[int]:
    natural = []
    for i, col in enumerate(cols):
        longest = max([len(col.title)] + [max((len(x) for x in str(r[i]).split("\n")), default=0) for r in rows])
        natural.append(max(col.min_w, min(col.max_w, longest)))
    avail = total - (3 * len(cols) + 1)
    widths = natural[:]
    flex_idx = [i for i, col in enumerate(cols) if col.flex]
    over = sum(widths) - avail
    if over > 0:  # shrink flexible columns (proportionally), then anything above its minimum
        for group in (flex_idx, list(range(len(cols)))):
            room = sum(widths[i] - cols[i].min_w for i in group)
            if room <= 0:
                continue
            take = min(over, room)
            for i in group:
                cut = round(take * (widths[i] - cols[i].min_w) / room)
                widths[i] -= cut
                over -= cut
            if over <= 0:
                break
        while sum(widths) > avail and max(widths) > 1:  # rounding leftovers: trim the widest column
            widths[widths.index(max(widths))] -= 1
    elif flex_idx:  # grow flexible columns into the spare width, up to their natural need
        spare = -over
        for i in flex_idx:
            need = max(0, max([vlen(str(r[i])) for r in rows] + [0]) - widths[i])
            grow = min(spare, need)
            widths[i] += grow
            spare -= grow
    return widths


def table(cols: list, rows: list, total: int | None = None, separators: bool = True, title: str = "") -> str:
    total = total or width()
    widths = _fit(cols, rows, total)
    g = lambda s: c(s, "grey")  # noqa: E731
    top = g("┌" + "┬".join("─" * (w + 2) for w in widths) + "┐")
    mid = g("├" + "┼".join("─" * (w + 2) for w in widths) + "┤")
    thin = g("├" + "┼".join("┄" * (w + 2) for w in widths) + "┤")
    bot = g("└" + "┴".join("─" * (w + 2) for w in widths) + "┘")
    out = []
    if title:
        out.append(c(title, "bold", "byellow"))
    out.append(top)
    out.append(g("│") + g("│").join(" " + c(col.title[:w].ljust(w), "bold") + " " for col, w in zip(cols, widths)) + g("│"))
    out.append(mid)
    for k, row in enumerate(rows):
        cells = []
        for col, w, val in zip(cols, widths, row):
            lines = col.wrap(str(val), w)
            styled = [(col.style(l) if col.style else l) for l in lines]
            cells.append([(s, len(l)) for s, l in zip(styled, lines)])
        height = max(len(x) for x in cells)
        for h in range(height):
            parts = []
            for col, w, cell in zip(cols, widths, cells):
                s, n = cell[h] if h < len(cell) else ("", 0)
                pad = " " * (w - n)
                parts.append(" " + (pad + s if col.align == ">" else s + pad) + " ")
            out.append(g("│") + g("│").join(parts) + g("│"))
        if separators and k < len(rows) - 1:
            out.append(thin)
    out.append(bot)
    return "\n".join(out)


def info_box(title: str, pairs: list[tuple[str, str]], total: int | None = None) -> str:
    """A boxed header: bold title line, then aligned key / value rows."""
    total = total or width()
    key_w = max([len(k) for k, _ in pairs] + [4])
    val_w = total - key_w - 6  # "│ " + key + "  " + value + " │"
    g = lambda s: c(s, "grey")  # noqa: E731
    out = [g("╭" + "─" * (total - 2) + "╮")]
    t = title if len(title) <= total - 4 else title[: total - 7] + "..."
    out.append(g("│ ") + c(t, "bold", "bcyan") + " " * (total - 4 - len(t)) + g(" │"))
    if pairs:
        out.append(g("├" + "─" * (total - 2) + "┤"))
    for k, v in pairs:
        for i, line in enumerate(wrap_plain(v, val_w)):
            key = (k if i == 0 else "").ljust(key_w)
            styled = inline(line)  # may drop markup characters, so pad by the visible width
            out.append(g("│ ") + c(key, "grey") + "  " + styled + " " * (val_w - vlen(styled)) + g(" │"))
    out.append(g("╰" + "─" * (total - 2) + "╯"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# search --pretty
# ---------------------------------------------------------------------------


def search(res: dict) -> str:
    w = width()
    mode = res.get("mode", "lexical")
    pairs = [("Godot", res["godot_version"]), ("Mode", mode), ("Results", str(len(res["results"])))]
    if res["lang"] != "gdscript":
        pairs.append(("Lang", res["lang"]))
    out = [info_box("search " + "".join([f'"{res["query"]}"'] + [f' --also "{a}"' for a in res.get("also", [])]), pairs, w)]
    for note in res["notes"]:
        for k, l in enumerate(wrap_plain(note, w - 3)):
            out.append((c(" ⚠ ", "byellow") if k == 0 else "   ") + inline(l))
    if not res["results"]:
        out.append(" No results. Try fewer or different words, `lookup <name>`, or `toc`.")
        return "\n".join(out)
    rows = []
    for i, r in enumerate(res["results"], 1):
        tier = r.get("tier") or ""
        flags = [f for f in r["flags"] if f in ("deprecated", "experimental", "outdated")]
        kind = r["kind"].replace("_", " ") + ("".join(f"\n{f}" for f in flags))
        if r["kind"] in ("page", "section"):
            snippet = re.sub(r"\s+", " ", r["snippet"]).strip().replace("**", "").replace("`", "")
            detail = f"{r['title']}\n{snippet}"
        else:
            sig = r["signature"].split("\n")[0]
            if r["kind"] == "class":
                sig = "extends " + " < ".join(sig.split(" < ")[1:3]) if " < " in sig else ""
            detail = "\n".join(x for x in (sig, (r["summary"] or "").replace("`", "").replace("**", "")) if x)
        rows.append([str(i), f"{TIER_ICON.get(tier, '')} {tier}".strip(), kind, r["id"], detail])

    def tier_style(line):
        t = line.split()[-1] if line.split() else ""
        return c(line, *TIER_STYLE[t]) if t in TIER_STYLE else line

    # first line of each detail cell: API signature (highlighted) or manual breadcrumb (green)
    api_first = {r[4].split("\n")[0] for r, res_r in zip(rows, res["results"]) if res_r["kind"] not in ("page", "section")}
    doc_first = {r[4].split("\n")[0] for r, res_r in zip(rows, res["results"]) if res_r["kind"] in ("page", "section")}

    def detail(line):
        if line in api_first:
            return highlight_code(line)
        if line in doc_first:
            return c(line, "green")
        return line

    cols = [Col("#", 2, 3, align=">", style=lambda l: c(l, "bold")),
            Col("Match", 9, 9, style=tier_style),
            Col("Kind", 8, 10, style=lambda l: c(l, "red") if l in ("deprecated", "experimental", "outdated")
                else c(l, KIND_STYLE.get(l.replace(" ", "_"), "grey"))),
            Col("Entry", 15, 46, flex=2, style=lambda l: c(l, "bold", "bcyan"), wrap=wrap_id),
            Col("Details", 20, 200, flex=3, style=detail)]
    out.append(table(cols, rows, w))
    for m in res["more"]:
        hint = f"--class {m['group']}" if not m["group"].startswith("doc:") else f"read {m['group']}"
        out += [c(l, "grey") for l in wrap_plain(f"  +{m['count']} more in {m['group']}  →  {hint}", w)]
    legend = "  ".join(c(f"{TIER_ICON[t]} {t}", *TIER_STYLE[t]) for t in ("strong", "partial", "weak"))
    out.append(c("  match: ", "grey") + legend)
    out.append(c("  full text: read <entry> --pretty", "grey"))
    return "\n".join(out)


# ---------------------------------------------------------------------------
# read --pretty
# ---------------------------------------------------------------------------


def _member_table(group: str, items: list[str], w: int) -> str:
    """Turn a class page's member list (signature bullets) into a table."""
    rows, notes_any = [], False

    def split_flags(s):
        notes = []
        m = OVERLOADS.search(s)
        if m:
            notes.append(f"+{m.group(1)} overloads")
            s = s[:m.start()]
        m = FLAG.search(s)
        if m:
            notes.append(m.group(1))
            s = s[:m.start()]
        return s.strip(), notes

    if group in ("Properties", "Theme properties"):
        for it in items:
            it, notes = split_flags(it)
            m = re.match(r"^(\S+): (.+?)(?: = (.*))?$", it)
            name, typ, default = (m.group(1), m.group(2), m.group(3) or "") if m else (it, "", "")
            rows.append([name, typ, default, ", ".join(notes)])
            notes_any |= bool(notes)
        cols = [Col("Name", 8, 40, flex=1, style=lambda l: c(l, "bblue")),
                Col("Type", 6, 34, flex=1, style=lambda l: c(l, "bmagenta")),
                Col("Default", 7, 40, flex=1, style=highlight_code)]
    elif group == "Constants":
        for it in items:
            it, notes = split_flags(it)
            name, _, value = it.partition(" = ")
            rows.append([name, value, ", ".join(notes)])
            notes_any |= bool(notes)
        cols = [Col("Name", 8, 50, flex=1, style=lambda l: c(l, "yellow")),
                Col("Value", 5, 40, flex=1, style=highlight_code)]
    elif group == "Enumerations":
        for it in items:
            it, notes = split_flags(it)
            m = re.match(r"^(enum|flags)\s+(\S+)\s*\{\s*(.*?)\s*\}$", it)
            if m:
                rows.append([m.group(2) + (" (flags)" if m.group(1) == "flags" else ""), m.group(3).replace(", ", "\n"), ""])
            else:
                rows.append([it, "", ""])
        cols = [Col("Enum", 8, 36, flex=1, style=lambda l: c(l, "yellow")),
                Col("Values", 10, 80, flex=2, style=highlight_code)]
    else:  # Methods, Signals, Constructors, Operators, Annotations
        for it in items:
            it, notes = split_flags(it)
            m = re.match(r"^(.+?)\((.*)\)(?: -> (.+?))?(?: \[(.+?)\])?$", it)
            if m:
                name, args, ret, quals = m.group(1), m.group(2), m.group(3) or "", m.group(4) or ""
                notes = ([quals] if quals else []) + notes
                rows.append([name, args.replace(", ", ",\n") if len(args) > 50 else args, ret, ", ".join(notes)])
            else:
                rows.append([it, "", "", ", ".join(notes)])
            notes_any |= bool(notes)
        first = {"Signals": "Signal", "Constructors": "Constructor", "Operators": "Operator",
                 "Annotations": "Annotation"}.get(group, "Method")
        cols = [Col(first, 8, 40, flex=1, style=lambda l: c(l, "bcyan" if group != "Signals" else "byellow")),
                Col("Arguments", 9, 70, flex=2, style=highlight_code)]
        if group not in ("Signals", "Annotations"):
            cols.append(Col("Returns", 7, 30, style=lambda l: c(l, "bmagenta")))
        else:
            rows = [r[:2] + r[3:] for r in rows]
    if notes_any:
        cols.append(Col("Notes", 5, 22, style=lambda l: c(l, "red") if l in ("deprecated", "experimental") else c(l, "grey")))
    else:
        rows = [r[:len(cols)] for r in rows]
    return table(cols, rows, w, separators=False)


def markdown(text: str) -> str:
    """Render the tool's markdown-ish text: info box, headings, member tables, fenced code, lists, paragraphs."""
    w = width()
    lines = text.split("\n")
    out, para, i = [], [], 0

    def flush():
        if para:
            out.extend(inline(l) for l in wrap_plain(" ".join(para), w))
            para.clear()

    def blank():
        if out and out[-1].strip():
            out.append("")

    # Title + metadata lines -> info box
    if lines and lines[0].startswith("# "):
        title = re.sub(r"\s*\{#[^}]+\}\s*$", "", lines[0][2:])
        pairs, i = [], 1
        m = re.match(r"^(.*?)\s+\((\w[\w ]*), Godot ([\d.]+)\)$", title)
        if m:
            title, pairs = m.group(1), [("Kind", m.group(2)), ("Godot", m.group(3))]
        while i < len(lines):
            line = lines[i]
            meta = re.match(r"^(ID|Source|Class|Inherits|Inherited by|Accessors):\s+(.*)$", line)
            if meta:
                pairs.append((meta.group(1), meta.group(2)))
            elif line.startswith("    ") and line.strip():  # member signature lines
                pairs.append(("Signature", line.strip()))
            else:
                break
            i += 1
        out.append(info_box(title, pairs, w))

    in_code, group, items, md_rows = False, None, [], []
    for line in lines[i:]:
        if group and not line.startswith("- "):  # end of a member list
            blank()
            out.append(_member_table(group, items, w))
            group, items = None, []
        fence = re.match(r"^\s*```(\S*)", line)
        if fence:
            flush()
            if not in_code:
                lang = fence.group(1).split("|")[0] or "code"
                blank()
                out.append(c(f"  ╭─ {lang} ", "grey") + c("─" * max(0, w - len(lang) - 8), "grey"))
            else:
                out.append(c("  ╰" + "─" * (w - 5), "grey"))
            in_code = not in_code
            continue
        if in_code:
            room = w - 6
            out.append(c("  │ ", "grey") + highlight_code(line[:room]))
            for k in range(room, len(line), room - 2):  # long lines continue, marked, instead of overflowing
                out.append(c("  │ ↪ ", "grey") + highlight_code(line[k:k + room - 2]))
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            title = re.sub(r"`([^`]+)`", r"\1", re.sub(r"\s*\{#[^}]+\}\s*$", "", heading.group(2)))
            mg = MEMBER_GROUP.match(title)
            if mg and level == 3:
                group = mg.group(1)
                blank()
                out.append(c(f"{mg.group(1)}", "bold", "byellow") + c(f"  ({mg.group(2)})", "grey"))
                continue
            blank()
            parts = wrap_plain(title, w - 2)
            mark, style = ("▌ ", ("bold", "byellow")) if level <= 2 else ("▸ ", ("bold", "yellow"))
            out += [c((mark if k == 0 else "  ") + p, *style) for k, p in enumerate(parts)]
            if level <= 2:
                out.append(c("─" * min(max(len(p) for p in parts) + 2, w), "yellow"))
            continue
        if group and line.startswith("- "):
            items.append(line[2:])
            continue
        if not line.strip():
            flush()
            blank()
            continue
        bullet = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)$", line)
        if bullet:
            flush()
            pad = bullet.group(1) + "  "
            for k, l in enumerate(wrap_plain(bullet.group(3), w - len(pad) - 2)):
                out.append(pad + (c("• ", "cyan") if k == 0 else "  ") + inline(l))
            continue
        callout = re.match(r"^\*\*(Note|Tip|Warning|Important|Caution|Deprecated|Experimental):\*\*\s*(.*)$", line)
        if callout:
            flush()
            para.append(f"**{callout.group(1)}:** {callout.group(2)}")  # styled after wrapping, by inline()
            continue
        if line.lstrip().startswith("|"):  # manual's markdown tables: collected, rendered as a box table
            flush()
            md_rows.append(line)
            continue
        if md_rows:
            blank()
            out.append(_markdown_table(md_rows, w))
            md_rows.clear()
        para.append(line.strip())
    if group:
        out.append(_member_table(group, items, w))
    if md_rows:
        out.append(_markdown_table(md_rows, w))
    flush()
    return "\n".join(out).strip("\n")


def _markdown_table(lines: list[str], w: int) -> str:
    rows = []
    for line in lines:
        cells = [x.strip() for x in line.strip().strip("|").split("|")]
        if all(set(x) <= set("-: ") for x in cells):
            continue  # separator row
        rows.append([re.sub(r"\*\*([^*]+)\*\*", r"\1", x).replace("`", "") for x in cells])
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    n = len(header)
    fixed = []
    for r in body:
        if len(r) == n + 1 and not r[0]:
            r = r[1:]  # converter artifact: stray leading empty cell
        fixed.append((r + [""] * n)[:n] if len(r) <= n else r[:n - 1] + [" ".join(r[n - 1:])])
    cols = [Col(h or " ", 6, 200, flex=1, style=(lambda l: c(l, "bcyan")) if k == 0 else None) for k, h in enumerate(header)]
    return table(cols, fixed, w, separators=True)


def read(data: dict) -> str:
    text = data["text"]
    if data.get("note"):
        text = re.sub(r"^NOTE [^\n]*\n+", "", text)  # the plain-text note line; shown as a callout instead
    body = markdown(text)
    if data.get("note"):
        note = [(c(" ⚠ ", "byellow") if k == 0 else "   ") + inline(l) for k, l in enumerate(wrap_plain(data["note"], width() - 3))]
        body = "\n".join(note) + "\n\n" + body
    return body
