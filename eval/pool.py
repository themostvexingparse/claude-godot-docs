"""Pooled relevance judging (TREC-style) for eval/semantic_queries.json.

For every query, the top-K results of several systems are merged, entries already accepted by the labels are
removed, and the rest are shuffled (so the judge can't tell which system returned what). A judge then marks the
candidates that answer the query according to RUBRIC; `apply` writes them to the query's `accept` list.

    python eval/pool.py build [--k 5]          # -> eval/pool.json
    python eval/pool.py show --from 0 --to 25  # compact view for judging
    python eval/pool.py apply eval/judgments/*.json
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "src"))
import eval_queries as ev  # noqa: E402
import godot_docs as gd  # noqa: E402

RUBRIC = """An entry is RELEVANT if a developer asking the query, after reading that entry alone, would know what
to use or do: the member/class that performs the task, or a manual section that specifically explains it.
NOT relevant: entries that merely share words, the wrong dimension (3D-only entry for a query that says 2D,
and vice versa), generic overviews that only mention the topic in passing, deprecated APIs."""
SYSTEMS = [("lexical", None, None), ("semantic", "bge-small-en-v1.5", None), ("hybrid", "bge-small-en-v1.5", 0.2),
           ("semantic", "gdocs-bge-small", None), ("hybrid", "gdocs-bge-small", 0.8)]
QFILE = ROOT / "semantic_queries.json"
POOL = ROOT / "pool.json"


def build(k: int, split: str | None):
    ix = gd.open_index(argparse.Namespace(docs=None, index=None))
    cases = json.loads(QFILE.read_text(encoding="utf-8"))
    pool = []
    for ci, c in enumerate(cases):
        if split and c["split"] != split:
            continue
        expected = c["expected"] + ["=" + a for a in c.get("accept", [])]
        cands = set()
        for mode, model, w in SYSTEMS:
            if w is not None:
                gd.HYBRID_WEIGHT = w
            res = gd.run_search(ix, c["query"], limit=k, mode=mode, model=model)
            cands.update(r["id"] for r in res["results"])
        cands = [x for x in cands if not any(ev.matches(x, e) for e in expected)]
        cands.sort(key=lambda x: hashlib.sha1((c["query"] + x).encode()).hexdigest())  # blind, stable order
        rows = ix.by_rowids([ix.get(x)["n"] for x in cands]).values()
        info = {r["id"]: f"({r['kind']}) {r['title'] if r['kind'] in ('page', 'section') else ''} "
                         f"{(r['signature'] or '').split(chr(10))[0] if r['kind'] not in ('page', 'section', 'class') else ''} "
                         f"- {gd.shorten(r['summary'] or '', 110)}" for r in rows}
        pool.append({"ci": ci, "query": c["query"], "labelled": c["expected"], "candidates": [[x, " ".join(info[x].split())] for x in cands]})
    POOL.write_text(json.dumps(pool, indent=1), encoding="utf-8")
    print(f"{len(pool)} queries, {sum(len(p['candidates']) for p in pool)} candidates to judge -> {POOL}")


def show(a: int, b: int):
    pool = json.loads(POOL.read_text(encoding="utf-8"))
    for p in pool[a:b]:
        print(f"Q{p['ci']} {p['query']!r}   [already: {', '.join(p['labelled'][:4])}]")
        for ci, (x, desc) in enumerate(p["candidates"]):
            print(f"  {ci}. {x} {desc}")


def apply(files: list):
    """Judgment files: {"Q<case index>": [candidate indexes judged relevant], ...}"""
    pool = {p["ci"]: p for p in json.loads(POOL.read_text(encoding="utf-8"))}
    cases = json.loads(QFILE.read_text(encoding="utf-8"))
    judged, added = 0, 0
    for f in files:
        for key, idxs in json.loads(Path(f).read_text(encoding="utf-8")).items():
            qi = int(key[1:])
            assert pool[qi]["query"] == cases[qi]["query"]
            acc = [pool[qi]["candidates"][i][0] for i in idxs]
            cases[qi]["accept"] = sorted(set(cases[qi].get("accept", [])) | set(acc))
            judged += 1
            added += len(acc)
    QFILE.write_text("[\n" + ",\n".join(json.dumps(c) for c in cases) + "\n]\n", encoding="utf-8")
    print(f"applied judgments for {judged} queries, {added} accepted entries added")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--k", type=int, default=5)
    b.add_argument("--split", help="only queries of this split (dev, val, test)")
    s = sub.add_parser("show")
    s.add_argument("--from", dest="a", type=int, default=0)
    s.add_argument("--to", dest="b", type=int, default=25, help="positions in the pool (not case indexes)")
    a = sub.add_parser("apply")
    a.add_argument("files", nargs="+")
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.k, args.split)
    elif args.cmd == "show":
        show(args.a, args.b)
    else:
        apply(args.files)
