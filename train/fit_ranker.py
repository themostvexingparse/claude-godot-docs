"""Learn the ranking weights (RANK_MODELS in godot_docs.py) from judged benchmark queries.

Pairwise logistic ranking ("the judged answer should outrank this other candidate") over the signals that
run_search computes for every candidate. Exact-name / Godot 3 alias rules stay fixed on top.

    python train/fit_ranker.py            # 2-fold CV on dev/val vs the hand formula, core check (no write)
    python train/fit_ranker.py --write    # fit on dev + val and write RANK_MODELS into godot_docs.py
    python train/fit_ranker.py --test     # one-time held-out comparison on the test split (after --write)

Labels: expected/accept ids of eval/semantic_queries.json; unjudged candidates count as wrong (pooled judging
covered the top 5 of every system, so this mostly affects deep candidates).
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
import eval_queries as ev  # noqa: E402
import godot_docs as gd  # noqa: E402

# lhand = log of the hand formula's soft score; the model learns corrections around it (it starts at weight 1)
FEATS = {
    "lexical": ["lhand", "bm", "cov2", "tcov", "doc", "ngram", "chint", "mhint", "thint"],
    "hybrid": ["lhand", "bm", "cov2", "tcov", "doc", "ngram", "chint", "mhint", "thint", "sem", "cos"],
}
SEARCH_MODE = {"lexical": "lexical", "hybrid": "hybrid"}


def collect(ix, cases, kind):
    """Per query: (feature matrix, labels, rules) for every candidate the search scored."""
    out = []
    saved = gd.RANK_MODE
    gd.RANK_MODE = "hand"
    for q, expected, _ in cases:
        gd._TRACE = []  # filled by run_search with every scored candidate
        gd.run_search(ix, q, limit=10, mode=SEARCH_MODE[kind])
        tr, gd._TRACE = gd._TRACE, None
        if not tr:
            continue
        tr = [t for t in tr if t["rule"] == 0]  # rule-decided candidates are not what we learn
        y = np.array([float(any(ev.matches(t["id"], e) for e in expected)) for t in tr])
        if not y.any() or y.all():
            continue  # nothing to learn from this query
        # keep every judged answer and the 100 strongest wrong candidates (the ones that compete for the top)
        neg = sorted((i for i in range(len(tr)) if not y[i]), key=lambda i: -tr[i]["score"])[:100]
        keep = [i for i in range(len(tr)) if y[i]] + neg
        X = np.array([[tr[i]["f"].get(k, 0.0) for k in FEATS[kind]] for i in keep])
        off = np.array([np.log(tr[i]["prior"]) for i in keep])  # fixed multiplier, as in the hand formula
        out.append((X, y[keep], off, kind))
    gd.RANK_MODE = saved
    return out


def fit(groups, l2=0.05, steps=4000, lr=0.05):
    """Pairwise logistic loss (each query weighted equally), regularised toward the hand formula (lhand = 1)."""
    w0 = np.array([1.0 if k == "lhand" else 0.0 for k in FEATS[groups[0][3]]])
    diffs, offs, weights = [], [], []
    for X, y, off, _ in groups:
        pos, neg = np.nonzero(y == 1)[0], np.nonzero(y == 0)[0]
        D = (X[pos][:, None, :] - X[neg][None, :, :]).reshape(-1, X.shape[1])
        diffs.append(D)
        offs.append((off[pos][:, None] - off[neg][None, :]).reshape(-1))
        weights.append(np.full(len(D), 1.0 / (len(D) * len(groups))))
    D, O, wt = np.vstack(diffs), np.concatenate(offs), np.concatenate(weights)
    w = w0.copy()
    for _ in range(steps):  # full-batch gradient descent
        p = 1.0 / (1.0 + np.exp(np.clip(D @ w + O, -30, 30)))
        w -= lr * (l2 * (w - w0) - D.T @ (wt * p))
    return w, 0.0


def as_model(kind, w, b):
    return {"bias": round(b, 4), "w": {k: round(float(v), 4) for k, v in zip(FEATS[kind], w)}}


def score(ix, cases, kind, model=None):
    gd.RANK_MODE = "learned" if model else "hand"
    gd.RANK_MODELS = {kind: model} if model else {}
    m = ev.evaluate(ix, cases, SEARCH_MODE[kind], None, quiet=True)
    gd.RANK_MODE, gd.RANK_MODELS = "hand", {}
    return m


def fmt(m):
    return f"{m['passed']}/{m['n']} MRR {m['mrr']:.3f} R@10 {m['recall10']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--l2", type=float, nargs="*", default=[1.0, 0.1, 0.01, 0.001])
    args = ap.parse_args()
    ix = gd.open_index(argparse.Namespace(docs=None, index=None))
    dev, val = ev.load_cases("semantic", "dev"), ev.load_cases("semantic", "val")
    core, test = ev.load_cases("core", None), ev.load_cases("semantic", "test")

    if args.test:
        src = (ROOT / "src" / "godot_docs.py").read_text(encoding="utf-8")
        models = json.loads(re.search(r"^RANK_MODELS: dict = (.*)$", src, re.M).group(1))
        for kind in FEATS:
            print(f"TEST {kind:7s} hand {fmt(score(ix, test, kind))} | learned {fmt(score(ix, test, kind, models[kind]))}")
        return

    models = {}
    for kind in FEATS:
        g_dev, g_val = collect(ix, dev, kind), collect(ix, val, kind)
        print(f"== {kind}: {len(g_dev)} dev / {len(g_val)} val queries with a judged answer among candidates")
        print(f"   hand     dev {fmt(score(ix, dev, kind))} | val {fmt(score(ix, val, kind))} | core {fmt(score(ix, core, kind))}")
        best = None
        for l2 in args.l2:
            m_val = as_model(kind, *fit(g_dev, l2))   # train dev -> score val
            m_dev = as_model(kind, *fit(g_val, l2))   # train val -> score dev
            a, b = score(ix, dev, kind, m_dev), score(ix, val, kind, m_val)
            full = as_model(kind, *fit(g_dev + g_val, l2))
            c = score(ix, core, kind, full)
            cv = (a["mrr"] + b["mrr"]) / 2
            print(f"   l2={l2:<5} CV dev {fmt(a)} | CV val {fmt(b)} | core (fit on both) {fmt(c)} | CV MRR {cv:.3f}")
            if best is None or cv > best[0]:
                best = (cv, l2, full)
        print(f"   chosen l2={best[1]}: {best[2]}")
        models[kind] = best[2]
    if args.write:
        p = ROOT / "src" / "godot_docs.py"
        src = p.read_text(encoding="utf-8")
        src = re.sub(r"^RANK_MODELS: dict = .*$", "RANK_MODELS: dict = " + json.dumps(models), src, count=1, flags=re.M)
        p.write_text(src, encoding="utf-8")
        print("wrote RANK_MODELS into godot_docs.py")


if __name__ == "__main__":
    main()
