"""Fit the per-mode confidence models (CONF_MODELS in godot_docs.py) on the pooled-label benchmark.

Fits on the dev + val splits, reports tier precision on the held-out test split, then writes the coefficients
into godot_docs.py.   python train/fit_confidence.py [--dry-run]
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

MODES = {  # mode name -> (search mode, model, features)
    "lexical": ("lexical", None, ["cov", "tcov", "rel", "rank", "doc"]),
    "semantic": ("semantic", "gdocs-bge-small", ["cos", "rel", "rank", "doc"]),
    "hybrid": ("hybrid", "gdocs-bge-small", ["cov", "tcov", "cos", "rel", "rank", "doc"]),
}


def collect(ix, cases, mode, model, feats):
    X, y = [], []
    for q, expected, _ in cases:
        for r in gd.run_search(ix, q, limit=10, mode=mode, model=model)["results"]:
            sig = r["_signals"]
            if sig.get("exact"):
                continue  # exact-name matches are handled by a fixed rule
            X.append([sig.get(f, 0.0) if f != "rank" else sig["rank"] / 10 for f in feats])
            y.append(float(any(ev.matches(r["id"], e) for e in expected)))
    return np.array(X), np.array(y)


def fit(X, y, l2=0.01, steps=4000, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    for _ in range(steps):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        w -= lr * (Z.T @ (p - y) / len(y) + l2 * w)
        b -= lr * float(np.mean(p - y))
    return w / sd, b - float(np.sum(w * mu / sd))  # back to raw-feature scale


def tiers(X, y, w, b):
    p = 1 / (1 + np.exp(-(X @ w + b)))
    out = {}
    for name, lo, hi in (("strong", gd.TIER_STRONG, 1.01), ("partial", gd.TIER_PARTIAL, gd.TIER_STRONG), ("weak", 0, gd.TIER_PARTIAL)):
        m = (p >= lo) & (p < hi)
        out[name] = (int(m.sum()), float(y[m].mean()) if m.any() else float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    ix = gd.open_index(argparse.Namespace(docs=None, index=None))
    gd.CONF_MODELS = {}
    fitset = ev.load_cases("semantic", "dev") + ev.load_cases("semantic", "val")
    test = ev.load_cases("semantic", "test")
    models = {}
    for name, (mode, model, feats) in MODES.items():
        X, y = collect(ix, fitset, mode, model, feats)
        w, b = fit(X, y)
        models[name] = {"bias": round(b, 4), "w": {f if f != "rank" else "rank": round(float(c) / (10 if f == "rank" else 1), 4)
                                                   for f, c in zip(feats, w)}}
        Xt, yt = collect(ix, test, mode, model, feats)
        print(f"{name}: fit on {len(y)} results ({y.mean():.0%} relevant); coefficients {models[name]}")
        print(f"   fit  tiers (count, precision): {tiers(X, y, w, b)}")
        print(f"   TEST tiers (count, precision): {tiers(Xt, yt, w, b)}", flush=True)
    if args.dry_run:
        return
    p = ROOT / "src" / "godot_docs.py"
    src = p.read_text(encoding="utf-8")
    src = re.sub(r"^CONF_MODELS: dict = .*$", "CONF_MODELS: dict = " + json.dumps(models), src, count=1, flags=re.M)
    p.write_text(src, encoding="utf-8")
    print("wrote CONF_MODELS into godot_docs.py")


if __name__ == "__main__":
    main()
