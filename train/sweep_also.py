"""Tune multi-phrasing fusion (search --also) on dev/val; core must not regress. Test split is NOT scored here.

Progress (step i/N, elapsed, ETA) goes to stdout unbuffered; run with `python -u ... > train/results/sweep_also.log`.
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
import eval_queries as ev  # noqa: E402
import godot_docs as gd  # noqa: E402

FUSIONS = (("rrf", 5), ("rrf", 10), ("rrf", 30), ("rrf", 60), ("rel_sum", 0), ("rel_max", 0), ("conf_or", 0))


def fmt(m):
    return f"{m['passed']:3d}/{m['n']} MRR {m['mrr']:.3f} R@10 {m['recall10']:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", type=int, default=0, help="only the first N queries per split (pipeline check)")
    ap.add_argument("--modes", default="lexical,hybrid")
    args = ap.parse_args()
    ix = gd.open_index(argparse.Namespace(docs=None, index=None))
    also = ev.load_also()
    cut = (lambda c: c[:args.smoke]) if args.smoke else (lambda c: c)
    sets = {s: cut(ev.load_cases("semantic", s)) for s in ("dev", "val")}
    core = cut(ev.load_cases("core", None))
    modes = args.modes.split(",")

    # each step = one (mode, config) row; cost ~ queries searched (multi-phrasing rows search twice)
    configs = [("single", None, 0), ("godot-terms only", None, 0)] + [(f"{f}{k if f == 'rrf' else ''}", f, k) for f, k in FUSIONS]
    steps = [(m, c) for m in modes for c in configs]
    n_sets = sum(len(c) for c in sets.values())
    cost = {c[0]: (n_sets + len(core)) if c[0] == "single" else n_sets * (1 if c[1] is None else 2) for c in configs}
    total = sum(cost[c[0]] for _, c in steps)
    done, t0 = 0, time.time()
    print(f"SWEEP START: {len(steps)} steps, ~{total} searches, modes {modes}, "
          f"{'SMOKE ' + str(args.smoke) + ' queries/split' if args.smoke else 'full splits'}", flush=True)
    for i, (mode, (name, fusion, rk)) in enumerate(steps, 1):
        gd.RRF_K = rk or gd.RRF_K
        cols = []
        for s, c in sets.items():
            if name == "single":
                m = ev.evaluate(ix, c, mode, quiet=True)
            elif fusion is None:  # Godot-terms phrasing alone
                m = ev.evaluate(ix, [(also[q], e, k) for q, e, k in c], mode, quiet=True)
            else:
                m = ev.evaluate(ix, c, mode, quiet=True, also=also, fusion=fusion)
            cols.append(f"{s} {fmt(m)}")
        if name == "single":
            cols.append(f"core {fmt(ev.evaluate(ix, core, mode, quiet=True))}")
        done += cost[name]
        el = time.time() - t0
        eta = el / done * (total - done)
        print(f"[{i:2d}/{len(steps)}] {mode:8s} {name:16s} {'  '.join(cols)}   "
              f"(elapsed {el / 60:.1f} min, ETA {eta / 60:.1f} min)", flush=True)
    print(f"SWEEP DONE in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
