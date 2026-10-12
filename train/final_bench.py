"""Final comparison: tune the hybrid weight on core + dev, then score the held-out test split once."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
import eval_queries as ev  # noqa: E402
import godot_docs as gd  # noqa: E402

WEIGHTS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
MODELS = ("bge-small-en-v1.5", "gdocs-bge-small")


def fmt(m):
    return f"{m['passed']}/{m['n']} ({100 * m['pass_rate']:.0f}%) MRR {m['mrr']:.3f} R@10 {m['recall10']:.2f} {m['ms_median']:.0f}ms"


def main():
    ix = gd.open_index(argparse.Namespace(docs=None, index=None))
    core, dev, test = ev.load_cases("core", None), ev.load_cases("semantic", "dev"), ev.load_cases("semantic", "test")
    best = {}
    for model in MODELS:
        rows = []
        for w in WEIGHTS:
            gd.HYBRID_WEIGHT = w
            a = ev.evaluate(ix, core, "hybrid", model, quiet=True)
            b = ev.evaluate(ix, dev, "hybrid", model, quiet=True)
            rows.append((w, a, b))
            print(f"sweep {model} w={w}: core {fmt(a)} | dev {fmt(b)}", flush=True)
        # keep the core set perfect if any weight does; then most dev passes, then dev MRR
        top_core = max(a["passed"] for _, a, _ in rows)
        w, _, _ = max((r for r in rows if r[1]["passed"] == top_core), key=lambda r: (r[2]["passed"], r[2]["mrr"]))
        best[model] = w
        print(f"chosen weight for {model}: {w} (core {top_core}/58)", flush=True)

    print("\n=== TEST split (held out, scored once) ===", flush=True)
    results = {"lexical": ev.evaluate(ix, test, "lexical", quiet=True)}
    for model in MODELS:
        results[f"{model} semantic"] = ev.evaluate(ix, test, "semantic", model, quiet=True)
        gd.HYBRID_WEIGHT = best[model]
        results[f"{model} hybrid w={best[model]}"] = ev.evaluate(ix, test, "hybrid", model, quiet=True)
        results[f"{model} core hybrid w={best[model]}"] = ev.evaluate(ix, core, "hybrid", model, quiet=True)
    results["lexical core"] = ev.evaluate(ix, core, "lexical", quiet=True)
    for k, m in results.items():
        print(f"{k:45s} {fmt(m)}", flush=True)
    out = Path(__file__).resolve().parent / "results" / "final_bench.json"
    out.write_text(json.dumps({"best_weights": best, "results": results}, indent=1), encoding="utf-8")
    print("FINAL BENCH DONE", flush=True)


if __name__ == "__main__":
    main()
