"""Tune the concept-glossary weights (CONCEPT_WEIGHT, CONCEPT_DECAY) and HYBRID_WEIGHT on dev splits only.
Rule: core must stay 58/58 lexical (and >= 57 hybrid); then most semantic-dev + concept-dev passes, then MRR."""
import argparse
import itertools
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
import eval_queries as ev  # noqa: E402
import godot_docs as gd  # noqa: E402

ix = gd.open_index(argparse.Namespace(docs=None, index=None))
core, sdev, cdev = ev.load_cases("core", None), ev.load_cases("semantic", "dev"), ev.load_cases("concept", "dev")
q = dict(quiet=True)
rows = []
for cw, dec in itertools.product((0.0, 0.15, 0.3, 0.5, 0.8), (0.1, 0.2)):
    if cw == 0.0 and dec == 0.2:
        continue
    gd.CONCEPT_WEIGHT, gd.CONCEPT_DECAY = cw, dec
    lc, lcd = ev.evaluate(ix, core, "lexical", **q), ev.evaluate(ix, cdev, "lexical", **q)
    for hw in (0.4, 0.6, 0.8, 1.0):
        gd.HYBRID_WEIGHT = hw
        hc, hs, hcd = (ev.evaluate(ix, c, "hybrid", **q) for c in (core, sdev, cdev))
        rows.append((cw, dec, hw, lc, lcd, hc, hs, hcd))
        print(f"cw={cw} dec={dec} hw={hw} | lex core {lc['passed']} cdev {lcd['passed']} {lcd['mrr']:.3f} | "
              f"hyb core {hc['passed']} sdev {hs['passed']} {hs['mrr']:.3f} cdev {hcd['passed']} {hcd['mrr']:.3f}", flush=True)
ok = [r for r in rows if r[3]["passed"] == 58 and r[5]["passed"] >= 57]
best = max(ok, key=lambda r: (r[6]["passed"] + r[7]["passed"], r[6]["mrr"] + r[7]["mrr"]))
print("BEST cw=%s dec=%s hw=%s" % best[:3])
