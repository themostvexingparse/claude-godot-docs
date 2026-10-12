#!/usr/bin/env python3
"""Fine-tune the embedding model on Godot-docs retrieval pairs, then export it for godot_embed.py.

Dev-only: needs torch, transformers and onnx on top of the runtime packages (numpy, onnxruntime, tokenizers).

    python train/train_embed.py                # train -> models/gdocs-bge-small/model_int8.onnx
    python train/train_embed.py --epochs 1 --dry-run # just build and report the training set

Training pairs (query -> index entry id):
  - train/queries_*.json: hand-written realistic agent/gamedev questions
  - manual section headings -> their section (generic headings like "Introduction" skipped)
  - Godot 3 names -> their Godot 4 target (from the index's alias table)
Pairs whose query is too similar to any benchmark query (eval/eval_queries.py core + eval/semantic_queries.json) are
dropped, so the benchmark stays held out. Hard negatives are mined with the base model: high-ranking entries
that are not the answer (and not a same-named sibling such as the 3D twin of a 2D member).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "eval")]
import godot_docs as gd  # noqa: E402
import godot_embed as ge  # noqa: E402

BASE = ROOT / "models" / "bge-small-en-v1.5"
OUT = ROOT / "models" / "gdocs-bge-small"
CHECKPOINT = ROOT / "train" / "checkpoint"  # full-precision weights of the last run (not needed at runtime)
STATE = CHECKPOINT / "state.pt"  # resumable training state, saved every --save-every steps
GENERIC_HEADINGS = {
    "introduction", "summary", "example", "examples", "setup", "overview", "conclusion", "usage", "notes", "note",
    "see also", "properties", "methods", "description", "tutorials", "getting started", "requirements", "about",
    "next steps", "troubleshooting", "limitations", "code", "the code", "the script", "result", "results",
}
LEAK_THRESHOLD = 0.85


LOG_FILE = ROOT / "train" / "train.log"


def log(msg: str):
    """stderr plus an unbuffered log file, so progress is visible even when stderr is piped or captured."""
    print(msg, file=sys.stderr, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def handwritten_pairs() -> list[tuple[str, str, str]]:
    pairs = []
    for f in sorted((ROOT / "train").glob("queries_*.json")):
        pairs += [(q, t, "handwritten") for q, t in json.loads(f.read_text(encoding="utf-8"))]
    return pairs


def structural_pairs(ix, rng: random.Random, max_sections: int) -> list[tuple[str, str, str]]:
    pairs = []
    rows = ix.conn.execute("SELECT id, name, body, lang FROM entries WHERE kind = 'section' AND id LIKE 'doc:tutorials/%' "
                           "OR (kind = 'section' AND id LIKE 'doc:getting_started/%')").fetchall()
    rows = [r for r in rows if r["lang"] in ("", "gdscript") and len(r["body"] or "") >= 300
            and (r["name"] or "").strip().lower() not in GENERIC_HEADINGS and len((r["name"] or "").split()) >= 2]
    rng.shuffle(rows)
    pairs += [(r["name"].lower(), r["id"], "heading") for r in rows[:max_sections]]
    for a in ix.aliases():
        if a["target"] and a["kind"] in ("class", "member") and ix.get(a["target"]):
            pairs.append((a["old"], a["target"], "godot3"))
    return pairs


def benchmark_queries() -> list[str]:
    import eval_queries
    return [q for q, _, _ in eval_queries.load_cases("all", None)]


def mine(ix, store, enc, pairs, rng: random.Random, n_neg: int):
    """Returns training examples: (query, positive passage, [hard negative passages])."""
    queries = [q for q, _, _ in pairs]
    qv = enc.encode(queries, query=True)
    first_passage = {}
    for k, n in enumerate(store.pn):
        first_passage.setdefault(int(n), k)
    texts_cache: dict = {}

    def passages(n):
        if n not in texts_cache:
            row = ix.by_rowids([n])[n]
            texts_cache[n] = (row, ge.passage_texts(row))
        return texts_cache[n]

    examples, sims_all = [], store.vecs @ qv.T  # (passages, queries)
    for j, (q, target, src) in enumerate(pairs):
        trow = ix.get(target)
        tn = trow["n"]
        _, tpass = passages(tn)
        # positive: the target's passage closest to the query (sections can have several chunks)
        pidx = np.nonzero(store.pn == tn)[0]
        pos = tpass[int(np.argmax(sims_all[pidx, j]))] if len(pidx) == len(tpass) else tpass[0]
        order = np.argsort(-sims_all[:, j])[:200]
        cands, seen = [], {tn}
        for k in order:
            n = int(store.pn[k])
            if n in seen:
                continue
            seen.add(n)
            row, ptexts = passages(n)
            if row["name"] == trow["name"] or (trow["parent"] and row["id"] == trow["parent"]) \
                    or (row["parent"] and row["parent"] == trow["id"]):
                continue  # same-named sibling or same page: likely also a right answer
            cands.append(ptexts[0])
            if len(cands) >= 30:
                break
        negs = rng.sample(cands[2:], min(n_neg, len(cands) - 2)) if len(cands) > 2 else []
        examples.append((q, pos, negs))
    return examples


def train(examples, args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModel.from_pretrained(BASE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps_per_epoch = (len(examples) + args.batch - 1) // args.batch
    total = steps_per_epoch * args.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, (total - s) / max(1, total - warm)))

    def embed(texts, max_len):
        b = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        out = model(**b).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(out, dim=-1)

    def save_state(step):
        """Resumable state (weights + optimizer + scheduler + step), written atomically."""
        CHECKPOINT.mkdir(parents=True, exist_ok=True)
        tmp = STATE.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "total": total, "n_examples": len(examples), "args": vars(args)}, tmp)
        os.replace(tmp, STATE)

    step = 0
    if args.resume and STATE.is_file():
        st = torch.load(STATE, weights_only=False)
        if st["total"] != total or st["n_examples"] != len(examples):
            raise RuntimeError(f"{STATE} is from a different run setup; delete it or drop --resume")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        log(f"resumed from step {step}/{total}")
    t0, start = time.time(), step
    for epoch in range(args.epochs):
        # fixed per-epoch order so a resumed run sees exactly the batches the interrupted one would have
        order = random.Random(args.seed * 1000 + epoch).sample(range(len(examples)), len(examples))
        for b, s in enumerate(range(0, len(examples), args.batch)):
            if epoch * steps_per_epoch + b < step:
                continue
            batch = [examples[k] for k in order[s:s + args.batch]]
            queries = [ge.QUERY_PREFIX + q for q, _, _ in batch]
            docs = [p for _, p, _ in batch] + [n for _, _, ns in batch for n in ns]
            qe = embed(queries, ge.QUERY_MAX_TOKENS)
            de = embed(docs, args.max_len)
            logits = qe @ de.T / args.temperature  # in-batch negatives + mined hard negatives
            loss = torch.nn.functional.cross_entropy(logits, torch.arange(len(batch)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            if step % 5 == 0 or step == total:
                el = time.time() - t0
                rate = el / max(1, step - start)
                log(f"epoch {epoch + 1} step {step}/{total} loss {loss.item():.4f} "
                    f"({rate:.1f}s/step, eta {rate * (total - step) / 60:.1f} min)")
            if step % args.save_every == 0 and step < total:
                save_state(step)
                log(f"checkpoint saved at step {step}")
        if (epoch + 1) * steps_per_epoch <= start:
            continue  # epoch finished before a --resume: its weights were already saved back then
        # end of epoch: full weights usable by --export-only (and the resumable state)
        model.save_pretrained(CHECKPOINT)
        tok.save_pretrained(CHECKPOINT)
        save_state(step)
        log(f"epoch {epoch + 1} done: weights saved to {CHECKPOINT}")
    model.eval()
    return model, tok


def export(src: Path, out: Path, meta: dict | None = None) -> Path:
    """HF checkpoint -> int8 ONNX + tokenizer.json in `out`. Returns the int8 model path."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModel, AutoTokenizer

    class Encoder(torch.nn.Module):  # keyword call: positional order of BertModel.forward differs across versions
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.m(input_ids=input_ids, attention_mask=attention_mask,
                          token_type_ids=token_type_ids).last_hidden_state

    tok = AutoTokenizer.from_pretrained(src)
    model = Encoder(AutoModel.from_pretrained(src)).eval()
    out.mkdir(parents=True, exist_ok=True)
    fp32 = out / "model_fp32.onnx"
    dummy = tok(["export example", "a longer second example sentence"], padding=True, return_tensors="pt")
    names = ["input_ids", "attention_mask", "token_type_ids"]
    axes = {n: {0: "batch", 1: "seq"} for n in names}
    axes["last_hidden_state"] = {0: "batch", 1: "seq"}
    with torch.no_grad():
        ref = model(dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"])[:, 0]
        torch.onnx.export(model, (dummy["input_ids"], dummy["attention_mask"], dummy["token_type_ids"]), str(fp32),
                          input_names=names, output_names=["last_hidden_state"], dynamic_axes=axes,
                          opset_version=17, dynamo=False)
    int8 = out / "model_int8.onnx"
    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
    fp32.unlink()
    shutil.copy(BASE / "tokenizer.json", out / "tokenizer.json")
    # sanity check: the int8 ONNX model must agree with the torch model
    enc = ge.Encoder(out)
    got = enc.encode(["export example", "a longer second example sentence"])
    ref = torch.nn.functional.normalize(ref, dim=-1).numpy()
    agree = float((got * ref).sum(1).min())
    log(f"exported {int8} ({int8.stat().st_size / 1e6:.0f} MB), int8 vs torch cosine {agree:.4f}")
    if agree < 0.95:
        raise RuntimeError(f"exported model disagrees with torch model (cosine {agree:.3f})")
    if meta is not None:
        (out / "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return int8


def ensure_base() -> str | None:
    """The base model isn't shipped (search only needs models/gdocs-bge-small). Returns an error message if it is
    missing; otherwise makes sure its int8 ONNX export (used to mine hard negatives) exists."""
    if not (BASE / "config.json").is_file():
        return (f"Base model not found in {BASE}. Download it first (about 130 MB):\n"
                f"  pip install huggingface_hub\n"
                f"  huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir {BASE}")
    if not (BASE / "model_int8.onnx").is_file():
        tmp = ROOT / "train" / "base_export"
        shutil.move(str(export(BASE, tmp)), str(BASE / "model_int8.onnx"))
        shutil.rmtree(tmp)
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=0.02)
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--negatives", type=int, default=1, help="mined hard negatives per query")
    ap.add_argument("--max-sections", type=int, default=1500, help="cap on heading->section pairs")
    ap.add_argument("--handwritten-repeat", type=int, default=2, help="oversample the hand-written pairs")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--save-every", type=int, default=20, help="steps between resumable checkpoints (~3 min)")
    ap.add_argument("--max-examples", type=int, default=0, help="truncate the training set (smoke tests)")
    ap.add_argument("--resume", action="store_true", help=f"continue from the last checkpoint state")
    ap.add_argument("--check-export", action="store_true", help="export the base model to a temp folder and verify")
    ap.add_argument("--export-only", action="store_true", help="re-export the last saved checkpoint")
    ap.add_argument("--snapshot", action="store_true",
                    help="export the latest mid-run state.pt to models/gdocs-bge-small-step<N> (safe while training runs)")
    args = ap.parse_args()
    err = ensure_base()
    if err:
        print(err, file=sys.stderr)
        return 1
    if args.snapshot:
        import torch
        from transformers import AutoModel, AutoTokenizer
        st = torch.load(STATE, weights_only=False, map_location="cpu")
        snap = ROOT / "train" / f"snapshot_step{st['step']}"
        model = AutoModel.from_pretrained(BASE)
        model.load_state_dict(st["model"])
        model.save_pretrained(snap)
        AutoTokenizer.from_pretrained(BASE).save_pretrained(snap)
        export(snap, ROOT / "models" / f"gdocs-bge-small-step{st['step']}",
               {"snapshot_of_step": st["step"], "total_steps": st["total"]})
        shutil.rmtree(snap)
        return 0
    if args.check_export:
        tmp = ROOT / "train" / "export_check"
        export(BASE, tmp)
        shutil.rmtree(tmp)
        return 0
    if args.export_only:
        mf = CHECKPOINT / "train_meta.json"
        meta = json.loads(mf.read_text(encoding="utf-8")) if mf.is_file() else {"note": "exported from checkpoint"}
        export(CHECKPOINT, OUT, meta)
        return 0

    rng = random.Random(args.seed)
    ix = gd.open_index(argparse.Namespace(docs=None, index=None))
    store = ge.store_for(ix, BASE.name, build_limit=None)
    enc = store.encoder
    pairs = handwritten_pairs() + structural_pairs(ix, rng, args.max_sections)

    bench = benchmark_queries()
    bv = enc.encode(bench, query=True)
    pv = enc.encode([q for q, _, _ in pairs], query=True)
    sim = (pv @ bv.T).max(axis=1)
    kept = [p for p, s in zip(pairs, sim) if s < LEAK_THRESHOLD]
    dropped = [(p[0], bench[int(np.argmax(pv[k] @ bv.T))]) for k, (p, s) in enumerate(zip(pairs, sim)) if s >= LEAK_THRESHOLD]
    by_src = {}
    for _, _, src in kept:
        by_src[src] = by_src.get(src, 0) + 1
    log(f"pairs: {len(pairs)} total, {len(dropped)} dropped as too close to a benchmark query, kept {by_src}")
    for q, b in dropped[:15]:
        log(f"  dropped {q!r}  ~  {b!r}")
    if args.dry_run:
        return 0

    t0 = time.time()
    examples = mine(ix, store, enc, kept, rng, args.negatives)
    hand = [e for e, p in zip(examples, kept) if p[2] == "handwritten"]
    examples += hand * (args.handwritten_repeat - 1)
    if args.max_examples:  # smoke tests
        examples = examples[:args.max_examples]
    log(f"mined hard negatives in {time.time() - t0:.0f}s; {len(examples)} training examples")
    meta = {"base": BASE.name, "pairs": by_src, "dropped_for_leakage": len(dropped), "examples": len(examples),
            "args": vars(args), "started_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    (CHECKPOINT / "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    train(examples, args)
    meta["trained_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (CHECKPOINT / "train_meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    export(CHECKPOINT, OUT, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
