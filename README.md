# claude-godot-docs — offline Godot docs search for AI agents

A single-file, standard-library-only Python tool (3.9+) that indexes the official
[godot-docs](https://github.com/godotengine/godot-docs) sources (class reference + manual)
into a local SQLite FTS5 database and answers queries with compact, agent-friendly text.

- No dependencies, no network, no API keys.
- The index (`data/godot_docs_index.db`, ~60 MB) is all it needs at query time. It ships prebuilt from the Godot 4.5
  docs (commit `33e08e1`), so the docs checkout itself is optional.
- With a docs checkout available, it rebuilds itself (~5-10 s) when the docs or the tool change. Without one, or
  if a rebuild fails, it keeps the existing index and prints a single warning instead of an error.
- Every result carries an ID that can be passed to `read`.
- Agents can add their own guess in Godot terms with `--also`. It is merged with the plain question and
  closes most of the wording gap (held-out MRR 0.34 -> 0.86).
- Knows Godot 3 → 4 renames (`KinematicBody2D`, `yield`, `onready`, `rand_range`, ...) and says so.
- Flags deprecated APIs, including every member of a deprecated class (`TileMap.set_cell` -> `TileMapLayer`):
  `DEPRECATED:` line in `read` and `members`, `[deprecated]` in `lookup` and search results, ranked below
  their replacements. Worked out at query time (`Index.flags_of`), so the index doesn't need rebuilding.

## Layout

```
gdocs, gdocs.cmd        shortcuts (sh / Windows) for src/godot_docs.py
src/                    the tool: godot_docs.py (CLI, stdlib only), godot_embed.py (optional hybrid search),
                        godot_pretty.py (--pretty output)
data/                   godot_docs_index.db (prebuilt index), concepts.json (concept glossary);
                        embedding vectors are built here (not committed)
models/gdocs-bge-small/ fine-tuned embedding model (int8 ONNX) for semantic/hybrid search
skills/godot-docs/      Claude Code skill
tests/                  unit + smoke tests
eval/                   search-quality benchmark (eval_queries.py), query sets, pooled judging (pool.py, judgments/)
train/                  embedding fine-tuning (train_embed.py), ranking/confidence fitting, sweeps; results/ logs
```

## Setup

The repository includes the built index (`data/godot_docs_index.db`), so lexical search (the default) works
right away: `gdocs index` prints the index status. For semantic/hybrid search, install the packages and build
the embedding vectors once (see [Semantic / hybrid search](#semantic--hybrid-search-optional)):
```sh
pip install numpy onnxruntime tokenizers
gdocs index --vectors                                             # about 3-4 minutes on CPU, resumable
```

**Rebuilding from the docs** (only needed after changing the RST conversion code, or for a newer Godot version):
```sh
git clone --depth 1 --branch 4.5 https://github.com/godotengine/godot-docs.git
python src/godot_docs.py index --rebuild --docs path/to/godot-docs    # or set GODOT_DOCS_PATH
python src/godot_docs.py index --vectors                              # re-embed changed passages (seconds to minutes)
```
The tool also finds a checkout automatically when this folder sits inside it (`godot-docs/<this folder>/`) or
next to it (`<this folder>/godot-docs/`).

If no checkout is found, or a rebuild fails, the existing index stays in use and one warning is printed on stderr
(`[godot_docs] WARNING: ...`). It is repeated only after the tool or the docs change again; the state is kept in
`data/godot_docs_index.db.warned`. A failed rebuild is not retried on every call. `index --rebuild` always retries and
reports. Only a missing index with no docs to build it from is an error.

## Shortcut: `gdocs`

`gdocs <command> ...` is the same as `python src/godot_docs.py <command> ...`:
- `gdocs.cmd` for PowerShell and cmd.
- `gdocs` (shell script) for Git Bash, macOS and Linux.

Add the repository folder to your PATH to use it anywhere:
- **Windows (PowerShell):**
  `[Environment]::SetEnvironmentVariable('Path', [Environment]::GetEnvironmentVariable('Path','User') + ';<this folder>', 'User')`.
  Then open a new terminal.
- **macOS/Linux:** `export PATH="$PATH:/path/to/gdocs"` in your shell profile.

## Claude Code skill

`skills/godot-docs/SKILL.md` teaches Claude when and how to use the tool. Install it per user or per project:
- **Per user:** copy the `godot-docs` folder to `~/.claude/skills/`.
- **Per project:** copy it to `<game repo>/.claude/skills/` so it's committed for the whole team.

If `gdocs` isn't on PATH, update the fallback path (`<repo>/src/godot_docs.py`) in the skill's "Running it" section.

## Commands

| Command | Purpose |
|---|---|
| `search "<query>" [--also "<phrasing>"]... [-k KIND] [-c CLASS] [-p PATH] [-n N] [-m MODE]` | Ranked search over API + manual. `--also` adds phrasings of the same question (see below). Kinds: `class method property signal constant enum annotation theme_item page section`, or groups `api`, `member`, `docs`. Modes: `lexical` (default), `semantic`, `hybrid` (see below). |
| `read <id> [--full] [--offset N] [--max-chars N]` | Full text of one entry. Classes show description + member signatures (`--full` adds every member's docs). Pages list their sections. |
| `lookup <symbol>` | Every class defining a name, Godot 3 aliases, typo suggestions. |
| `members <Class> [--inherited] [-k KIND] [-f TEXT]` | Member list with one-line summaries. |
| `hierarchy <Class> [--depth N] [--brief]` | Ancestors and subclass tree. |
| `examples "<query>" [-c CLASS] [-n N]` | Search code blocks only. |
| `toc [folder or page id]` | Browse the manual tree / a page's sections. |
| `index [--rebuild] [--vectors] [--model M]` | Index status or forced rebuild; `--vectors` builds the embedding vectors. |

`search` and `read` also take `--pretty`. It prints structured, colorful output for people in a terminal:
- **search:** a header box (query, version, mode, result count), then one table of results with columns
  #, Match tier, Kind, Entry ID and Details (signature and summary, or breadcrumb and snippet).
- **read:** an info box (kind, signature, class, inheritance, source), then:
  - each member group of a class page as a table (Properties: Name / Type / Default;
    Methods: Method / Arguments / Returns / Notes; Signals; Constants; Enumerations);
  - the manual's own tables as box tables;
  - boxed, GDScript-highlighted code blocks;
  - no 10k-character cutoff unless `--max-chars` is given.

Everything wraps to the terminal width (70 to 140 columns). The default output stays the compact format meant
for AI agents. Set `NO_COLOR=1` to keep the layout without colors.

Common flags: `--json` (machine-readable), `--lang gdscript|csharp|all` (which tab of GDScript/C#
code pairs to show; default `gdscript`, or env `GODOT_DOCS_LANG`).

IDs: classes `Node2D` · members `Node2D.look_at` · pages `doc:tutorials/2d/2d_movement` ·
sections `doc:tutorials/2d/2d_movement#8-way-movement`. `read` also accepts loose forms:
`look_at` (if unique), `CharacterBody2D.position` (inherited → `Node2D.position`), `doc_signals`
(RST label), `singletons_autoload` (page basename), Godot 3 names.

Exit codes: `0` found · `1` not found/ambiguous (with suggestions) · `2` usage error.
Long output is cut at 10,000 characters with a `--offset` continuation hint.

## Type hints for everyday words

Words like "enemy", "player", "sprite", "pickup", "bullet" or "wall", and "object" next to a physics word, map
to the node types they usually mean (`TYPE_HINTS` in `src/godot_docs.py`): `CharacterBody2D/3D`, `Sprite2D`,
`Area2D/3D`, `StaticBody2D/3D`, `RigidBody2D/3D`. "2D" or "3D" in the query keeps only that dimension. The hinted
classes are added to the keyword search, and their members get a small ranking boost. API-name queries are left
alone. Env `GODOT_DOCS_TYPE_HINTS=off|members|hint|syn` switches the variant (default `members`).

Benchmark (2026-10-07):
- Dev and val splits: small gains in hybrid mode (dev MRR 0.618 -> 0.633); neutral in lexical mode.
- Held-out test split: neutral (hybrid 54/100 both ways; MRR 0.419 -> 0.417).

A generic "node" expansion (any thing-word -> "node") was tried first and dropped: "node" is in too much of the
docs to help tell results apart.

## Concept glossary (`data/concepts.json`)

Questions often use words the docs never use: other engines' names ("prefab", "OnTriggerEnter", "WaitForSeconds",
"Time.deltaTime") or gameplay phrasing for math and physics ("how far away is the enemy", "invisible line to see
what the player is looking at"). `concepts.json` maps such phrases to the Godot entries they mean (73 concepts,
~350 phrases). Each phrase acts as extra text of its targets:
- lexical: like keywords. A phrase counts only when it matches at least two of the query's words, or all of a
  one-word phrase whose word the docs barely use ("prefab", "coroutine"). A full phrase match gives its targets
  a bonus (`CONCEPT_WEIGHT`), smaller further down the target list (`CONCEPT_DECAY`).
- semantic/hybrid: an extra embedded passage of each target (a short question-like phrase is much easier for the
  small model to match than reference text). Changing the file re-embeds only new phrases (a few seconds).
- A matched concept with a `note` prints it as a NOTE line, e.g. "Godot has no prefabs: a saved scene
  (PackedScene) plays that role ...".

Editing: keep phrases specific (no lone generic words), check targets exist (`read <id>`), and never copy
benchmark questions into it. `GODOT_DOCS_CONCEPTS=<file>` points to another glossary (a missing file disables it).

Benchmark (2026-10-08, after an external review where 5 such questions all missed): new set
`eval/concept_queries.json` (report 5 / dev 33 / test 33). Weights tuned on dev (`train/sweep_concepts.py`):
- Held-out concept test: hybrid 11/33 -> 25/33 (MRR 0.245 -> 0.735), lexical 5 -> 17, semantic 10 -> 25.
- Older semantic test split (written before the glossary): hybrid MRR 0.417 -> 0.487, lexical 0.340 -> 0.357;
  with `--also` unchanged (0.869 / 0.861). Val: hybrid 53/75 -> 59/75. Report questions: 5/5 (lexical and hybrid).
- Caveat: the glossary and the concept set were written by the same model family; the older semantic split is the
  more independent check.

## Multi-phrasing search (`--also`)

`search "<question>" --also "<Godot terms>"` searches each phrasing separately and merges the lists. The calling
agent usually knows Godot's vocabulary, so its guess ("Engine time_scale", "Decal", "one_way_collision")
bridges the wording gap that causes most misses.

Merging (`MULTI_FUSION = "conf_or"`): each result's calibrated confidence from each phrasing's list is combined as
P(at least one is right), with reciprocal rank as a tie-break. Duplicates collapse to one result, the
3-per-class cap is applied after merging, and the header echoes every phrasing.

Benchmark: `eval/also_phrasings.json` holds one Godot-terms phrasing per benchmark question, written from the
question text only (labels not consulted). `eval/eval_queries.py --also` uses them; `train/sweep_also.py` compares
merge methods on dev/val (`train/results/sweep_also.log`); `train/results/also_test.log` is the single test-split scoring.

| MRR (dev / val / test) | plain question | Godot terms only | `--also` (conf_or) |
|---|---|---|---|
| lexical | 0.517 / 0.447 / 0.340 | 0.890 / 0.898 / – | 0.879 / 0.904 / 0.856 |
| hybrid  | 0.633 / 0.599 / 0.417 | 0.916 / 0.940 / – | 0.880 / 0.933 / 0.870 |

Other merges on dev/val (hybrid): RRF k=10 0.895 / 0.866, rel_sum 0.847 / 0.877, rel_max 0.839 / 0.833.

Caveat: the phrasings were written by the same model family that wrote and judged the benchmark, with full Godot
knowledge, so these numbers are what a well-informed agent gets. A vague or wrong guess helps less; the
plain question keeps those cases from getting worse than a normal search. Real `GODOT_DOCS_QUERY_LOG` data
(which records `also`) is the honest check.

## Learned ranking (experiment, off)

`train/fit_ranker.py` fits ranking weights from judged benchmark queries. It uses a pairwise logistic model over
the signals `run_search` records for each candidate (keyword score, concept coverage, title coverage, embedding
similarity, kind, hint flags). Exact-name and Godot 3 alias rules stay fixed on top. Env `GODOT_DOCS_RANKER=learned`
switches it on once `RANK_MODELS` is filled.

Result (2026-10-07, 2-fold cross-validation on dev/val): it does not beat the hand-tuned formula.
- Replacing the formula: hybrid MRR 0.590 vs 0.616, lexical 0.439 vs 0.482, and the core set drops.
- Learning corrections around the formula: the more freedom the corrections get, the worse they do on the
  other fold (hybrid 0.621 -> 0.610 -> 0.600 -> 0.296), and the best fit is the formula itself.

About 75 judged questions per fold aren't enough to learn more than the formula already encodes. Re-run the
script once real agent queries (`GODOT_DOCS_QUERY_LOG`) have been judged, at a few hundred or more.

## Semantic / hybrid search (optional)

`--mode hybrid` mixes the keyword ranking with sentence-embedding similarity, which finds answers worded
differently from the query ("stop my character sliding down hills" -> `floor_stop_on_slope`). `--mode semantic`
uses embeddings only. The default stays `lexical`: it needs nothing beyond the standard library, and the embedding
code (`godot_embed.py`, numpy, onnxruntime, tokenizers) is imported only when a search asks for another mode.
Env `GODOT_DOCS_MODE` changes the default.

- **Packages:** `pip install numpy onnxruntime tokenizers`.
- **Model:** `models/gdocs-bge-small/` (bge-small-en-v1.5, 33M parameters, fine-tuned on Godot-docs queries,
  int8 ONNX, 33 MB), self-contained, no download at runtime. `--model <folder>` or env `GODOT_DOCS_MODEL` picks
  another model folder (one holding `tokenizer.json` + `model_int8.onnx`).
- **Vectors:** `data/godot_docs_index_vectors.<model>.npz` next to the index (a cache, not committed) (about 31k passages: one per API member,
  up to 6 chunks per manual section). Build them once with `gdocs index --vectors`, which takes about 3-4
  minutes on an 8-thread CPU, shows progress, and resumes if interrupted. Later doc or tool changes re-embed
  only the passages whose text changed. A search never blocks on a large build: without vectors it prints a
  NOTE and returns lexical results.
- **Speed:** a cold CLI hybrid query takes about 0.5 s (lexical: 0.15 s), with or without `--also`;
  in-process about 40 ms (75 ms with one `--also`).
- **Results** (new held-out `test` split, 100 questions, pooled labels): lexical 43/100 (MRR 0.345); hybrid with
  the stock model 45/100 (0.364); fine-tuned semantic 54/100 (0.404); fine-tuned hybrid 54/100 (MRR 0.419,
  recall@10 0.66). Core set: lexical 58/58 (MRR 0.859), fine-tuned hybrid 57/58 (MRR 0.913).
  With a Godot-terms `--also` guess both modes reach about 0.86-0.87 (see above), so hybrid matters most
  when the caller has no good guess.

## Quality checks

```sh
python -m unittest discover tests -v                    # converter unit tests + index smoke tests
python eval/eval_queries.py -v                          # core benchmark (58 queries used to tune the lexical ranker)
python eval/eval_queries.py --set semantic --split dev  # paraphrased queries, dev split (tune on this)
python eval/eval_queries.py --set semantic --split test --mode hybrid   # held-out split: report only, never tune on it
python eval/eval_queries.py --set semantic --split dev --also           # with the --also phrasings
python eval/eval_queries.py --set concept --split dev --mode hybrid     # other engines' terms / gameplay phrasing
python -u train/sweep_also.py > train/results/sweep_also.log            # compare --also merge methods on dev/val (~4 min)
```

`eval/semantic_queries.json` has 250 paraphrased questions in three splits:
- `dev` (75): for tuning.
- `val` (75): the former test split. It was scored during development, so treat it as seen.
- `test` (100): written 2026-10-07 after training, checked against the training queries for near-duplicates
  (cosine >= 0.85 with the same answer -> replaced), never used for tuning.

Labels are prefixes in `expected` plus exact ids in `accept`. `accept` comes from pooled judging (`eval/pool.py`):
the top 5 of every system are merged, shuffled and judged against a written rubric (see `pool.py`), so the labels
don't favour any one system. Judgments are kept in `eval/judgments/`. `train/train_embed.py` drops any training
query too similar to a benchmark query.

`eval/pool.py` and `train/final_bench.py` also compare against the stock `bge-small-en-v1.5` model, which is no
longer shipped (removed 2026-10-08; their results are recorded above and in `train/results/final_bench.log`). Download
it as described under fine-tuning before re-running them.

Opt-in query logging for future benchmark material: set `GODOT_DOCS_QUERY_LOG=<file>` and every `search` appends a
JSON line (query, `--also` phrasings, mode, filters, result ids).

## Fine-tuning the embedding model (dev only)

`python train/train_embed.py` fine-tunes bge-small on `train/queries_*.json` (hand-written question -> entry pairs),
manual headings -> sections and Godot 3 -> 4 renames, with hard negatives mined by the base model, then exports
`models/gdocs-bge-small/model_int8.onnx`. It needs `torch transformers onnx` and about 35 minutes on CPU.
Progress goes to `train/train.log`. Afterwards run `gdocs index --vectors` and the benchmarks.

The base model isn't shipped. Download it once before training (about 130 MB); `train/train_embed.py` says so if it's
missing, and creates the base model's int8 export (used to mine hard negatives) on first run:
```sh
pip install huggingface_hub
huggingface-cli download BAAI/bge-small-en-v1.5 --local-dir models/bge-small-en-v1.5
```

The `[strong|partial|weak]` tags in search output come from small logistic models (`CONF_MODELS` in
`src/godot_docs.py`) fitted on the dev and val splits by `python train/fit_confidence.py`. That script prints the
precision of each tier on the test split. Re-run it after ranking changes.

## Agent instructions (paste into CLAUDE.md / AGENTS.md of the game repo)

Adjust the path to where this repository lives.

```markdown
## Godot documentation — check before writing GDScript

This project uses **Godot 4.5**. Your training data mixes Godot 3 and Godot 4 APIs, so verify
any class, method, property, signal or annotation against the local docs before using it:

    python tools/gdocs/src/godot_docs.py <command> ...

- `search "how to make the camera follow the player" --also "Camera2D position_smoothing"`: how-to
  questions; always add `--also` with the Godot class/method names you think are involved
- `read <id>`: full docs for an ID from a result (e.g. `read CharacterBody2D.move_and_slide`)
- `lookup <name>`: which classes define a name; flags Godot 3 names
- `members <Class> --inherited -f <text>`: does this class (or a parent) have X?
- `examples "<query>"`: GDScript code samples from the docs

Rules:
1. If you are not sure an API exists in Godot 4.5 with that exact signature, run `read` or `lookup` first.
2. Any `NOTE Godot 3 -> 4` line in the output means you were about to use a Godot 3 API. Use the replacement it names.
3. For "how do I..." questions, search with `--also "<Godot names you expect>"`, then read the manual
   section that `search` returns before writing code. "No strong match" means: change the `--also` guess.
4. Output stops at 10k characters. Continue with the `--offset N` shown at the end.
5. Never pass `--pretty`: it is reserved for human use. The default output is the agent format (`--json` for raw data).
```
