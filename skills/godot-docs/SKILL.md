---
name: godot-docs
description: Search and read the official Godot 4.5 documentation offline (class reference + manual) with the `gdocs` CLI. Use whenever writing, reviewing or debugging GDScript/Godot code, when unsure whether a class, method, property, signal, annotation or constant exists or what its exact signature is, when answering "how do I ... in Godot" questions, and when code may contain Godot 3 APIs (KinematicBody2D, yield, onready, export var, rand_range, instance(), setget, ...) or deprecated ones (TileMap, ParallaxBackground).
---

# Godot docs search (`gdocs`)

The project targets **Godot 4.5**. Your training data mixes Godot 3 and Godot 4 APIs, and many Godot 3
names still look plausible. `gdocs` answers from a local index of the official 4.5 docs: the full class
reference (about 1,000 classes, 23,000 members) and the manual (about 490 pages, split into sections).
Treat its output as the source of truth over your memory.

**Run it:** `gdocs <command> ...`. If `gdocs` isn't on PATH, use
`python C:/Users/ozano/Desktop/gdocs/src/godot_docs.py <command> ...` (or find `godot_docs.py`). Python 3.9+,
no packages, no network, under half a second per call. A stderr line starting `[godot_docs] WARNING:`
means it is using its existing index; results are valid, so carry on.

## Find the right entry: pick the command by what you already know

| You know... | Run | Example |
|---|---|---|
| the exact `Class.member` | `read <id>` | `gdocs read CharacterBody2D.move_and_slide` |
| a name, but not its class | `lookup <name>` | `gdocs lookup look_at` |
| the class, but not the member's name | `members <Class> --inherited -f <word>` | `gdocs members Sprite2D --inherited -f flip` |
| only what you want to do | `search "<question>" --also "<likely names>"` | `gdocs search "hide the mouse pointer" --also "Input mouse_mode"` |
| a line of possibly old code | `search "<the line>"` | `gdocs search "yield(get_tree().create_timer(1), 'timeout')"` |
| an engine error about an unknown name | `lookup <name>` | `gdocs lookup KinematicBody2D` |

Then **`read` every API you will call** before writing code, to confirm the exact name, arguments and
return type. Only use names you have confirmed.

### `--also`: always add your guess in Godot terms to how-to searches

Put the plain question first and the classes, methods or properties you think are involved in `--also`:
```sh
gdocs search "bullet hole on a wall" --also "Decal"
gdocs search "jump through platforms from below" --also "one_way_collision"
gdocs search "camera shake when hit" --also "Camera2D offset" --also "Tween"
```
- Most misses are wording gaps between the question and the docs; your guess bridges them. A wrong guess
  costs little, because the plain question is still searched and the lists are merged.
- Keep each guess to a few names; don't reword the question. Pass `--also` again for a second guess.
- On held-out questions this found a right answer in the top 10 for 100 of 100, against 51 without it.
- Skip it for pure identifier queries (`move_and_slide`) and pasted code.

## Commands

```sh
gdocs search "<query>" [--also "<guess>"]... [-k KIND] [-c CLASS] [-p PATH] [-n N] [-m MODE]
gdocs read <id> [--full] [--offset N]       # one entry; --full adds every member's description to a class
gdocs lookup <name>                         # every class defining a name, Godot 3 renames, typo fixes
gdocs members <Class> [--inherited] [-f TEXT] [-k KIND] [--no-summary]
gdocs hierarchy <Class> [--depth N] [--brief]
gdocs examples "<query>" [-c CLASS] [-n N]  # GDScript code blocks only, each with its source ID
gdocs toc [folder | page id]                # browse the manual, e.g. toc tutorials/physics
```
Search filters:
- `-k/--kind`: `class method property signal constant enum annotation page section`, or the groups `api`,
  `member` and `docs` (manual pages and sections).
- `-c/--class X`: only X and its members. This also lifts the 3-results-per-class cap.
- `-p/--path tutorials/2d`: only manual pages under that folder.

`--mode hybrid` adds sentence-embedding matching (about 0.5 s instead of 0.15 s). Use it when you can't make
a confident `--also` guess; with a good guess both modes score about the same. If it prints
`NOTE Embedding vectors not built`, the results are plain keyword results.

**`--pretty` is reserved for human use. Agents must never pass it** (to any command: `search`, `read`, ...).
It prints colored, full-length output meant for a person in a terminal, which wastes your context and is harder
to parse. The default output is the agent format; use `--json` if you need machine-readable output.

## IDs

| Entry | ID | Example |
|---|---|---|
| Class | `Class` | `CharacterBody2D`, `@GlobalScope`, `@GDScript` |
| Member | `Class.name` | `Node2D.look_at`, `Node.ready` (signal), `@GDScript.@export` |
| Enum / constant | `Class.Enum`, `Class.CONSTANT` | `Node.ProcessMode`, `Node.PROCESS_MODE_ALWAYS` |
| Manual page / section | `doc:<path>` / `doc:<path>#<slug>` | `doc:tutorials/2d/2d_movement#8-way-movement` |

- `read` is forgiving. Trailing `()` is fine. `CharacterBody2D.position` resolves to the inherited
  `Node2D.position`. Godot 3 names (`onready`, `KinematicBody2D`) resolve to their replacement with a `NOTE`.
- A name defined by several classes (`look_at`, `get_tree`) lists the candidate IDs; pick one and `read` it.
  For 2D/3D twins, pick the one matching the project.
- Global functions (`lerp`, `randf_range`, `print`) live in `@GlobalScope`; annotations (`@export`,
  `@onready`) and `preload`/`load` in `@GDScript`.
- Quote IDs that contain `#` or `@` in the shell.

## Reading the output

- **`NOTE Godot 3 -> 4: ...`**: you used a Godot 3 name or idiom. Switch to the replacement it names.
- **Other `NOTE` lines** (e.g. "Godot has no prefabs: a saved scene (PackedScene) plays that role ..."): your
  question used another engine's term or a gameplay phrase; the note gives the Godot idiom. Unity/Unreal words
  are fine to search with.
- **`[deprecated]`** / **`DEPRECATED: ...`**: don't use it in new code. This includes every member of a
  deprecated class (`TileMap.set_cell` -> use `TileMapLayer.set_cell`). The note names the replacement.
- **`[strong]` / `[partial]` / `[weak]`**: how likely a result answers the query (about 1 in 2, 1 in 5 and
  1 in 10). Read a result before relying on it, especially a weak one.
- **"No strong match"**: the docs use other words. Change the `--also` guess or try `--mode hybrid`.
- **`(+N more in X: --class X)`**: results are capped at 3 per class or page. Follow the hint for the rest.
- **`Read: read <id>`** at the end is the suggested next command.
- **Signatures**: `name(arg: Type = default) -> Return [const, virtual, static, vararg]`. `virtual` means
  you override it (`_ready`, `_physics_process`); `static` means you call it on the class.
- **Long output** stops at 10,000 characters with `Continue with: read <id> --offset N`. Prefer reading one
  manual section at a time over whole pages.
- **`[outdated]`** on a manual page: cross-check it against the class reference.
- **Exit codes**: `0` found, `1` nothing found or ambiguous (suggestions printed), `2` bad usage.

## Limits

- It knows only the official docs: no community answers. If nothing relevant comes up after two or three
  `--also` guesses and both modes, say so; don't guess an API.
- It covers Godot 4.5 only. APIs added in 4.6+ are absent on purpose.
