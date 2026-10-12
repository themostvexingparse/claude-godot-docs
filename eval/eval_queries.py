#!/usr/bin/env python3
"""Search-quality benchmark: runs realistic agent queries and checks the expected entry ranks high.

Usage: python eval/eval_queries.py [-v] [--mode lexical|semantic|hybrid] [--set core|semantic|all] [--split dev|test]
Each case: (query, list of acceptable id prefixes, k). A case passes if any result in the top k
starts with one of the prefixes.

Sets: `core` = the cases below (used while tuning the lexical ranker); `semantic` = eval/semantic_queries.json,
150 paraphrased queries held out from all training data, split into dev (for tuning) and test (report only);
`concept` = eval/concept_queries.json, "concept gap" questions (other engines' terms, gameplay words for math,
unnamed features): report (the 5 from the 2026-10-08 external review), dev (tuning), test (held out).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import godot_docs as gd  # noqa: E402

CASES = [
    # Natural-language / conceptual
    ("move character with keyboard", ["doc:tutorials/2d/2d_movement", "doc:getting_started/first_2d_game/03.coding_the_player"], 5),
    ("how to use signals", ["doc:getting_started/step_by_step/signals"], 3),
    ("wait one second before doing something", ["SceneTree.create_timer", "doc:tutorials/scripting/gdscript/gdscript_basics#awaiting", "Timer"], 5),
    ("change scene", ["SceneTree.change_scene_to_file", "doc:tutorials/scripting/change_scenes_manually", "SceneTree.change_scene_to_packed"], 5),
    ("save game data to file", ["doc:tutorials/io/saving_games", "FileAccess"], 5),
    ("play a sound effect", ["AudioStreamPlayer", "doc:tutorials/audio"], 5),
    ("detect collision between two areas", ["Area2D.area_entered", "Area2D", "doc:tutorials/physics/using_area_2d", "Area2D.body_entered"], 5),
    ("spawn enemies at random positions", ["doc:getting_started/first_2d_game/05.the_main_game_scene", "PackedScene.instantiate", "doc:tutorials/scripting/nodes_and_scene_instances"], 5),
    ("global variables accessible from any script", ["doc:tutorials/scripting/singletons_autoload"], 3),
    ("pause the game", ["doc:tutorials/scripting/pausing_games", "SceneTree.paused"], 3),
    ("make camera follow the player", ["Camera2D", "doc:getting_started/first_3d_game", "doc:tutorials/2d/2d_movement"], 5),
    ("tween a property smoothly", ["Tween.tween_property", "Tween", "Node.create_tween"], 3),
    ("export a variable to the inspector", ["@GDScript.@export", "doc:tutorials/scripting/gdscript/gdscript_exports"], 3),
    ("static typing in gdscript", ["doc:tutorials/scripting/gdscript/static_typing"], 3),
    ("jump and gravity platformer", ["CharacterBody2D", "doc:tutorials/physics", "doc:getting_started/first_3d_game", "doc:tutorials/2d/2d_movement"], 5),
    ("raycast from code", ["doc:tutorials/physics/ray-casting", "PhysicsDirectSpaceState2D.intersect_ray", "PhysicsDirectSpaceState3D.intersect_ray", "RayCast2D"], 5),
    ("export game for web browser", ["doc:tutorials/export/exporting_for_web"], 3),
    ("animate a sprite sheet", ["AnimatedSprite2D", "doc:tutorials/2d/2d_sprite_animation", "SpriteFrames", "Sprite2D.hframes"], 5),
    ("read json file", ["JSON", "JSON.parse_string", "FileAccess", "doc:tutorials/io"], 5),
    ("random number between two values", ["@GlobalScope.randf_range", "@GlobalScope.randi_range", "RandomNumberGenerator", "doc:tutorials/math/random_number_generation"], 3),
    ("pathfinding for enemies", ["NavigationAgent2D", "NavigationAgent3D", "doc:tutorials/navigation", "AStar2D", "AStarGrid2D"], 5),
    ("health bar UI", ["ProgressBar", "TextureProgressBar", "doc:getting_started/first_2d_game/06.heads_up_display"], 5),
    ("connect a button press to a function", ["BaseButton.pressed", "doc:getting_started/step_by_step/signals", "Button"], 5),
    ("delta time physics process", ["Node._physics_process", "doc:tutorials/scripting/idle_and_physics_processing"], 3),
    ("get mouse position", ["Viewport.get_mouse_position", "CanvasItem.get_global_mouse_position", "CanvasItem.get_local_mouse_position"], 3),
    ("load a scene and add it as child", ["PackedScene.instantiate", "Node.add_child", "doc:tutorials/scripting/nodes_and_scene_instances", "@GDScript.preload", "@GDScript.load"], 5),
    ("2d lights and shadows", ["doc:tutorials/2d/2d_lights_and_shadows"], 3),
    ("particles explosion effect", ["GPUParticles2D", "CPUParticles2D", "doc:tutorials/2d/particle_systems_2d", "doc:tutorials/3d/particles"], 5),
    ("multiplayer high level networking", ["doc:tutorials/networking/high_level_multiplayer"], 3),
    ("gamepad input", ["doc:tutorials/inputs/controllers_gamepads_joysticks", "Input.get_joy", "InputEventJoypad"], 5),
    ("get all nodes in group", ["doc:tutorials/scripting/groups", "SceneTree.get_nodes_in_group"], 3),
    ("collision layers and masks", ["doc:tutorials/physics/physics_introduction#collision-layers-and-masks"], 3),
    ("draw a line", ["doc:tutorials/2d/custom_drawing_in_2d", "CanvasItem.draw_line", "Line2D"], 3),
    ("typed array", ["doc:tutorials/scripting/gdscript/gdscript_basics#typed-arrays", "doc:tutorials/scripting/gdscript/static_typing"], 3),
    ("string format", ["String.format", "doc:tutorials/scripting/gdscript/gdscript_format_string"], 2),
    ("AnimationPlayer play backwards", ["AnimationPlayer.play_backwards"], 1),
    ("wait using a timer", ["Timer", "SceneTree.create_timer"], 3),
    # Identifiers / API
    ("move_and_slide", ["CharacterBody2D.move_and_slide", "CharacterBody3D.move_and_slide"], 2),
    ("CharacterBody2D", ["CharacterBody2D"], 1),
    ("look_at", ["Node2D.look_at", "Node3D.look_at", "Control"], 3),
    ("queue_free", ["Node.queue_free"], 1),
    ("get_node", ["Node.get_node"], 1),
    ("@onready", ["@GDScript.@onready"], 1),
    ("Input.is_action_just_pressed", ["Input.is_action_just_pressed"], 1),
    ("lerp", ["@GlobalScope.lerp"], 2),
    ("body_entered signal", ["Area2D.body_entered", "Area3D.body_entered", "RigidBody2D.body_entered"], 3),
    ("Vector2 normalized", ["Vector2.normalized"], 1),
    ("process_mode", ["Node.process_mode"], 2),
    ("_ready", ["Node._ready"], 1),
    ("Timer timeout", ["Timer.timeout"], 2),
    # Godot 3 names
    ("KinematicBody2D", ["CharacterBody2D"], 1),
    ("yield", ["doc:tutorials/scripting/gdscript/gdscript_basics#awaiting-signals-or-coroutines"], 2),
    ("onready var", ["@GDScript.@onready"], 2),
    ("instance()", ["PackedScene.instantiate"], 2),
    ("rand_range", ["@GlobalScope.randf_range"], 1),
    ("Spatial", ["Node3D"], 1),
    ("change_scene()", ["SceneTree.change_scene_to_file"], 1),
    ("deg2rad", ["@GlobalScope.deg_to_rad"], 1),
]


SEMANTIC_FILE = Path(__file__).resolve().parent / "semantic_queries.json"
CONCEPT_FILE = Path(__file__).resolve().parent / "concept_queries.json"


def matches(rid: str, e: str) -> bool:
    """`expected` entries are id prefixes (a page covers its sections); `=id` means exactly that entry."""
    return rid == e[1:] if e.startswith("=") else rid.startswith(e)


def load_cases(which: str, split: str | None) -> list:
    cases = []
    if which in ("core", "all"):
        cases += CASES
    files = {"semantic": [SEMANTIC_FILE], "concept": [CONCEPT_FILE], "all": [SEMANTIC_FILE, CONCEPT_FILE]}
    for f in files.get(which, []):
        for c in json.loads(f.read_text(encoding="utf-8")):
            if split is None or c["split"] == split:
                cases.append((c["query"], c["expected"] + ["=" + a for a in c.get("accept", [])], c["k"]))
    return cases


ALSO_FILE = Path(__file__).resolve().parent / "also_phrasings.json"


def load_also() -> dict:
    return {k: v for k, v in json.loads(ALSO_FILE.read_text(encoding="utf-8")).items() if not k.startswith("_")}


def evaluate(ix, cases, mode="lexical", model=None, verbose=False, quiet=False, also=None, fusion=None) -> dict:
    """also: {query: second phrasing} -> search with both phrasings (multi_search), as an agent using --also would."""
    passed, mrr, r10, times = 0, 0.0, 0, []
    gd.run_search(ix, "warm up", limit=10, mode=mode, model=model)  # load model / vectors outside the timing
    for query, expected, k in cases:
        t0 = time.perf_counter()
        if also is not None:
            res = gd.multi_search(ix, [query, also.get(query, "")], limit=10, mode=mode, model=model, fusion=fusion)
        else:
            res = gd.run_search(ix, query, limit=10, mode=mode, model=model)
        times.append(time.perf_counter() - t0)
        ids = [r["id"] for r in res["results"]]
        rank = next((i + 1 for i, rid in enumerate(ids) if any(matches(rid, e) for e in expected)), None)
        ok = rank is not None and rank <= k
        passed += ok
        r10 += rank is not None
        mrr += 1.0 / rank if rank else 0.0
        if not quiet and (verbose or not ok):
            print(f"{'PASS' if ok else 'FAIL'}  rank={rank}  k={k}  {query!r}")
            if not ok:
                for i, rid in enumerate(ids[:5], 1):
                    print(f"        {i}. {rid}")
    n = len(cases)
    times.sort()
    return {"n": n, "passed": passed, "pass_rate": passed / n, "mrr": mrr / n, "recall10": r10 / n,
            "ms_median": 1000 * times[n // 2], "ms_p95": 1000 * times[min(n - 1, int(n * 0.95))]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("-q", "--quiet", action="store_true", help="summary line only")
    ap.add_argument("--mode", choices=gd.MODES, default="lexical")
    ap.add_argument("--model", help="embedding model folder under models/")
    ap.add_argument("--set", dest="which", choices=("core", "semantic", "concept", "all"), default="core")
    ap.add_argument("--split", choices=("dev", "val", "test", "report"), help="semantic/concept sets: dev (tuning), val (old test, seen), test (held out), report (concept only)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--also", action="store_true", help="also pass the second phrasing from eval/also_phrasings.json")
    ap.add_argument("--fusion", help="fusion method for --also (default: godot_docs.MULTI_FUSION)")
    args = ap.parse_args()
    ix = gd.open_index(argparse.Namespace(docs=None, index=None))
    cases = load_cases(args.which, args.split)
    m = evaluate(ix, cases, args.mode, args.model, args.verbose, args.quiet or args.json,
                 load_also() if args.also else None, args.fusion)
    if args.json:
        print(json.dumps(m))
    else:
        print(f"\n[{args.which}{'/' + args.split if args.split else ''} {args.mode}{' +also' if args.also else ''}] {m['passed']}/{m['n']} passed "
              f"({100 * m['pass_rate']:.0f}%), MRR {m['mrr']:.3f}, recall@10 {m['recall10']:.2f}, "
              f"median {m['ms_median']:.0f} ms, p95 {m['ms_p95']:.0f} ms")
    return 0 if m["passed"] == m["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
