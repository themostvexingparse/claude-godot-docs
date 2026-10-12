#!/usr/bin/env python3
"""Unit tests for the RST conversion and query analysis, plus smoke tests against the real index.

Run: python -m unittest discover tests -v
"""
import argparse
import subprocess
import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
import godot_docs as gd  # noqa: E402


def converter(cls=None):
    r = gd.Resolver()
    r.anchors["class_vector2"] = ("Vector2", "Vector2", "class")
    r.anchors["class_node2d_method_rotate"] = ("Node2D.rotate", "rotate", "method")
    r.anchors["doc_signals"] = ("doc:getting_started/step_by_step/signals", "Using signals", "doc")
    r.pages["tutorials/2d/2d_movement"] = ("doc:tutorials/2d/2d_movement", "2D movement overview")
    return gd.RstConverter(r, "tutorials/2d/page.rst", cls)


class InlineTests(unittest.TestCase):
    def test_refs(self):
        c = converter()
        self.assertEqual(c.inline(":ref:`Vector2<class_Vector2>`"), "Vector2")
        self.assertEqual(c.inline(":ref:`rotate()<class_Node2D_method_rotate>`"), "`Node2D.rotate()`")
        self.assertEqual(converter("Node2D").inline(":ref:`rotate()<class_Node2D_method_rotate>`"), "`rotate()`")
        self.assertEqual(c.inline("See :ref:`doc_signals`."), "See Using signals [doc:getting_started/step_by_step/signals].")
        self.assertEqual(c.inline(":ref:`🔗<class_Vector2>`"), "")
        self.assertEqual(c.inline(":ref:`Unknown <nowhere>`"), "Unknown")

    def test_doc_role_relative(self):
        self.assertEqual(converter().inline(":doc:`2d_movement`"), "2D movement overview [doc:tutorials/2d/2d_movement]")

    def test_misc_roles_and_escapes(self):
        c = converter()
        self.assertEqual(c.inline("Press :kbd:`Ctrl + S`"), "Press Ctrl + S")
        self.assertEqual(c.inline(":abbr:`API (Application Programming Interface)`"), "API")
        self.assertEqual(c.inline("``a\\ b`` and \\ **Note:**"), "`a\\ b` and **Note:**")
        self.assertEqual(c.inline("`Godot <https://godotengine.org>`__"), "[Godot](https://godotengine.org)")
        self.assertEqual(c.inline("|void|"), "void")


class BlockTests(unittest.TestCase):
    def test_code_tabs_and_filter(self):
        c = converter()
        text = c.text([
            ".. tabs::",
            " .. code-tab:: gdscript GDScript",
            "",
            "    var x = 1",
            "",
            " .. code-tab:: csharp",
            "",
            "    int x = 1;",
        ])
        self.assertIn("```gdscript|g", text)
        self.assertEqual([(l, code) for l, code, _ in c.codes], [("gdscript", "var x = 1"), ("csharp", "int x = 1;")])
        gds = gd.filter_code(text, "gdscript")
        self.assertIn("var x = 1", gds)
        self.assertNotIn("int x", gds)
        self.assertNotIn("|g", gds)
        cs = gd.filter_code(text, "csharp")
        self.assertIn("int x = 1;", cs)
        self.assertNotIn("var x", cs)
        both = gd.filter_code(text, "all")
        self.assertIn("var x", both)
        self.assertIn("int x", both)

    def test_tab_group_without_preferred_language_is_kept(self):
        text = converter().text([".. tabs::", " .. code-tab:: bash", "", "    scons", ""])
        self.assertIn("scons", gd.filter_code(text, "gdscript"))

    def test_literal_block(self):
        text = converter().text(["Example::", "", "    print(1)", "", "After."])
        self.assertEqual(text, "Example:\n\n```gdscript\nprint(1)\n```\n\nAfter.")

    def test_admonition(self):
        text = converter().text([".. note:: First line", "   continues here.", "", "Next"])
        self.assertTrue(text.startswith("**Note:** First line\ncontinues here."))
        self.assertIn("Next", text)

    def test_skipped_directives_and_labels(self):
        text = converter().text([".. _doc_x:", "", ".. image:: img/a.png", "   :width: 10", "", "Text", "", ".. a comment"])
        self.assertEqual(text, "Text")

    def test_grid_table(self):
        text = converter().text(["+-----+-----+", "| A   | B   |", "+=====+=====+", "| x   | y   |", "| x2  |     |", "+-----+-----+"])
        self.assertEqual(text, "| A | B |\n| x x2 | y |")

    def test_headings(self):
        lines = ["Title", "=====", "", "Intro", "", "Sub :ref:`thing <x>`", "-----------------------", "", "Body"]
        heads = gd.find_headings(lines)
        self.assertEqual([(h.level, h.title) for h in heads], [(1, "Title"), (2, "Sub thing")])
        self.assertEqual(gd.slugify("8-way movement"), "8-way-movement")


class SignatureTests(unittest.TestCase):
    def test_method(self):
        c = converter("Node2D")
        sig, quals = gd.format_signature(
            c, "|void| **look_at**\\ (\\ point\\: :ref:`Vector2<class_Vector2>`\\ ) |const| :ref:`🔗<class_Node2D_method_rotate>`", "method")
        self.assertEqual(sig, "look_at(point: Vector2) -> void [const]")
        self.assertEqual(quals, ["const"])

    def test_property_and_constant(self):
        c = converter()
        self.assertEqual(gd.format_signature(c, ":ref:`Vector2<class_Vector2>` **position** = ``Vector2(0, 0)``", "property")[0],
                         "position: Vector2 = Vector2(0, 0)")
        self.assertEqual(gd.format_signature(c, "**NOTIFICATION_READY** = ``13``", "constant")[0], "NOTIFICATION_READY = 13")

    def test_split_ident(self):
        self.assertEqual(gd.split_ident("CharacterBody2D"), ["character", "body", "2d"])
        self.assertEqual(gd.split_ident("HTTPRequest"), ["http", "request"])
        self.assertEqual(gd.split_ident("move_and_slide"), ["move", "and", "slide"])


class RebuildFallbackTests(unittest.TestCase):
    """A rebuild that is impossible or fails keeps the existing index, with one warning, instead of erroring."""

    def setUp(self):
        import sqlite3
        import tempfile
        from pathlib import Path
        self.tmp = tempfile.TemporaryDirectory()
        self.index = Path(self.tmp.name) / "idx.db"
        conn = sqlite3.connect(self.index)
        conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        conn.executemany("INSERT INTO meta VALUES (?, ?)", [("built_at", "2026-10-07"), ("godot_version", "4.5")])
        conn.commit()
        conn.close()
        self.meta = gd.read_index_meta(self.index)
        self.saved_build = gd.build_index

    def tearDown(self):
        gd.build_index = self.saved_build
        self.tmp.cleanup()

    def call(self, docs_root, meta="existing", force=False):
        import contextlib
        import io
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            gd.rebuild_or_keep(docs_root, self.index, self.meta if meta == "existing" else meta, force)
        return err.getvalue()

    def test_no_docs_warns_once(self):
        self.assertIn("WARNING: no godot-docs checkout found", self.call(None))
        self.assertEqual(self.call(None), "")  # same tool version: not repeated
        self.assertIn("WARNING", self.call(None, force=True))  # index --rebuild always reports

    def test_failed_rebuild_warns_and_keeps_index(self):
        calls = []

        def broken(docs_root, index_path, log=None):
            calls.append(1)
            index_path.with_name(f"{index_path.name}.tmp{gd.os.getpid()}").write_text("partial")
            raise ValueError("bad rst")

        gd.build_index = broken
        docs = self.index.parent
        out = self.call(docs)
        self.assertIn("WARNING: reindexing failed (ValueError: bad rst)", out)
        self.assertIn("built 2026-10-07", out)
        self.assertTrue(self.index.is_file())
        self.assertFalse(any(p.name.startswith("idx.db.tmp") for p in docs.iterdir()))  # partial build removed
        self.assertEqual(self.call(docs), "")  # known failure: not retried on every call
        self.assertEqual(len(calls), 1)

    def test_no_index_and_no_docs_is_still_an_error(self):
        with self.assertRaises(gd.DocsError):
            self.call(None, meta=None)


class DeprecationTests(unittest.TestCase):
    """Members of a deprecated class (TileMap -> TileMapLayer) are reported as deprecated everywhere."""

    @classmethod
    def setUpClass(cls):
        cls.ix = gd.open_index(argparse.Namespace(docs=None, index=None))

    def ns(self, **kw):
        base = dict(kind=None, cls=None, path=None, limit=8, lang="gdscript", mode="lexical", model=None, also=None,
                    full=False, offset=0, max_chars=None, inherited=False, filter=None, no_summary=True)
        return argparse.Namespace(**{**base, **kw})

    def test_member_of_deprecated_class_is_flagged(self):
        row = self.ix.get("TileMap.set_cell")
        self.assertIn("deprecated", self.ix.flags_of(row))
        self.assertIn("TileMapLayer", self.ix.deprecation_note(row))
        self.assertEqual(self.ix.deprecation_note(self.ix.get("TileMapLayer.set_cell")), "")
        self.assertEqual(self.ix.deprecation_note(self.ix.get("TileMap")), "")  # the class page says it itself

    def test_read_shows_deprecation(self):
        _, text, _ = gd.cmd_read(self.ix, self.ns(id="TileMap.set_cell"))
        self.assertIn("DEPRECATED: TileMap is deprecated", text.split("\n\n")[0])  # in the header, before the description

    def test_lookup_puts_deprecated_namesakes_last(self):
        data, text, _ = gd.cmd_lookup(self.ix, self.ns(symbol="set_cell"))
        ids = [m["id"] for m in data["matches"]]
        self.assertLess(ids.index("TileMapLayer.set_cell"), ids.index("TileMap.set_cell"))
        self.assertIn("TileMap.set_cell  (method)", text)
        self.assertTrue(next(m for m in data["matches"] if m["id"] == "TileMap.set_cell")["deprecated"])

    def test_members_and_search_flag_it(self):
        _, text, _ = gd.cmd_members(self.ix, self.ns(cls="ParallaxLayer", max_chars=gd.DEFAULT_MAX_CHARS))
        self.assertIn("DEPRECATED: ParallaxLayer is deprecated", text)
        res = gd.run_search(self.ix, "TileMap set_cell", limit=10)
        hit = next(r for r in res["results"] if r["id"] == "TileMap.set_cell")
        self.assertIn("deprecated", hit["flags"])


class IndexSmokeTests(unittest.TestCase):
    """Requires the godot-docs checkout next to this folder (or GODOT_DOCS_PATH)."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.ix = gd.open_index(argparse.Namespace(docs=None, index=None))
        except gd.DocsError as exc:
            raise unittest.SkipTest(str(exc))

    def test_counts(self):
        self.assertGreater(int(self.ix.meta["classes"]), 900)
        self.assertGreater(int(self.ix.meta["pages"]), 400)

    def test_every_class_member_has_signature(self):
        bad = self.ix.conn.execute(
            "SELECT id FROM entries WHERE kind NOT IN ('class','page','section') AND (signature IS NULL OR signature = '')").fetchall()
        self.assertEqual(bad, [])

    def test_no_raw_rst_left_in_bodies(self):
        leaks = self.ix.conn.execute(
            "SELECT id FROM entries WHERE body LIKE '%:ref:`%' OR body LIKE '%.. code-tab::%' OR body LIKE '%.. note::%' LIMIT 5").fetchall()
        self.assertEqual([r[0] for r in leaks], [])

    def test_curated_alias_targets_resolve(self):
        for a in gd.CURATED_ALIASES:
            row = self.ix.conn.execute("SELECT target FROM aliases WHERE old = ?", (a[0],)).fetchone()
            self.assertTrue(row and row[0] and self.ix.get(row[0]), f"alias {a[0]!r} has no valid target")
        for old, new in gd.CURATED_RENAMES:
            self.assertIsNotNone(self.ix.get(new), f"rename target {new!r} for {old!r} does not exist")

    def test_resolve(self):
        self.assertEqual(gd.resolve_id(self.ix, "move_and_slide")[0], None)  # ambiguous: 2D and 3D
        self.assertEqual(gd.resolve_id(self.ix, "CharacterBody2D.position")[0]["id"], "Node2D.position")
        self.assertEqual(gd.resolve_id(self.ix, "KinematicBody2D")[0]["id"], "CharacterBody2D")
        self.assertEqual(gd.resolve_id(self.ix, "doc_signals")[0]["id"], "doc:getting_started/step_by_step/signals")
        self.assertEqual(gd.resolve_id(self.ix, "singletons_autoload")[0]["id"], "doc:tutorials/scripting/singletons_autoload")
        self.assertEqual(gd.resolve_id(self.ix, "onready")[0]["id"], "@GDScript.@onready")

    def test_operator_ids(self):
        row = self.ix.get("Vector2.operator*")
        self.assertIsNotNone(row)
        self.assertGreaterEqual(len(row["signature"].splitlines()), 3)  # overloads merged
        self.assertIsNotNone(self.ix.get("float.operator**"))

    def test_read_paginates(self):
        args = argparse.Namespace(id="@GlobalScope", full=False, offset=0, max_chars=3000, lang="gdscript")
        data, text, code = gd.cmd_read(self.ix, args)
        self.assertEqual(code, 0)
        self.assertIsNotNone(data["next_offset"])
        self.assertIn("--offset", text)


    def test_two_letter_classes_indexed(self):  # "OS"/"IP" have a two-char title underline
        self.assertIsNotNone(self.ix.get("OS"))
        self.assertIsNotNone(self.ix.get("OS.get_user_data_dir"))
        self.assertIsNotNone(self.ix.get("IP"))


class RenderSearchTests(unittest.TestCase):
    def _res(self, tiers):
        rows = [{"id": f"Node.m{i}", "kind": "method", "title": f"Node.m{i}", "signature": f"m{i}() -> void",
                 "summary": "Does a thing.", "snippet": "", "flags": [], "tier": t, "confidence": 0.5} for i, t in enumerate(tiers)]
        return {"query": "q", "godot_version": "4.5", "lang": "gdscript", "mode": "hybrid", "notes": [], "results": rows, "more": []}

    def test_weak_tail_collapsed_and_best_read(self):
        text = gd.render_search(self._res(["weak", "partial", "weak", "weak", "weak"]))
        self.assertIn("1. [weak] Node.m0 (method)", text)
        self.assertIn("3. [weak] Node.m2", text)  # at least MIN_FULL_RESULTS shown in full
        self.assertIn("+2 weaker: Node.m3, Node.m4", text)
        self.assertIn("No strong match.", text)
        self.assertTrue(text.endswith("Read: read Node.m1"))  # first partial when nothing is strong

    def test_strong_results_no_rephrase_hint(self):
        text = gd.render_search(self._res(["strong", "weak", "weak", "weak"]))
        self.assertNotIn("No strong match", text)
        self.assertTrue(text.endswith("Read: read Node.m0"))

    def test_json_has_tiers_not_internal_signals(self):
        ix = gd.open_index(argparse.Namespace(docs=None, index=None))
        args = argparse.Namespace(query="look_at", kind=None, cls=None, path=None, limit=3, lang="gdscript",
                                  mode="lexical", model=None)
        data, text, code = gd.cmd_search(ix, args)
        self.assertEqual(code, 0)
        self.assertEqual(data["results"][0]["tier"], "strong")
        self.assertTrue(all("_signals" not in r for r in data["results"]))

    def test_also_merges_phrasings(self):
        ix = gd.open_index(argparse.Namespace(docs=None, index=None))
        args = argparse.Namespace(query="slow motion effect", also=["Engine time_scale"], kind=None, cls=None, path=None,
                                  limit=5, lang="gdscript", mode="lexical", model=None)
        data, text, code = gd.cmd_search(ix, args)
        ids = [r["id"] for r in data["results"]]
        self.assertIn("Engine.time_scale", ids)
        self.assertEqual(len(ids), len(set(ids)))  # each entry once, even if both phrasings found it
        self.assertTrue(all("_signals" not in r and "_group" not in r for r in data["results"]))
        self.assertTrue(text.startswith('search "slow motion effect" --also "Engine time_scale"'))

    def test_also_single_phrasing_is_plain_search(self):
        ix = gd.open_index(argparse.Namespace(docs=None, index=None))
        a = gd.multi_search(ix, ["look_at", "look_at", " "], limit=5)
        b = gd.run_search(ix, "look_at", limit=5)
        self.assertEqual([r["id"] for r in a["results"]], [r["id"] for r in b["results"]])


class PrettyTests(unittest.TestCase):
    def setUp(self):
        import godot_pretty
        self.gp = godot_pretty
        self.saved, godot_pretty.COLOUR = godot_pretty.COLOUR, False

    def tearDown(self):
        self.gp.COLOUR = self.saved

    def test_markdown_layout(self):
        text = self.gp.markdown("# Title  (class, Godot 4.5)\nSource: x.rst:1\n\n## Members (`read X.y`)\n"
                                "- start(time_sec: float = -1) -> void\n\n```gdscript\nfunc f():\n    pass\n```")
        self.assertIn("Members (read X.y)", text)
        self.assertIn("• start(time_sec: float = -1) -> void", text)
        self.assertIn("│ func f():", text)
        self.assertNotIn("```", text)

    def test_member_tables_and_alignment(self):
        text = self.gp.markdown("# Timer  (class, Godot 4.5)\nInherits: Node < Object\n\n### Properties (2)\n"
                                "- one_shot: bool = false\n- time_left: float\n\n### Methods (2)\n"
                                "- start(time_sec: float = -1) -> void\n- is_stopped() -> bool [const]\n")
        self.assertIn("│ Name      │ Type   │ Default │", text)
        self.assertIn("│ start      │ time_sec: float = -1 │ void    │       │", text)
        self.assertIn("│ is_stopped │                      │ bool    │ const │", text)
        boxes, cur = [], []
        for line in text.split("\n") + [""]:
            if line[:1] in ("┌", "│", "├", "└", "╭", "╰") and line[-1:] in ("┐", "│", "┤", "┘", "╮", "╯"):
                cur.append(len(line))
            elif cur:
                boxes.append(cur)
                cur = []
        self.assertEqual(len(boxes), 3)  # info box + two member tables
        self.assertTrue(all(len(set(b)) == 1 for b in boxes), boxes)

    def test_wrap_id_breaks_at_joints(self):
        self.assertEqual(self.gp.wrap_id("doc:tutorials/scripting/pausing_games#how-pausing-works", 26),
                         ["doc:tutorials/scripting/", "pausing_games#", "how-pausing-works"])

    def test_colour_codes_only_when_enabled(self):
        self.assertEqual(self.gp.c("x", "bold"), "x")
        self.gp.COLOUR = True
        self.assertEqual(self.gp.c("x", "bold"), "\x1b[1mx\x1b[0m")


class EmbedTextTests(unittest.TestCase):
    """Passage building for the optional embedding layer (no model or numpy needed)."""

    def test_lexical_search_does_not_import_embedding_stack(self):
        code = (f"import sys; sys.path.insert(0, {str(SRC)!r}); import argparse, godot_docs as gd; "
                "ix = gd.open_index(argparse.Namespace(docs=None, index=None)); "
                "gd.run_search(ix, 'queue_free'); "
                "print(sorted(m for m in ('numpy', 'onnxruntime', 'tokenizers', 'godot_embed', 'godot_pretty') if m in sys.modules))")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
        self.assertEqual(out, "[]")

    def test_chunks(self):
        import godot_embed as ge
        text = "\n".join(f"Paragraph {i} " + "word " * 60 for i in range(20))
        chunks = ge._chunks(text, size=500, limit=4)
        self.assertEqual(len(chunks), 4)
        self.assertTrue(all(len(c) <= 500 for c in chunks))
        self.assertEqual(ge._chunks(""), [""])
        long_para = "Sentence one is here. " * 100
        self.assertTrue(all(len(c) <= 1000 for c in ge._chunks(long_para, limit=10)))

    def test_prose_drops_code(self):
        import godot_embed as ge
        self.assertEqual(ge._prose("Intro **bold** `x`\n```gdscript\nvar a = 1\n```\nOutro"), "Intro bold x\nOutro")

    def test_passage_texts(self):
        import godot_embed as ge
        ix = gd.open_index(argparse.Namespace(docs=None, index=None))
        row = ix.conn.execute("SELECT * FROM entries WHERE id = 'CharacterBody2D.floor_stop_on_slope'").fetchone()
        (p,) = ge.passage_texts(row)
        self.assertTrue(p.startswith("CharacterBody2D.floor_stop_on_slope (property) floor_stop_on_slope: bool"))
        self.assertIn("slopes", p)
        sec = ix.conn.execute("SELECT * FROM entries WHERE id = 'doc:tutorials/2d/2d_movement#8-way-movement'").fetchone()
        ps = ge.passage_texts(sec)
        self.assertTrue(all(x.startswith(sec["title"]) for x in ps))
        self.assertNotIn("```", "".join(ps))
        enum = ix.conn.execute("SELECT * FROM entries WHERE id = 'Node.ProcessMode'").fetchone()
        self.assertIn("PROCESS_MODE_ALWAYS", ge.passage_texts(enum)[0])


class ConceptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ix = gd.open_index(argparse.Namespace(docs=None, index=None))

    def ids(self, query, n=5):
        return [r["id"] for r in gd.run_search(self.ix, query, limit=n)["results"]]

    def test_targets_exist(self):
        import json
        data = json.loads(gd.CONCEPTS_FILE.read_text(encoding="utf-8"))
        missing = [t for c in data["concepts"] for t in c["targets"] if self.ix.get(t) is None]
        self.assertEqual(missing, [])

    def test_cross_engine_term(self):
        res = gd.run_search(self.ix, "spawn a prefab", limit=5)
        self.assertIn("PackedScene.instantiate", [r["id"] for r in res["results"]][:3])
        self.assertTrue(any("PackedScene" in n for n in res["notes"]))

    def test_common_word_alone_does_not_trigger(self):
        # "close" is in the phrase "how close is" (distance), but alone it is too common to count
        res = gd.run_search(self.ix, "close a popup window", limit=5)
        self.assertFalse(any("distance_to" in n for n in res["notes"]))
        self.assertNotIn("Vector2.distance_to", [r["id"] for r in res["results"]])

    def test_filters_apply_to_concept_targets(self):
        self.assertTrue(all(i.startswith("Area") for i in self.ids("ontriggerenter", 3)))
        # its targets are signals and classes, so a method-only search has nothing to show
        self.assertEqual(gd.run_search(self.ix, "ontriggerenter", kinds=["method"])["results"], [])

    def test_extra_passages(self):
        ns, texts, sig = self.ix.extra_passages()
        self.assertEqual(len(ns), len(texts))
        self.assertIn("prefab", texts)
        self.assertTrue(sig)


if __name__ == "__main__":
    unittest.main()
