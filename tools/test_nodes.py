"""CPU tests for the comfyui-krea2-batch nodes: python tools/test_nodes.py
(the Contact Sheet tests need torch and are skipped without it)."""
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
try:
    import torch
except ImportError:
    torch = None
    sys.modules["torch"] = types.ModuleType("torch")
spec = importlib.util.spec_from_file_location("krea2_pack", ROOT / "custom_nodes/comfyui-krea2-batch/__init__.py")
pack = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pack)


def run(prompts="a\nb", prompts_file="", prefix="", suffix="", caption="", seed=10,
        seed_mode="increment", limit=0):
    return pack.Krea2PromptList().run(prompts, prompts_file, prefix, suffix, caption, seed,
                                      seed_mode, limit)


class PromptListTests(unittest.TestCase):
    def test_lines_comments_and_blanks(self):
        prompts, captions, seeds, count = run("# header\n\n  first line  \nsecond\n   # note\n")
        self.assertEqual(prompts, ["first line", "second"])
        self.assertEqual(count, 2)

    def test_prefix_suffix_and_captions(self):
        prompts, captions, _, _ = run("front view\nprofile", prefix="Show the same person:",
                                      suffix="Plain background.", caption="zxq person,")
        self.assertEqual(prompts, ["Show the same person: front view Plain background.",
                                   "Show the same person: profile Plain background."])
        self.assertEqual(captions, ["zxq person, front view", "zxq person, profile"])

    def test_seeds(self):
        self.assertEqual(run("a\nb\nc", seed=7)[2], [7, 8, 9])
        self.assertEqual(run("a\nb\nc", seed=7, seed_mode="fixed")[2], [7, 7, 7])
        self.assertEqual(run("a\nb", seed=0xffffffffffffffff)[2], [0xffffffffffffffff, 0])

    def test_limit(self):
        prompts, captions, seeds, count = run("a\nb\nc\nd", limit=2)
        self.assertEqual((prompts, count, seeds), (["a", "b"], 2, [10, 11]))

    def test_empty_list_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no prompts"):
            run("# only a comment\n\n")

    def test_file_replaces_the_text_box_and_is_watched(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "list.txt"
            path.write_text("﻿one\ntwo\n", encoding="utf-8")
            self.assertEqual(run("ignored", prompts_file=str(path))[0], ["one", "two"])
            first = pack.Krea2PromptList.IS_CHANGED("x", str(path))
            path.write_text("three\n", encoding="utf-8")
            self.assertNotEqual(first, pack.Krea2PromptList.IS_CHANGED("x", str(path)))
            pack.input_dir = lambda: d          # relative names resolve in ComfyUI/input
            self.assertEqual(run(prompts_file="list.txt")[0], ["three"])
        with self.assertRaisesRegex(FileNotFoundError, "ComfyUI/input"):
            run(prompts_file=os.path.join(tempfile.gettempdir(), "no-such-krea2-list.txt"))

    def test_shipped_prompt_file_parses(self):
        lines = pack.read_lines("", str(ROOT / "prompts/krea2_batch_prompts.txt"))
        self.assertEqual(len(lines), 6)

    def test_schema(self):
        cls = pack.Krea2PromptList
        self.assertEqual(cls.OUTPUT_IS_LIST, (True, True, True, False))
        self.assertEqual(len(cls.RETURN_TYPES), len(cls.RETURN_NAMES))


@unittest.skipIf(torch is None, "torch not installed")
class ContactSheetTests(unittest.TestCase):
    def test_grid_geometry_and_placement(self):
        tiles = [torch.full((1, 4, 6, 3), v / 10) for v in range(5)]
        (sheet,) = pack.Krea2ContactSheet().run(tiles, [3], [2], [1.0])
        self.assertEqual(tuple(sheet.shape), (1, 2 * 4 + 3 * 2, 3 * 6 + 4 * 2, 3))
        self.assertAlmostEqual(float(sheet[0, 2, 2, 0]), 0.0)          # tile 0
        self.assertAlmostEqual(float(sheet[0, 2 + 4 + 2, 2 + 6 + 2, 0]), 0.4)   # tile 4
        self.assertAlmostEqual(float(sheet[0, 0, 0, 0]), 1.0)          # background
        self.assertAlmostEqual(float(sheet[0, -3, -3, 0]), 1.0)        # empty 6th cell

    def test_batches_are_flattened_and_sizes_matched(self):
        tiles = [torch.zeros((2, 4, 4, 3)), torch.ones((1, 8, 8, 3))]
        (sheet,) = pack.Krea2ContactSheet().run(tiles, [4], [0], [0.0])
        self.assertEqual(tuple(sheet.shape), (1, 4, 12, 3))


if __name__ == "__main__":
    unittest.main(verbosity=1)
