import copy
import json
from pathlib import Path
import tempfile
import unittest
import freeze_naming_prompt_v2 as view


class NamingViewTests(unittest.TestCase):
    def test_exact_task_order_caps_and_seeds(self):
        payload, meta = view.build()
        self.assertEqual(len(payload["tasks"]), 12)
        self.assertEqual(payload["samples_per_task"], 8)
        for i, task in enumerate(payload["tasks"]):
            self.assertEqual(task["seed"], 2026090800 + i)
            self.assertEqual(task["cap"], 2048 if task["stage_id"].startswith("target_small") else 768)
        self.assertEqual([(t["stage_id"], t["variant"]) for t in payload["tasks"][-3:]],
            [(s, "official") for s in ("identity", "append_R", "append_B")])

    def test_exactly_one_common_prompt_delta_all_other_fields_unchanged(self):
        catalogue = json.loads((view.HERE / "generated/catalogue.json").read_text(encoding="utf-8"))
        original = {s["stage_id"]: s for s in catalogue["stages"]}
        payload, _ = view.build()
        for task in payload["tasks"]:
            row = copy.deepcopy(task["row"])
            old = original[task["stage_id"]]["rows_by_variant"][task["variant"]][0]
            self.assertEqual(row["messages"][0]["content"], old["messages"][0]["content"] + "\n\n" + view.PROMPT_DELTA)
            row["messages"] = old["messages"]
            self.assertEqual(row, old)

    def test_generic_names_do_not_fix_route(self):
        self.assertIn("may be start, end, NONE, or any intermediate name", view.PROMPT_DELTA)
        self.assertIn("including one declared later", view.PROMPT_DELTA)
        self.assertNotIn("NEXT n0", view.PROMPT_DELTA)
        self.assertNotIn("PAINTER", view.PROMPT_DELTA)
        self.assertNotIn("PULLER", view.PROMPT_DELTA)

    def test_freeze_idempotent_refuses_replacement(self):
        with tempfile.TemporaryDirectory(dir=view.HERE) as path:
            output = Path(path)
            view.prepare(output)
            before = (output / "screen_tasks.json").read_bytes()
            view.prepare(output)
            self.assertEqual(before, (output / "screen_tasks.json").read_bytes())
            (output / "screen_tasks.json").write_text("changed", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                view.prepare(output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
