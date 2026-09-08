import copy
import json
import unittest

from verge_round_core import parse_proposal, config, stage_tokens, allocate_tokens, centered_advantages


class ProtocolTest(unittest.TestCase):
    def test_schema(self):
        cfg = config()
        valid = {"stages": [{"family": "append_sequence", "colors": 2, "length": 1, "mutation": "none"}]}
        self.assertEqual(parse_proposal(json.dumps(valid), cfg), valid)
        for field, bad in (("length", True), ("colors", 3), ("mutation", "pattern"), ("family", "unknown")):
            altered = copy.deepcopy(valid)
            altered["stages"][0][field] = bad
            self.assertIsNone(parse_proposal(json.dumps(altered), cfg))
        self.assertIsNone(parse_proposal('{"stages":[]}', cfg))
        self.assertIsNone(parse_proposal(json.dumps({"stages": valid["stages"] * 5}), cfg))
        self.assertIsNone(parse_proposal("I suggest a curriculum", cfg))

    def test_budget(self):
        for n in range(1, 5):
            parts = stage_tokens(131072, n)
            self.assertEqual(sum(parts), 131072)
            self.assertEqual(parts[-1], 32768)
        self.assertEqual(allocate_tokens([2, 4, 6], 5), [2, 3, 0])
        self.assertEqual(allocate_tokens([2, 4], 100), [2, 4])
        self.assertEqual(allocate_tokens([0, 0], 100), [0, 0])

    def test_centering(self):
        self.assertEqual(centered_advantages([0, 0, 0]), [0, 0, 0])
        self.assertEqual(centered_advantages([-2, -2, -2]), [0, 0, 0])
        a = centered_advantages([-.1, -.2, -.3])
        self.assertGreater(a[0], 0)
        self.assertLess(a[2], 0)
        self.assertAlmostEqual(sum(a), 0)


if __name__ == "__main__":
    unittest.main()
