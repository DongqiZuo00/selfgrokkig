"""Synthetic JSON-schema acceptance tests; no hand-authored proposal is submitted."""
import copy
import importlib.util
import json
import unittest

from json_schema_support import expand_structured_constants


def fixture_schema():
    options = []
    for name, op in (("fixture_copy", "input"), ("fixture_fixed", "append")):
        specification = {"op": op, "distribution": {"alphabet": "RB", "length": [0, 8]},
                         "parameters": [True, None, 3]}
        options.append({"type": "object", "additionalProperties": False,
                        "required": ["stage_id", "kind", "spec"], "properties": {
                            "stage_id": {"const": name}, "kind": {"const": "tape_transform"},
                            "spec": {"const": specification}}})
    course = {"type": "array", "minItems": 3, "maxItems": 3, "items": {"oneOf": options}}
    return {"type": "object", "additionalProperties": False,
            "required": ["base_checkpoint", "curricula"], "properties": {
                "base_checkpoint": {"const": "SYNTHETIC_base"},
                "curricula": {"type": "object", "additionalProperties": False,
                              "required": ["g1", "g2", "g3"],
                              "properties": {name: copy.deepcopy(course) for name in ("g1", "g2", "g3")}}}}


def fixture_value():
    value = {"stage_id": "fixture_copy", "kind": "tape_transform",
             "spec": {"op": "input", "distribution": {"alphabet": "RB", "length": [0, 8]},
                      "parameters": [True, None, 3]}}
    return {"base_checkpoint": "SYNTHETIC_base",
            "curricula": {name: [copy.deepcopy(value) for _ in range(3)] for name in ("g1", "g2", "g3")}}


class ConstExpansionTests(unittest.TestCase):
    def test_nested_structured_constants_are_explicit_and_input_unchanged(self):
        source = {"const": {"x": [1, {"y": "z"}]}}
        before = copy.deepcopy(source)
        expanded, count = expand_structured_constants(source)
        self.assertEqual(source, before)
        self.assertEqual(count, 1)
        self.assertFalse(expanded["additionalProperties"])
        array = expanded["properties"]["x"]
        self.assertEqual((array["minItems"], array["maxItems"]), (2, 2))
        self.assertFalse(array["items"])
        self.assertEqual(array["prefixItems"][1]["properties"]["y"], {"const": "z"})

    def test_literal_enum_and_property_named_const_are_not_treated_as_keywords(self):
        source = {"type": "object", "properties": {"const": {"type": "object", "enum": [{"const": {"x": 1}}]}}}
        actual, count = expand_structured_constants(source)
        self.assertEqual(actual, source)
        self.assertEqual(count, 0)

    def test_conflicting_or_unhandled_structured_const_constraints_fail(self):
        for source in ({"const": {}, "type": "array"}, {"const": {}, "required": ["x"]}):
            with self.assertRaises(ValueError):
                expand_structured_constants(source)


@unittest.skipUnless(importlib.util.find_spec("xgrammar") is not None, "run in installed vLLM CPU environment")
class ActualXGrammarJsonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import xgrammar as xgr
        cls.xgr = xgr
        info = xgr.TokenizerInfo([chr(i) for i in range(128)] + ["<eos>"], xgr.VocabType.RAW,
                                vocab_size=129, stop_token_ids=[128])
        normalized, cls.expansions = expand_structured_constants(fixture_schema())
        cls.compiled = xgr.GrammarCompiler(info, max_threads=2).compile_json_schema(json.dumps(normalized), any_whitespace=True)

    def accepts(self, value, prefix=""):
        text = json.dumps(value, separators=(",", ":"))
        matcher = self.xgr.GrammarMatcher(self.compiled)
        if prefix and not matcher.accept_string(prefix):
            return False
        return matcher.accept_string(text[len(prefix):]) and matcher.accept_token(128) and matcher.is_terminated()

    def test_full_proposal_shape_and_empty_or_open_brace_prefix(self):
        self.assertEqual(self.expansions, 6)
        self.assertTrue(self.accepts(fixture_value()))
        self.assertTrue(self.accepts(fixture_value(), "{"))

    def test_wrong_base_kind_spec_or_course_length_are_rejected(self):
        valid = fixture_value()
        variants = []
        for field, value in (("stage_id", "unknown"), ("kind", "registered"), ("spec", {"op": "wrong"})):
            candidate = copy.deepcopy(valid)
            candidate["curricula"]["g1"][0][field] = value
            variants.append(candidate)
        candidate = copy.deepcopy(valid)
        candidate["base_checkpoint"] = "OTHER"
        variants.append(candidate)
        candidate = copy.deepcopy(valid)
        candidate["curricula"]["g3"].pop()
        variants.append(candidate)
        candidate = copy.deepcopy(valid)
        candidate["extra"] = 1
        variants.append(candidate)
        for value in variants:
            with self.subTest(value=value):
                self.assertFalse(self.accepts(value))


if __name__ == "__main__":
    unittest.main(verbosity=2)
