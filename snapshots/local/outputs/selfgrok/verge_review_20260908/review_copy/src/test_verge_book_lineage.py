import unittest
from verge_book_lineage import optimizer_counters


class CounterTests(unittest.TestCase):
    def fixture(self, prior):
        return {"runtime_state": {"teacher_steps_prior": prior}, "optimizer": {
            "state": {0: {"step": prior}} if prior else {},
            "param_groups": [{"lr": 1e-6, "weight_decay": 0}]}}

    def test_fresh_and_inherited(self):
        self.assertEqual(optimizer_counters(self.fixture(0), 0, 1e-6)["parameter_states"], 0)
        self.assertEqual(optimizer_counters(self.fixture(2), 2, 1e-6)["distinct_parameter_step_counters"], [2])

    def test_lost_optimizer_rejected(self):
        saved = self.fixture(2)
        saved["optimizer"]["state"] = {}
        with self.assertRaises(RuntimeError):
            optimizer_counters(saved, 2, 1e-6)

    def test_changed_lr_or_counter_rejected(self):
        with self.assertRaises(RuntimeError):
            optimizer_counters(self.fixture(2), 1, 1e-6)
        with self.assertRaises(RuntimeError):
            optimizer_counters(self.fixture(2), 2, 1e-5)


if __name__ == "__main__":
    unittest.main()
