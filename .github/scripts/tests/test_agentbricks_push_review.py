"""Check automatic push-review budget persistence and exhaustion."""

from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "agentbricks_push_review.py"
spec = importlib.util.spec_from_file_location("push_review", SCRIPT)
budget = importlib.util.module_from_spec(spec)
spec.loader.exec_module(budget)


def comment(attempts, login="github-actions[bot]", identifier=42):
    return {
        "id": identifier, "user": {"login": login},
        "body": f"{budget.MARKER} {attempts} -->\nAttempt reserved.",
    }


class BudgetTest(unittest.TestCase):
    def test_first_attempt_creates_a_counter(self):
        result = budget.plan([], 3, "https://example.com/run")
        self.assertFalse(result["skip"])
        self.assertEqual(result["attempts"], 1)
        self.assertIsNone(result["id"])
        self.assertIn("1/3", result["body"])

    def test_last_attempt_updates_the_same_comment(self):
        result = budget.plan([comment(2)], 3, "https://example.com/run")
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(result["id"], 42)
        self.assertFalse(result["skip"])

    def test_exhausted_or_disabled_budget_does_not_reserve(self):
        self.assertTrue(budget.plan([comment(3)], 3, "url")["skip"])
        self.assertTrue(budget.plan([comment(4)], 3, "url")["skip"])
        self.assertTrue(budget.plan([], 0, "url")["skip"])

    def test_only_the_latest_bot_counter_is_used(self):
        records = [comment(99, login="author"), comment(1, identifier=1), comment(2)]
        self.assertEqual(budget.plan(records, 3, "url")["attempts"], 3)

    def test_invalid_configuration_or_counter_stops(self):
        with self.assertRaises(ValueError):
            budget.plan([], -1, "url")
        record = comment(1)
        record["body"] = budget.MARKER + " unreadable -->"
        with self.assertRaises(ValueError):
            budget.plan([record], 3, "url")

    def run_cli(self):
        budget.main(["--repo", "org/repo", "--pr", "1", "--limit", "3", "--run-url", "url"])

    def test_cli_reads_all_pages_and_persists_before_allowing(self):
        output = io.StringIO()
        with patch.object(budget, "request", side_effect=[[[comment(1, identifier=1)], [comment(2)]], {}]) as api:
            with redirect_stdout(output):
                self.run_cli()
        self.assertTrue(api.call_args_list[0].kwargs["paginate"])
        self.assertEqual(api.call_args_list[1].args[0], "repos/org/repo/issues/comments/42")
        self.assertIn("3/3", api.call_args_list[1].args[1]["body"])
        self.assertEqual(output.getvalue(), "skip=false\n")

    def test_cli_does_not_allow_review_when_persistence_fails(self):
        output = io.StringIO()
        with patch.object(budget, "request", side_effect=[[[]], RuntimeError("write failed")]):
            with redirect_stdout(output), self.assertRaises(RuntimeError):
                self.run_cli()
        self.assertEqual(output.getvalue(), "")

    def test_cli_does_not_write_after_exhaustion(self):
        output = io.StringIO()
        with patch.object(budget, "request", return_value=[[comment(3)]]) as api:
            with redirect_stdout(output):
                self.run_cli()
        self.assertEqual(api.call_count, 1)
        self.assertEqual(output.getvalue(), "skip=true\n")


if __name__ == "__main__":
    unittest.main()
