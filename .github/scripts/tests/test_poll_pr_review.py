"""Tests for exact-SHA PRReviewer polling state."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "poll_pr_review.py"
spec = importlib.util.spec_from_file_location("poll_pr_review", SCRIPT_PATH)
poller = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(poller)

SHA = "a" * 40


def classify(**overrides):
    values = {
        "expected_sha": SHA,
        "actual_sha": SHA,
        "labels": [],
        "comments": [],
        "reviews": [],
        "statuses": [],
        "checks": [],
    }
    values.update(overrides)
    return poller.classify(**values)


def comment():
    return {"user": {"type": "Bot"}, "body": f"<!-- pr-reviewer-sha: {SHA} -->\nReview"}


class ClassifyTest(unittest.TestCase):
    def test_pending_without_an_exact_review(self):
        self.assertEqual(classify()["state"], "pending")

    def test_head_movement_stops_the_loop(self):
        self.assertEqual(classify(actual_sha="b" * 40)["state"], "head_changed")

    def test_completed_review_without_approval(self):
        result = classify(
            labels=["review-risk: medium"],
            comments=[comment()],
            statuses=[{"context": "PRReviewer", "state": "success"}],
        )
        self.assertEqual(result["state"], "reviewed_not_approved")
        self.assertEqual(result["risk_label"], "review-risk: medium")

    def test_exact_bot_approval_is_terminal(self):
        result = classify(
            labels=["review-risk: low"],
            comments=[comment()],
            reviews=[
                {
                    "commit_id": SHA,
                    "state": "APPROVED",
                    "user": {"type": "Bot"},
                    "body": "<!-- pr-reviewer-approval -->",
                }
            ],
        )
        self.assertEqual(result["state"], "approved")

    def test_stale_approval_does_not_count(self):
        result = classify(
            comments=[comment()],
            reviews=[
                {
                    "commit_id": "b" * 40,
                    "state": "APPROVED",
                    "user": {"type": "Bot"},
                    "body": "<!-- pr-reviewer-approval -->",
                }
            ],
        )
        self.assertEqual(result["state"], "pending")

    def test_changes_requested_and_failure_stop_the_loop(self):
        requested = classify(reviews=[{"commit_id": SHA, "state": "CHANGES_REQUESTED"}])
        failed = classify(statuses=[{"context": "PRReviewer", "state": "failure"}])
        self.assertEqual(requested["state"], "changes_requested")
        self.assertEqual(failed["state"], "failed")

    def test_multiple_risk_labels_are_inconsistent(self):
        result = classify(labels=["review-risk: low", "review-risk: high"])
        self.assertEqual(result["state"], "inconsistent")

    def test_unrelated_checks_do_not_complete_the_review(self):
        result = classify(
            labels=["review-risk: low"],
            comments=[comment()],
            checks=[{"name": "unit-tests", "status": "completed", "conclusion": "success"}],
        )
        self.assertEqual(result["state"], "pending")


if __name__ == "__main__":
    unittest.main()
