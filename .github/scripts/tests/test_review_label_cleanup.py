"""Verify risk-label reconciliation across pushes, publication, and API failures."""

import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "agentbricks_risk_labels.py"
spec = importlib.util.spec_from_file_location("risk_labels", SCRIPT)
labels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(labels)


def comment(head, label, identifier=1, user="github-actions[bot]"):
    return {"id": identifier, "user": {"login": user}, "body": (
        f"<!-- agentbricks-reviewer-risk-sha: {head} -->\n"
        f"<!-- agentbricks-reviewer-risk-label: {label} -->\n"
    )}


class FakeGitHub:
    def __init__(self, comments=()):
        self.head = "a"
        self.comments = list(comments)
        self.labels = {"unrelated", "risk:high"}
        self.on_mutation = None

    def __call__(self, endpoint, data=None, method=None, **kwargs):
        if "/pulls/" in endpoint:
            return {"head": {"sha": self.head}}
        if "/comments?" in endpoint:
            return [self.comments[:1], self.comments[1:]]
        if method == "POST":
            self.labels.update(data["labels"])
        elif method == "DELETE":
            self.labels.discard(endpoint.rsplit("/", 1)[1])
        else:
            raise AssertionError(endpoint)
        if self.on_mutation:
            callback, self.on_mutation = self.on_mutation, None
            callback()


class LabelCleanupTest(unittest.TestCase):
    def test_current_review_sets_one_risk_label_and_preserves_other_labels(self):
        api = FakeGitHub([comment("a", "risk:low"), comment("a", "risk:medium", 2)])
        self.assertEqual(labels.reconcile(api, "org/repo", 1), "risk:medium")
        self.assertEqual(api.labels, {"unrelated", "risk:medium"})

    def test_all_risk_labels_replace_previous_assessments(self):
        for label in labels.LABELS:
            with self.subTest(label=label):
                api = FakeGitHub([comment("a", label)])
                api.labels.update(labels.LABELS)
                self.assertEqual(labels.reconcile(api, "org/repo", 1), label)
                self.assertEqual(api.labels, {"unrelated", label})

    def test_unreviewed_head_clears_old_risk(self):
        api = FakeGitHub([comment("previous", "risk:low")])
        api.labels.update(labels.LABELS)
        self.assertIsNone(labels.reconcile(api, "org/repo", 1))
        self.assertEqual(api.labels, {"unrelated"})

    def test_human_comments_do_not_define_risk(self):
        api = FakeGitHub([comment("a", "risk:low", user="author")])
        self.assertIsNone(labels.reconcile(api, "org/repo", 1))

    def test_push_during_publication_is_reconciled(self):
        api = FakeGitHub([comment("a", "risk:low")])
        api.on_mutation = lambda: setattr(api, "head", "b")
        self.assertIsNone(labels.reconcile(api, "org/repo", 1))
        self.assertEqual(api.labels, {"unrelated"})

    def test_new_review_during_cleanup_is_restored(self):
        api = FakeGitHub()
        def publish():
            api.comments.append(comment("a", "risk:high"))
            api.labels.add("risk:high")
        api.on_mutation = publish
        self.assertEqual(labels.reconcile(api, "org/repo", 1), "risk:high")
        self.assertEqual(api.labels, {"unrelated", "risk:high"})

    def test_new_assessment_on_same_head_replaces_old_one(self):
        api = FakeGitHub([comment("a", "risk:low")])
        api.on_mutation = lambda: api.comments.append(comment("a", "risk:medium", 2))
        self.assertEqual(labels.reconcile(api, "org/repo", 1), "risk:medium")
        self.assertEqual(api.labels, {"unrelated", "risk:medium"})

    def test_continuous_updates_stop_after_bounded_attempts(self):
        count = 0
        def unstable(*args):
            nonlocal count
            count += 1
            return str(count), None, None
        with patch.object(labels, "assessment", side_effect=unstable):
            with self.assertRaisesRegex(RuntimeError, "kept changing"):
                labels.reconcile(FakeGitHub(), "org/repo", 1)
        self.assertEqual(count, 6)

    def test_missing_label_is_the_only_ignored_http_error(self):
        for code in (0, 404, 403, 500, None):
            stdout = f"HTTP/2.0 {code} response\r\n" if code else ""
            stderr = "network failure" if code is None else "request failed"
            result = subprocess.CompletedProcess([], 0 if code == 0 else 1, stdout, stderr)
            with self.subTest(code=code), patch.object(labels.subprocess, "run", return_value=result):
                if code in (0, 404):
                    self.assertIsNone(labels.github("endpoint", method="DELETE", allow_missing=True))
                else:
                    with self.assertRaises(RuntimeError):
                        labels.github("endpoint", method="DELETE", allow_missing=True)


if __name__ == "__main__":
    unittest.main()
