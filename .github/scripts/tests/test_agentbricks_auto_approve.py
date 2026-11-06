"""Tests for PRReviewer's AgentBricks auto-approval gate."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "agentbricks_auto_approve.py"
spec = importlib.util.spec_from_file_location("agentbricks_auto_approve", SCRIPT_PATH)
gate = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(gate)


def review(
    score=100,
    reviewers="2/2",
    blocking="None.",
    security="None.",
    uncertain="None.",
    notes="None.",
):
    return f"""**Merge safety:** {score}/100 (LOW RISK) · reviewers: {reviewers} · findings by lens: none

## Blocking issues

{blocking}

## Security vulnerabilities

{security}

## Non-blocking notes

{notes}

## Uncertain findings

{uncertain}

## Approach

Sound.

## Summary

### Changes
A change.
"""


def diff(*paths, added=1):
    hunks = []
    for path in paths:
        body = "".join(f"+line {i}\n" for i in range(added))
        hunks.append(
            f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,{added} @@\n{body}"
        )
    return "".join(hunks)


class DecideTest(unittest.TestCase):
    def assert_refused(self, result, fragment):
        self.assertFalse(result["approve"], result)
        self.assertTrue(any(fragment in reason for reason in result["reasons"]), result["reasons"])

    def test_clean_small_review_is_approved(self):
        result = gate.decide(review(), diff("integrations/agentbricks/README.md"))
        self.assertEqual(
            result,
            {"score": 100, "risk_label": "review-risk: low", "approve": True, "reasons": []},
        )

    def test_non_blocking_notes_do_not_block_approval(self):
        notes = "- [LOW] [HIGH CONFIDENCE] `README.md:3` — typo → readers stumble."
        self.assertTrue(
            gate.decide(review(score=99, notes=notes), diff("integrations/agentbricks/README.md"))["approve"]
        )

    def test_low_score_is_refused(self):
        self.assert_refused(
            gate.decide(review(score=94), diff("integrations/agentbricks/README.md")), "below 95"
        )

    def test_missing_reviewer_is_refused(self):
        self.assert_refused(
            gate.decide(review(reviewers="1/2"), diff("integrations/agentbricks/README.md")), "1 of 2"
        )

    def test_single_reviewer_is_refused(self):
        self.assert_refused(
            gate.decide(review(reviewers="1/1"), diff("integrations/agentbricks/README.md")), "at least 2"
        )

    def test_any_blocking_security_or_uncertain_finding_is_refused(self):
        finding = "- [High] [HIGH CONFIDENCE] `a.py:1` — x → y."
        for field, title in (
            ("blocking", "Blocking issues"),
            ("security", "Security vulnerabilities"),
            ("uncertain", "Uncertain findings"),
        ):
            with self.subTest(title):
                self.assert_refused(
                    gate.decide(review(**{field: finding}), diff("integrations/agentbricks/a.py")),
                    f'"{title}" is not empty',
                )

    def test_missing_section_or_summary_line_is_refused(self):
        text = review().replace("## Security vulnerabilities\n\nNone.\n", "")
        self.assert_refused(
            gate.decide(text, diff("integrations/agentbricks/a.py")),
            'no "Security vulnerabilities" section',
        )
        self.assert_refused(
            gate.decide("## Blocking issues\n\nNone.\n", diff("integrations/agentbricks/a.py")),
            "no merge safety line",
        )

    def test_large_diff_is_refused(self):
        self.assert_refused(
            gate.decide(review(), diff("integrations/agentbricks/a.py", added=201)),
            "over the 200-line limit",
        )
        self.assertTrue(
            gate.decide(review(), diff("integrations/agentbricks/a.py", added=200))["approve"]
        )

    def test_risk_label_copies_model_score_band(self):
        path = "integrations/agentbricks/a.py"
        for score, label in (
            (100, "review-risk: low"),
            (90, "review-risk: low"),
            (89, "review-risk: medium"),
            (70, "review-risk: medium"),
            (69, "review-risk: high"),
            (0, "review-risk: high"),
        ):
            with self.subTest(score):
                self.assertEqual(gate.decide(review(score=score), diff(path))["risk_label"], label)

    def test_auto_approval_requires_an_agentbricks_only_diff(self):
        result = gate.decide(
            review(),
            diff("integrations/agentbricks/a.py", "integrations/openai/b.py"),
        )
        self.assert_refused(result, "outside integrations/agentbricks/**")

    def test_protected_paths_are_refused(self):
        for path, fragment in (
            (".github/workflows/pr-reviewer.yml", "CI or review configuration"),
            (".claude/skills/pr-review/SKILL.md", "CI or review configuration"),
            ("integrations/agentbricks/pyproject.toml", "dependency manifest"),
            ("integrations/agentbricks/requirements-dev.txt", "dependency manifest"),
            ("integrations/agentbricks/src/credential.py", "credential or auth"),
            ("integrations/agentbricks/src/oauth.py", "credential or auth"),
            ("integrations/agentbricks/src/auth.py", "credential or auth"),
        ):
            with self.subTest(path):
                self.assert_refused(gate.decide(review(), diff(path)), fragment)

    def test_names_that_only_contain_auth_are_allowed(self):
        self.assertTrue(
            gate.decide(
                review(),
                diff(
                    "integrations/agentbricks/docs/authors.md",
                    "integrations/agentbricks/src/author_utils.py",
                ),
            )["approve"]
        )

    def test_renamed_path_counts_both_names(self):
        rename = (
            "diff --git a/integrations/agentbricks/a.py b/integrations/openai/a.py\n"
            "similarity index 100%\n"
        )
        self.assert_refused(gate.decide(review(), rename), "outside integrations/agentbricks/**")

    def test_empty_diff_is_refused(self):
        self.assert_refused(gate.decide(review(), ""), "changes no files")


if __name__ == "__main__":
    unittest.main()
