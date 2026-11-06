"""Tests for AgentBricksReviewer's auto-approval gate."""

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


CODE = "integrations/agentbricks/src/databricks_agentbricks/cli/app.py"
TEST = "integrations/agentbricks/tests/unit_tests/agent_hint_test.py"


class DecideTest(unittest.TestCase):
    def assert_refused(self, result, fragment):
        self.assertFalse(result["approve"], result)
        self.assertTrue(any(fragment in reason for reason in result["reasons"]), result["reasons"])

    def test_clean_small_review_is_approved(self):
        result = gate.decide(review(), diff("README.md", CODE, TEST))
        self.assertEqual(result, {"approve": True, "reasons": []})

    def test_non_blocking_notes_do_not_block_approval(self):
        notes = "- [LOW] [HIGH CONFIDENCE] `README.md:3` — typo → readers stumble."
        self.assertTrue(gate.decide(review(score=99, notes=notes), diff("README.md"))["approve"])

    def test_low_score_is_refused(self):
        self.assert_refused(gate.decide(review(score=94), diff("README.md")), "below 95")

    def test_missing_reviewer_is_refused(self):
        self.assert_refused(gate.decide(review(reviewers="1/2"), diff("README.md")), "1 of 2")

    def test_single_reviewer_is_refused(self):
        self.assert_refused(gate.decide(review(reviewers="1/1"), diff("README.md")), "at least 2")

    def test_any_blocking_security_or_uncertain_finding_is_refused(self):
        finding = "- [High] [HIGH CONFIDENCE] `a.py:1` — x → y."
        for field, title in (
            ("blocking", "Blocking issues"),
            ("security", "Security vulnerabilities"),
            ("uncertain", "Uncertain findings"),
        ):
            with self.subTest(title):
                self.assert_refused(gate.decide(review(**{field: finding}), diff(CODE)), f'"{title}" is not empty')

    def test_missing_section_or_summary_line_is_refused(self):
        text = review().replace("## Security vulnerabilities\n\nNone.\n", "")
        self.assert_refused(gate.decide(text, diff(CODE)), 'no "Security vulnerabilities" section')
        self.assert_refused(gate.decide("## Blocking issues\n\nNone.\n", diff(CODE)), "no merge safety line")

    def test_large_diff_is_refused(self):
        self.assert_refused(gate.decide(review(), diff(CODE, added=201)), "over the 200-line limit")
        self.assertTrue(gate.decide(review(), diff(CODE, added=200))["approve"])

    def test_protected_paths_are_refused(self):
        for path, fragment in (
            (".github/workflows/agentbricks-reviewer.yml", "CI or review configuration"),
            (".claude/skills/agentbricks-review/SKILL.md", "CI or review configuration"),
            ("REVIEW.md", "CI or review configuration"),
            ("integrations/agentbricks/REVIEW.md", "CI or review configuration"),
            ("integrations/openai/REVIEW.md", "CI or review configuration"),
            ("integrations/openai/pyproject.toml", "dependency manifest"),
            ("uv.lock", "dependency manifest"),
            ("integrations/langchainjs/package-lock.json", "dependency manifest"),
            ("requirements-dev.txt", "dependency manifest"),
            ("src/databricks_ai_bridge/model_serving_obo_credential_strategy.py", "credential or auth"),
            ("databricks_mcp/src/databricks_mcp/oauth.py", "credential or auth"),
            ("integrations/agentbricks/src/databricks_agentbricks/auth.py", "credential or auth"),
            ("integrations/agentbricks/pyproject.toml", "dependency manifest"),
        ):
            with self.subTest(path):
                self.assert_refused(gate.decide(review(), diff(path)), fragment)

    def test_names_that_only_contain_auth_are_allowed(self):
        self.assertTrue(gate.decide(review(), diff("docs/authors.md", "integrations/agentbricks/src/databricks_agentbricks/author_utils.py"))["approve"])

    def test_other_guidance_docs_remain_eligible(self):
        for path in (
            "AGENTS.md",
            "CLAUDE.md",
            "integrations/agentbricks/AGENTS.md",
            "integrations/agentbricks/CLAUDE.md",
            ".agents/skills/agentbricks-review/SKILL.md",
        ):
            with self.subTest(path):
                self.assertTrue(gate.decide(review(), diff(path))["approve"])

    def test_changes_outside_agentbricks_are_refused_unless_docs(self):
        for path in (
            "src/databricks_ai_bridge/utils/vector_search.py",
            "tests/databricks_ai_bridge/utils/test_vector_search.py",
            "integrations/openai/src/databricks_openai/utils/clients.py",
            "docs/source/conf.py",
            "integrations/agentbricksx/a.py",
        ):
            with self.subTest(path):
                self.assert_refused(gate.decide(review(), diff(path)), "outside integrations/agentbricks/")
        docs = diff("README.md", "integrations/openai/README.md", "docs/source/index.rst", "CONTRIBUTING.md")
        self.assertTrue(gate.decide(review(), docs)["approve"])

    def test_renamed_path_counts_both_names(self):
        rename = "diff --git a/integrations/agentbricks/a.py b/.github/a.py\nsimilarity index 100%\n"
        self.assert_refused(gate.decide(review(), rename), "CI or review configuration")
        moved_out = "diff --git a/integrations/agentbricks/a.py b/src/a.py\nsimilarity index 100%\n"
        self.assert_refused(gate.decide(review(), moved_out), "outside integrations/agentbricks/")

    def test_empty_diff_is_refused(self):
        self.assert_refused(gate.decide(review(), ""), "changes no files")


if __name__ == "__main__":
    unittest.main()
