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
    change_safety=100,
    change_risk="LOW",
):
    return f"""**Review score:** {score}/100 · reviewers: {reviewers} · findings by lens: none

**Change safety:** {change_safety}/100 ({change_risk} RISK)
**Risk rationale:** Small compatible change with focused tests and straightforward rollback.

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


CODE = "integrations/agentbricks/src/databricks_agentbricks/utils/formatting.py"
TEST = "integrations/agentbricks/tests/unit_tests/agent_hint_test.py"


class DecideTest(unittest.TestCase):
    def assert_refused(self, result, fragment):
        self.assertFalse(result["approve"], result)
        self.assertTrue(any(fragment in reason for reason in result["reasons"]), result["reasons"])

    def test_clean_small_review_is_approved(self):
        result = gate.decide(review(), diff("README.md", CODE, TEST))
        self.assertTrue(result["approve"], result)
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["risk"]["label"], "risk:low")

    def test_non_blocking_notes_do_not_block_approval(self):
        notes = "- [LOW] [HIGH CONFIDENCE] `README.md:3` — typo → readers stumble."
        self.assertTrue(gate.decide(review(score=99, notes=notes), diff("README.md"))["approve"])

    def test_low_score_is_refused(self):
        self.assert_refused(gate.decide(review(score=94), diff("README.md")), "below 95")

    def test_review_score_is_separate_from_risk_wording(self):
        for old_line in (
            "**Merge safety:** 100/100 (LOW RISK)",
            "**Review score:** 100/100 (LOW RISK)",
        ):
            with self.subTest(old_line=old_line):
                text = review().replace("**Review score:** 100/100", old_line)
                self.assert_refused(gate.decide(text, diff("README.md")), "no review score line")

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
        self.assert_refused(gate.decide("## Blocking issues\n\nNone.\n", diff(CODE)), "no review score line")

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


class ChangeRiskTest(unittest.TestCase):
    def test_findings_score_does_not_replace_change_risk(self):
        result = gate.decide(review(change_safety=70, change_risk="MEDIUM"), diff(CODE))
        self.assertFalse(result["approve"])
        self.assertEqual(result["risk"]["label"], "risk:medium")
        self.assertEqual(result["risk"]["safety_score"], 70)

    def test_risk_bands_at_boundaries(self):
        for score, level in [(100, "low"), (90, "low"), (89, "medium"), (60, "medium"), (59, "high"), (0, "high")]:
            with self.subTest(score=score):
                result = gate.decide(review(change_safety=score, change_risk=level.upper()), diff(CODE))
                self.assertEqual(result["risk"]["level"], level)
                self.assertEqual(result["approve"], level == "low")

    def test_missing_or_invalid_assessment_withholds_approval(self):
        for text in (
            review().replace("**Change safety:** 100/100 (LOW RISK)", ""),
            review(change_safety=101),
            review(change_safety=100, change_risk="HIGH"),
            review() + "\n**Change safety:** 100/100 (LOW RISK)\n",
        ):
            with self.subTest(text=text):
                result = gate.decide(text, diff(CODE))
                self.assertFalse(result["approve"])
                self.assertEqual(result["risk"]["label"], "risk:high")
                self.assertIsNone(result["risk"]["safety_score"])

    def test_public_interface_and_runtime_changes_have_medium_floor(self):
        for path in (
            "integrations/agentbricks/src/databricks_agentbricks/cli/app.py",
            "integrations/agentbricks/src/databricks_agentkit/runtime/client.py",
            "integrations/agentbricks/src/databricks_agentbricks/__init__.py",
            "integrations/agentbricks/src/databricks_agentbricks/clients/managed_runtime_store.py",
            "integrations/agentbricks/src/databricks_agentbricks/clients/conversation_store_client.py",
        ):
            with self.subTest(path=path):
                result = gate.decide(review(), diff(path))
                self.assertFalse(result["approve"])
                self.assertEqual(result["risk"]["safety_score"], 89)
                self.assertEqual(result["risk"]["model_safety_score"], 100)

    def test_sensitive_paths_have_high_floor(self):
        for path in (
            "REVIEW.md", "integrations/agentbricks/REVIEW.md", "uv.lock",
            ".github/workflows/main.yml",
            "integrations/agentbricks/src/auth/helper.py",
            "integrations/agentbricks/src/migrations/update.py",
        ):
            with self.subTest(path=path):
                result = gate.decide(review(), diff(path))
                self.assertFalse(result["approve"])
                self.assertEqual(result["risk"]["safety_score"], 59)
                self.assertEqual(result["risk"]["label"], "risk:high")

    def test_cross_package_production_changes_have_medium_floor(self):
        result = gate.decide(review(), diff(CODE, "integrations/openai/src/helper.py"))
        self.assertEqual(result["risk"]["label"], "risk:medium")
        self.assertTrue(any("multiple packages" in reason for reason in result["risk"]["reasons"]))

    def test_docs_and_tests_in_multiple_packages_can_remain_low_risk(self):
        result = gate.decide(review(), diff("integrations/openai/README.md", "integrations/agentbricks/cli.md"))
        self.assertTrue(result["approve"], result)
        result = gate.decide(review(), diff(TEST))
        self.assertTrue(result["approve"], result)

    def test_size_and_binary_changes_raise_risk(self):
        for content, expected in (
            (diff(CODE, added=1001), "high"),
            (diff(*[f"docs/{i}.md" for i in range(11)]), "medium"),
            (diff(*[f"docs/{i}.md" for i in range(31)]), "high"),
            (diff(CODE) + "Binary files a/image.png and b/image.png differ\n", "high"),
        ):
            with self.subTest(expected=expected):
                self.assertEqual(gate.decide(review(), content)["risk"]["level"], expected)

    def test_line_limit_controls_approval_without_changing_risk(self):
        content = diff(CODE, added=201)
        limited = gate.decide(review(), content)
        self.assertEqual(limited["risk"]["level"], "low")
        self.assertFalse(limited["approve"])
        self.assertIn("201 changed lines is over the 200-line limit", limited["reasons"])
        relaxed = gate.decide(review(), content, max_lines=300)
        self.assertTrue(relaxed["approve"], relaxed)
        self.assertEqual(relaxed["risk"], limited["risk"])

    def test_canonical_changed_paths_drive_scope_and_risk(self):
        paths = ["integrations/agentbricks/src/cli/new command.py"]
        result = gate.decide(review(), diff("docs/example.md"), paths=paths)
        self.assertEqual(result["risk"]["level"], "medium")
        self.assertFalse(result["approve"])

    def test_low_risk_does_not_override_correctness_findings(self):
        result = gate.decide(review(score=75, blocking="A verified bug."), diff("README.md"))
        self.assertEqual(result["risk"]["label"], "risk:low")
        self.assertFalse(result["approve"])

    def test_safety_section_explains_assessment_and_eligibility(self):
        text = review()
        decision = gate.decide(text, diff("integrations/agentbricks/src/cli/app.py"))
        section = gate.safety_section(text, decision)
        self.assertIn("### Safety assessment", section)
        self.assertIn("Review score (findings and coverage) | 100/100", section)
        self.assertIn("Change safety (impact of the change) | 89/100", section)
        self.assertIn("Small compatible change", section)
        self.assertIn("Model change safety before CI adjustments: 100/100", section)
        self.assertIn("Auto-approval content checks:** Not eligible.", section)
        self.assertIn("public interface, runtime, or persisted state", section)

    def test_safety_section_distinguishes_content_checks_from_final_approval(self):
        text = review()
        section = gate.safety_section(text, gate.decide(text, diff("README.md")))
        self.assertIn("Auto-approval content checks:** Passed.", section)
        self.assertIn("Final approval also requires automation to be enabled", section)


if __name__ == "__main__":
    unittest.main()
