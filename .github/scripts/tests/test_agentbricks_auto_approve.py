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
    change_risk="LOW",
):
    return f"""**Review score:** {score}/100 · reviewers: {reviewers} · findings by lens: none

**Change risk:** {change_risk}
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
        result = gate.decide(review(reviewers="1/2"), diff("README.md"))
        self.assert_refused(result, "1 of 2")
        self.assertEqual(result["risk"]["level"], "low")

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

    def test_bold_section_labels_count_as_sections(self):
        text = review()
        for title in ("Blocking issues", "Security vulnerabilities", "Non-blocking notes", "Uncertain findings", "Approach"):
            text = text.replace(f"## {title}\n", f"**{title}**\n")
        self.assertTrue(gate.decide(text, diff(CODE))["approve"])
        finding = "- [HIGH] [HIGH CONFIDENCE] `a.py:1` — x → y."
        flagged = review(blocking=finding).replace("## Blocking issues\n", "**Blocking issues:**\n")
        self.assert_refused(gate.decide(flagged, diff(CODE)), '"Blocking issues" is not empty')

    def test_none_with_a_caveat_is_empty_but_a_listed_finding_is_not(self):
        caveat = "None. The UI label couldn't be confirmed offline; this is a verification limit, not a defect."
        self.assertTrue(gate.decide(review(uncertain=caveat), diff(CODE))["approve"])
        for body in ("None.\n- [MEDIUM] [UNCERTAIN] `a.py:1` — x → y.", "1. [LOW] `a.py:1` — x → y.", "Nothing serious, but x → y."):
            with self.subTest(body=body):
                self.assert_refused(gate.decide(review(uncertain=body), diff(CODE)), '"Uncertain findings" is not empty')

    def test_large_diff_is_refused(self):
        self.assert_refused(gate.decide(review(), diff(CODE, added=501)), "over the 500-line limit")
        self.assertTrue(gate.decide(review(), diff(CODE, added=500))["approve"])

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
    def test_moderate_risk_can_pass_with_clean_findings_and_score(self):
        result = gate.decide(review(change_risk="MODERATE"), diff(CODE))
        self.assertTrue(result["approve"])
        self.assertEqual(result["risk"]["label"], "risk:medium")

    def test_risk_classifications(self):
        for name, level in (("LOW", "low"), ("MODERATE", "medium"), ("HIGH", "high"), ("Unrated", "unrated")):
            with self.subTest(name=name):
                result = gate.decide(review(change_risk=name), diff(CODE))
                self.assertEqual(result["risk"]["level"], level)
                self.assertTrue(result["approve"], result)

    def test_high_risk_does_not_override_other_approval_gates(self):
        for kwargs in (
            {"score": 94}, {"reviewers": "1/2"},
            {"blocking": "A bug."}, {"security": "A vulnerability."},
            {"uncertain": "An unverified finding."},
        ):
            with self.subTest(kwargs=kwargs):
                result = gate.decide(review(change_risk="HIGH", **kwargs), diff(CODE))
                self.assertFalse(result["approve"], result)
                self.assertEqual(result["risk"]["label"], "risk:high")
        for content in (diff("REVIEW.md"), diff(CODE, added=501)):
            with self.subTest(content=content):
                result = gate.decide(review(change_risk="HIGH"), content)
                self.assertFalse(result["approve"], result)

    def test_missing_or_invalid_assessment_is_advisory(self):
        for text in (
            review().replace("**Change risk:** LOW", ""),
            review(change_risk="Unrated"),
            review(change_risk="CRITICAL"),
            review(change_risk="100/100 (LOW RISK)"),
            review() + "\n**Change risk:** LOW\n",
            review() + "\n**Change risk:** UNKNOWN\n",
            review().replace("**Change risk:** LOW", "**Change safety:** 100/100 (LOW RISK)"),
        ):
            with self.subTest(text=text):
                result = gate.decide(text, diff(CODE))
                self.assertTrue(result["approve"], result)
                self.assertEqual(result["risk"]["label"], "risk:unrated")

    def test_missing_or_duplicate_rationale_is_advisory(self):
        for text in (
            review().replace("**Risk rationale:** Small compatible change with focused tests and straightforward rollback.", ""),
            review().replace("Small compatible change with focused tests and straightforward rollback.", ""),
            review() + "\n**Risk rationale:** Another assessment.\n",
        ):
            with self.subTest(text=text):
                result = gate.decide(text, diff(CODE))
                self.assertTrue(result["approve"], result)
                self.assertEqual(result["risk"]["label"], "risk:unrated")

    def test_unrated_is_preserved_even_for_sensitive_paths(self):
        text = review().replace("**Change risk:** LOW", "")
        decision = gate.decide(text, diff("REVIEW.md"))
        self.assertFalse(decision["approve"])
        self.assertEqual(decision["risk"]["label"], "risk:unrated")
        self.assertIn("**Change risk:** UNRATED · `risk:unrated`", gate.overview(text, decision))
        self.assertTrue(decision["risk"]["reasons"])

    def test_high_assessment_is_not_downgraded_by_lower_floors(self):
        text = review(change_risk="HIGH")
        decision = gate.decide(text, diff(CODE))
        self.assertTrue(decision["approve"])
        self.assertEqual(decision["risk"]["label"], "risk:high")
        comment = gate.overview(text, decision)
        self.assertIn("**Change risk:** HIGH · `risk:high`", comment)
        self.assertIn("Small compatible change", comment)
        self.assertIn("eligible for auto-approval", comment)

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
                self.assertTrue(result["approve"], result)
                self.assertEqual(result["risk"]["level"], "medium")
                self.assertEqual(result["risk"]["model_level"], "low")

    def test_sensitive_paths_have_high_floor(self):
        for path, approved in (
            ("REVIEW.md", False), ("integrations/agentbricks/REVIEW.md", False),
            ("uv.lock", False), (".github/workflows/main.yml", False),
            ("integrations/agentbricks/src/auth/helper.py", True),
            ("integrations/agentbricks/src/migrations/update.py", True),
        ):
            with self.subTest(path=path):
                result = gate.decide(review(), diff(path))
                self.assertEqual(result["approve"], approved)
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
        content = diff(CODE, added=501)
        limited = gate.decide(review(), content)
        self.assertEqual(limited["risk"]["level"], "low")
        self.assertFalse(limited["approve"])
        self.assertIn("501 changed lines is over the 500-line limit", limited["reasons"])
        relaxed = gate.decide(review(), content, max_lines=600)
        self.assertTrue(relaxed["approve"], relaxed)
        self.assertEqual(relaxed["risk"], limited["risk"])

    def test_canonical_changed_paths_drive_scope_and_risk(self):
        paths = ["integrations/agentbricks/src/cli/new command.py"]
        result = gate.decide(review(), diff("docs/example.md"), paths=paths)
        self.assertEqual(result["risk"]["level"], "medium")
        self.assertTrue(result["approve"], result)

    def test_review_verdict_is_independent_of_risk_scope_and_size(self):
        for path, risk, added in ((CODE, "LOW", 1), (CODE, "MODERATE", 1),
                                 ("REVIEW.md", "LOW", 1), (CODE, "HIGH", 1),
                                 (CODE, "Unrated", 1), (CODE, "LOW", 501),
                                 ("integrations/openai/src/helper.py", "LOW", 1)):
            with self.subTest(path=path, risk=risk, added=added):
                result = gate.decide(review(change_risk=risk), diff(path, added=added))
                self.assertTrue(result["review_passed"], result)
                self.assertEqual(result["review_reasons"], [])
                self.assertEqual(result["approve"], path == CODE and added == 1)

    def test_review_failures_withhold_ai_label_verdict(self):
        for text in (review(score=94), review(score=101), review(reviewers="1/2"),
                     review(blocking="A bug."), review(security="A vulnerability."),
                     review(uncertain="Unverified finding."),
                     review().replace("## Security vulnerabilities", "## Missing section")):
            with self.subTest(text=text):
                result = gate.decide(text, diff(CODE))
                self.assertFalse(result["review_passed"])
                self.assertFalse(result["approve"])
                self.assertTrue(result["review_reasons"])

    def test_low_risk_does_not_override_correctness_findings(self):
        result = gate.decide(review(score=75, blocking="A verified bug."), diff("README.md"))
        self.assertEqual(result["risk"]["label"], "risk:low")
        self.assertFalse(result["approve"])

    def test_overview_has_score_risk_summary_and_approval(self):
        text = review()
        decision = gate.decide(text, diff("integrations/agentbricks/src/cli/app.py"))
        section = gate.overview(text, decision)
        self.assertTrue(section.startswith(
            "**Review score:** 100/100 · reviewers: 2/2 · **Change risk:** MODERATE · `risk:medium`\n"))
        self.assertEqual(section.count("/100"), 1)
        for heading in ("### Summary", "### Risk", "### AI approval"):
            self.assertIn(heading, section)
        self.assertIn("No findings.", section)
        self.assertIn("**Approach:** Sound.", section)
        self.assertEqual(section.count("Small compatible change"), 1)
        self.assertIn("CI raised the reviewer's LOW to MODERATE:", section)
        self.assertIn("public interface, runtime, or persisted state", section)
        self.assertIn("Review checks passed (`ai-approved`), and the PR is eligible for auto-approval.", section)
        self.assertNotIn("To unblock", section)
        self.assertNotIn("Safety assessment", section)

    def test_overview_lists_findings_and_how_to_fix_them(self):
        finding = "- [HIGH] [HIGH CONFIDENCE] `a.py:1` — x → y."
        text = review(score=80, reviewers="1/2", blocking=finding)
        section = gate.overview(text, gate.decide(text, diff(CODE)))
        self.assertIn(f"**Blocking issues**\n{finding}", section)
        self.assertNotIn("No findings.", section)
        self.assertIn("Review checks not passed;", section)
        self.assertIn("- Fix the blocking issues above, or reply explaining why they don't apply, "
                      "then comment `/review` (the score must reach 95).", section)
        self.assertIn("- A reviewer didn't finish: comment `/review force` to run the review again.", section)

    def test_overview_names_each_blocked_path_group_once(self):
        paths = [".github/workflows/a.yml", "src/databricks_ai_bridge/core.py"]
        text = review()
        section = gate.overview(text, gate.decide(text, diff(*paths, added=150), max_lines=200))
        self.assertIn("- Split the PR to at most 200 changed lines (it has 300).", section)
        self.assertEqual(section.count("`.github/workflows/a.yml`"), 1)
        self.assertIn("- CI or review configuration files can't be auto-approved; move them to a separate PR (see **Risk**).",
                      section)
        self.assertIn("- Files outside `integrations/agentbricks/` that aren't docs can't be auto-approved; "
                      "move them to a separate PR (`src/databricks_ai_bridge/core.py`).", section)

    def test_overview_says_when_the_pr_is_eligible(self):
        text = review()
        section = gate.overview(text, gate.decide(text, diff("README.md")))
        self.assertIn("Review checks passed (`ai-approved`), and the PR is eligible for auto-approval.", section)
        self.assertNotIn("To unblock", section)

    def test_long_path_lists_are_truncated(self):
        paths = [f"src/pkg/m{i}.py" for i in range(8)]
        section = gate.overview(review(), gate.decide(review(), diff(*paths)))
        self.assertIn("`src/pkg/m4.py`, and 3 more", section)


if __name__ == "__main__":
    unittest.main()
