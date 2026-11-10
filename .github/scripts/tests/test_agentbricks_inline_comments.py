"""Tests for posting AgentBricksReviewer findings as inline comments."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "agentbricks_inline_comments.py"
spec = importlib.util.spec_from_file_location("agentbricks_inline_comments", SCRIPT_PATH)
inline = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(inline)

HEAD = "64e2771916a02e8430034f13e20b16aa039930dc"
CODE = "src/databricks_ai_bridge/utils/vector_search.py"

DIFF = f"""diff --git a/{CODE} b/{CODE}
index 1111111..2222222 100644
--- a/{CODE}
+++ b/{CODE}
@@ -160,6 +160,10 @@ def parse_vector_search_response(
     for row in rows:
         score = row[-1]
-        keep = True
+        if min_score and score < min_score:
+            continue
+        keep = True
+++ counter = 1
         results.append(row)
diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-a
-b
"""

HIGH = (
    f"[HIGH] [HIGH CONFIDENCE] `{CODE}:163` — Passing `min_score=0.0` with a negative-scored row "
    "→ the truthiness check disables filtering. Use `if min_score is not None and score < min_score:`."
)
REVIEW = f"""**Review score:** 90/100 · reviewers: 2/2 · findings by lens: correctness (1)

## Blocking issues
- {HIGH}
  Verified-against: `{CODE}:78-173`.

## Security vulnerabilities
None.

## Non-blocking notes
- [LOW] [MEDIUM CONFIDENCE] `docs/index.md:4` — not in the diff → stays in the summary.
- [LOW] [HIGH CONFIDENCE] `{CODE}:161-164` — range → spans lines.

## Uncertain findings
None.

## Approach
Sound. See `{CODE}:163`.

## Summary
### Changes
A change.
"""


class CommentableLinesTest(unittest.TestCase):
    def test_added_and_context_lines_count_but_deleted_lines_do_not(self):
        lines = inline.commentable_lines(DIFF)
        self.assertEqual(lines[CODE], set(range(160, 167)))
        self.assertNotIn("old.py", lines)

    def test_added_line_that_looks_like_a_header_stays_in_the_file(self):
        self.assertIn(165, inline.commentable_lines(DIFF)[CODE])


class PlanTest(unittest.TestCase):
    def test_only_findings_in_finding_sections_on_diff_lines_are_anchored(self):
        comments, anchored = inline.plan(REVIEW, DIFF, HEAD)
        self.assertEqual([(c["path"], c["line"]) for c in comments], [(CODE, 163), (CODE, 164)])
        self.assertEqual([f["section"] for f in anchored], ["Blocking issues", "Non-blocking notes"])

    def test_ranges_become_multi_line_comments(self):
        comments, _ = inline.plan(REVIEW, DIFF, HEAD)
        self.assertEqual(comments[1]["start_line"], 161)
        self.assertEqual(comments[1]["start_side"], "RIGHT")

    def test_comment_body_has_markers_tags_and_continuation_lines(self):
        body = inline.plan(REVIEW, DIFF, HEAD)[0][0]["body"]
        self.assertIn(inline.INLINE_MARKER, body)
        self.assertIn("**[HIGH] [HIGH CONFIDENCE]** · Blocking issues", body)
        self.assertIn(f"Verified-against: `{CODE}:78-173`.", body)

    def test_lines_outside_the_diff_fall_back_to_a_single_line(self):
        review = f"## Non-blocking notes\n- [LOW] [HIGH CONFIDENCE] `{CODE}:150-163` — x → y.\n"
        comments, _ = inline.plan(review, DIFF, HEAD)
        self.assertEqual([(c["line"], "start_line" in c) for c in comments], [(163, False)])

    def test_bold_section_labels_are_recognized(self):
        review = f"**Blocking issues**\n- [HIGH] [HIGH CONFIDENCE] `{CODE}:163` — x → y.\n"
        self.assertEqual(len(inline.plan(review, DIFF, HEAD)[0]), 1)


class RewriteTest(unittest.TestCase):
    def test_posted_findings_become_one_count_line_per_section(self):
        _, anchored = inline.plan(REVIEW, DIFF, HEAD)
        summary = inline.rewrite(REVIEW, anchored)
        self.assertIn("## Blocking issues\n- Left 1 inline comment on the changed lines.\n\n## Security", summary)
        self.assertIn(
            "## Non-blocking notes\n- [LOW] [MEDIUM CONFIDENCE] `docs/index.md:4` — not in the diff → stays in the summary.\n"
            "- Left 1 inline comment on the changed lines.\n",
            summary,
        )
        self.assertNotIn("Verified-against", summary)
        self.assertNotIn("Passing `min_score=0.0`", summary)
        self.assertTrue(summary.endswith("A change.\n"))

    def test_count_is_plural_for_several_findings(self):
        review = (
            "## Blocking issues\n"
            f"- [HIGH] [HIGH CONFIDENCE] `{CODE}:162` — a → b.\n"
            f"- [HIGH] [HIGH CONFIDENCE] `{CODE}:163` — c → d.\n"
        )
        _, anchored = inline.plan(review, DIFF, HEAD)
        self.assertEqual(inline.rewrite(review, anchored), "## Blocking issues\n- Left 2 inline comments on the changed lines.\n")

    def test_nothing_posted_leaves_the_review_unchanged(self):
        self.assertEqual(inline.rewrite(REVIEW, []), REVIEW)



def earlier(path=CODE, line=163, section="Blocking issues", outdated=False, minimized=False, id="C1"):
    return {"id": id, "path": path, "line": line, "section": section, "outdated": outdated, "minimized": minimized}


class ReconcileTest(unittest.TestCase):
    def test_same_finding_at_same_line_is_kept_not_reposted(self):
        comments, _ = inline.plan(REVIEW, DIFF, HEAD)
        to_post, keep, stale = inline.reconcile(comments, [earlier()])
        self.assertEqual([(c["path"], c["line"]) for c in to_post], [(CODE, 164)])
        self.assertEqual([c["id"] for c in keep], ["C1"])
        self.assertEqual(stale, [])

    def test_gone_moved_outdated_or_other_section_comments_are_minimized(self):
        comments, _ = inline.plan(REVIEW, DIFF, HEAD)
        olds = [
            earlier(line=150, id="moved"),
            earlier(outdated=True, id="outdated"),
            earlier(line=None, id="no-line"),
            earlier(section="Non-blocking notes", id="other-section"),
            earlier(section=None, id="old-format"),
        ]
        to_post, keep, stale = inline.reconcile(comments, olds)
        self.assertEqual(len(to_post), 2)
        self.assertEqual(keep, [])
        self.assertEqual([c["id"] for c in stale], ["moved", "outdated", "no-line", "other-section", "old-format"])

    def test_already_minimized_comments_are_not_minimized_again(self):
        _, keep, stale = inline.reconcile([], [earlier(minimized=True)])
        self.assertEqual((keep, stale), ([], []))

    def test_a_returning_finding_keeps_its_minimized_comment_for_restoring(self):
        comments, _ = inline.plan(REVIEW, DIFF, HEAD)
        _, keep, _ = inline.reconcile(comments, [earlier(minimized=True)])
        self.assertTrue(keep[0]["minimized"])

    def test_two_findings_on_one_line_each_need_their_own_comment(self):
        review = (
            "## Blocking issues\n"
            f"- [HIGH] [HIGH CONFIDENCE] `{CODE}:163` — a → b.\n"
            f"- [HIGH] [HIGH CONFIDENCE] `{CODE}:163` — c → d.\n"
        )
        comments, _ = inline.plan(review, DIFF, HEAD)
        to_post, keep, _ = inline.reconcile(comments, [earlier()])
        self.assertEqual((len(to_post), len(keep)), (1, 1))

    def test_comment_bodies_carry_the_section_marker(self):
        comments, _ = inline.plan(REVIEW, DIFF, HEAD)
        self.assertIn("<!-- agentbricks-reviewer-finding: Blocking issues -->", comments[0]["body"])
        self.assertEqual(inline.FINDING_MARKER.search(comments[0]["body"])["section"], "Blocking issues")


if __name__ == "__main__":
    unittest.main()
