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
        self.assertIn("<!-- agentbricks-reviewer-finding: 0 -->", body)
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
    def test_posted_findings_shrink_to_links_and_others_stay(self):
        _, anchored = inline.plan(REVIEW, DIFF, HEAD)
        summary = inline.rewrite(REVIEW, anchored, {0: "https://example/c0", 1: "https://example/c1"})
        self.assertIn(
            f"- [HIGH] [HIGH CONFIDENCE] `{CODE}:163` — Passing `min_score=0.0` with a negative-scored row"
            " → [inline comment](https://example/c0)",
            summary,
        )
        self.assertNotIn("Verified-against", summary)
        self.assertIn("`docs/index.md:4` — not in the diff → stays in the summary.", summary)
        self.assertIn("## Security vulnerabilities\nNone.", summary)
        self.assertTrue(summary.endswith("A change.\n"))

    def test_findings_without_a_posted_url_stay_in_full(self):
        _, anchored = inline.plan(REVIEW, DIFF, HEAD)
        self.assertEqual(inline.rewrite(REVIEW, anchored, {}), REVIEW)

    def test_short_form_keeps_code_spans_balanced(self):
        text = "[LOW] `" + "x" * 200 + "` → y"
        self.assertEqual(inline.short_form(text).count("`") % 2, 0)


if __name__ == "__main__":
    unittest.main()
