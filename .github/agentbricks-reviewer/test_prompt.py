"""Exercise the workflow's actual prompt builder without credentials or models."""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PromptTest(unittest.TestCase):
    def test_trusted_team_rules_injected_once(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github/workflows/agentbricks-reviewer.yml").read_text()
        start = workflow.index("          import hashlib, json, os, pathlib\n")
        end = workflow.index("\n          PYEOF", start)
        builder = "\n".join(line[10:] for line in workflow[start:end].splitlines())
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)

            class ScratchPaths(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if isinstance(node.value, str) and node.value.startswith("/tmp/"):
                        return ast.copy_location(ast.Constant(str(scratch / node.value[5:])), node)
                    return node

            tree = ast.fix_missing_locations(ScratchPaths().visit(ast.parse(builder)))
            meta = {
                "title": "Synthetic review", "headRefName": "fixture", "baseRefName": "main",
                "baseRefOid": "1" * 40, "headRefOid": "2" * 40, "additions": 1,
                "deletions": 0, "changedFiles": 1, "labels": [], "body": "Fixture change",
            }
            (scratch / "pr_meta.json").write_text(json.dumps(meta))
            (scratch / "pr_diff.txt").write_text("diff --git a/example.py b/example.py\n")
            (scratch / "pr_diff_summary.json").write_text(json.dumps({"markdown": "One file.", "test_files": []}))
            (scratch / "lockfile_pins.txt").write_text("")
            (scratch / "linked_issues.json").write_text(json.dumps({"issues": []}))
            with patch.dict(os.environ, {"GITHUB_WORKSPACE": str(root)}):
                exec(compile(tree, "workflow-prompt-builder", "exec"), {})
            prompt = (scratch / "review_prompt.txt").read_text()
            self.assertEqual(prompt.count((root / "REVIEW.md").read_text()), 1)
            self.assertIn('<team_review_rules source="REVIEW.md">', prompt)
            self.assertIn("BRIDGE-001", prompt)
            self.assertIn("BRIDGE-010", prompt)
            self.assertNotIn("name: agentbricks-review", prompt)
            self.assertIn("# Part 1: Coordinator process", prompt)


if __name__ == "__main__":
    unittest.main()
