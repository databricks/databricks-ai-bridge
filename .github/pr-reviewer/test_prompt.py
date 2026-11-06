"""Exercise the workflow's actual prompt builder without credentials or models."""

import ast
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PromptTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.root = root
        self.workflow = (root / ".github/workflows/pr-reviewer.yml").read_text()
        start = self.workflow.index(
            "          import hashlib, html, json, os, pathlib\n"
        )
        end = self.workflow.index("\n          PYEOF", start)
        builder = "\n".join(
            line[10:] for line in self.workflow[start:end].splitlines()
        )
        self.tree = ast.parse(builder)

    def build_prompt(self, changed_paths, workspace=None):
        workspace = workspace or self.root
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)

            class ScratchPaths(ast.NodeTransformer):
                def visit_Constant(self, node):
                    if isinstance(node.value, str) and node.value.startswith("/tmp/"):
                        return ast.copy_location(ast.Constant(str(scratch / node.value[5:])), node)
                    return node

            tree = ast.fix_missing_locations(ScratchPaths().visit(copy.deepcopy(self.tree)))
            meta = {
                "title": "Synthetic review", "headRefName": "fixture", "baseRefName": "main",
                "baseRefOid": "1" * 40, "headRefOid": "2" * 40, "additions": 1,
                "deletions": 0, "changedFiles": 1, "labels": [], "body": "Fixture change",
            }
            (scratch / "pr_meta.json").write_text(json.dumps(meta))
            (scratch / "pr_diff.txt").write_text("diff --git a/example.py b/example.py\n")
            (scratch / "pr_diff_summary.json").write_text(
                json.dumps({"markdown": "One file.", "test_files": []})
            )
            (scratch / "pr_changed_paths.json").write_text(json.dumps(changed_paths))
            (scratch / "lockfile_pins.txt").write_text("")
            (scratch / "linked_issues.json").write_text(json.dumps({"issues": []}))
            with patch.dict(os.environ, {"GITHUB_WORKSPACE": str(workspace)}):
                exec(compile(tree, "workflow-prompt-builder", "exec"), {})
            return (scratch / "review_prompt.txt").read_text()

    def test_arbitrary_root_and_scoped_rules_are_injected_once(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            skill = workspace / ".claude/skills/pr-review/SKILL.md"
            skill.parent.mkdir(parents=True)
            skill.write_text((self.root / skill.relative_to(workspace)).read_text())
            root_rules = "# General rules\n\nGENERAL-001\n"
            scoped_rules = "# Example team rules\n\nEXAMPLE-001\n"
            (workspace / "REVIEW.md").write_text(root_rules)
            scoped = workspace / "packages/example/REVIEW.md"
            scoped.parent.mkdir(parents=True)
            scoped.write_text(scoped_rules)

            prompt = self.build_prompt(
                ["packages/example/src/feature.py"], workspace=workspace
            )

        self.assertEqual(prompt.count(root_rules), 1)
        self.assertEqual(prompt.count(scoped_rules), 1)
        self.assertLess(prompt.index(root_rules), prompt.index(scoped_rules))
        self.assertIn('<review_rules source="REVIEW.md">', prompt)
        self.assertIn('<review_rules source="packages/example/REVIEW.md">', prompt)
        self.assertNotIn("name: pr-review", prompt)
        self.assertIn("# Part 1: Coordinator process", prompt)

    def test_agentbricks_change_loads_its_automation_policy(self):
        prompt = self.build_prompt(["integrations/agentbricks/src/example.py"])
        self.assertIn("AGENTBRICKS-003", prompt)
        self.assertIn("Automated review is enabled for this scope", prompt)

    def test_unrelated_change_loads_only_root_rules(self):
        prompt = self.build_prompt(["integrations/openai/example.py", "../outside.py"])
        self.assertEqual(prompt.count((self.root / "REVIEW.md").read_text()), 1)
        self.assertNotIn("AGENTBRICKS-001", prompt)
        self.assertNotIn("integrations/agentbricks/REVIEW.md", prompt)

    def test_workflow_does_not_hardcode_the_agentbricks_rule_file(self):
        self.assertNotIn('"integrations/agentbricks/REVIEW.md"', self.workflow)


if __name__ == "__main__":
    unittest.main()
