"""Exercise the workflow's actual prompt builder without credentials or models."""

import ast
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PromptTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).resolve().parents[2]
        self.root = root
        self.workflow = (root / ".github/workflows/agentbricks-reviewer.yml").read_text()
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
            skill = workspace / ".claude/skills/agentbricks-review/SKILL.md"
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
        self.assertNotIn("name: agentbricks-review", prompt)
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


class ChangedPathsTest(unittest.TestCase):
    def test_shallow_diverged_pr_includes_both_rename_scopes(self):
        root = Path(__file__).resolve().parents[2]
        workflow = (root / ".github/workflows/agentbricks-reviewer.yml").read_text()
        fetch = next(
            line.strip() for line in workflow.splitlines()
            if line.strip().startswith("git fetch --quiet --no-tags --depth=1 origin")
        )
        start = workflow.index("          import json, pathlib, subprocess, sys\n")
        end = workflow.index("\n          PYCHANGED", start)
        collector = "\n".join(line[10:] for line in workflow[start:end].splitlines())

        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            source = scratch / "source"
            source.mkdir()
            env = dict(
                os.environ, GIT_AUTHOR_NAME="Test", GIT_AUTHOR_EMAIL="test@example.com",
                GIT_COMMITTER_NAME="Test", GIT_COMMITTER_EMAIL="test@example.com",
            )

            def git(*args, cwd=source):
                return subprocess.check_output(
                    ["git", *args], cwd=cwd, env=env, stderr=subprocess.PIPE, text=True
                ).strip()

            git("init", "-q", "-b", "main")
            old = "packages/old name.py"
            new = "integrations/agentbricks/new name.py"
            (source / old).parent.mkdir(parents=True)
            (source / old).write_text("unchanged content\n")
            git("add", ".")
            git("commit", "-qm", "shared ancestor")
            merge_base = git("rev-parse", "HEAD")
            git("checkout", "-qb", "feature")
            (source / new).parent.mkdir(parents=True)
            git("mv", old, new)
            git("commit", "-qm", "move between scopes")
            head = git("rev-parse", "HEAD")
            git("checkout", "-q", "main")
            (source / "main-only.txt").write_text("unrelated base change\n")
            git("add", ".")
            git("commit", "-qm", "advance main")
            base = git("rev-parse", "HEAD")
            checkout = scratch / "checkout"
            git("clone", "-q", "--depth=1", source.as_uri(), str(checkout))
            subprocess.run(
                ["bash", "-c", fetch], cwd=checkout, check=True,
                env=dict(env, base_sha=base, head_sha=head, merge_base_sha=merge_base),
                capture_output=True, text=True,
            )
            self.assertEqual(git("rev-parse", "--is-shallow-repository", cwd=checkout), "true")
            output = scratch / "changed.json"
            collector = collector.replace('"/tmp/pr_changed_paths.json"', repr(str(output)))
            original_cwd = Path.cwd()
            try:
                os.chdir(checkout)
                with patch.object(sys, "argv", ["collector", merge_base, head]):
                    exec(compile(collector, "workflow-changed-paths", "exec"), {})
            finally:
                os.chdir(original_cwd)
            self.assertEqual(set(json.loads(output.read_text())), {old, new})


if __name__ == "__main__":
    unittest.main()
