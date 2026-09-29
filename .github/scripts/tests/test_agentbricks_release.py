"""Tests for the Agent Bricks release planner and metadata stamper."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "agentbricks_release.py"
spec = importlib.util.spec_from_file_location("agentbricks_release", SCRIPT_PATH)
release = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(release)


class ReleasePlanTest(unittest.TestCase):
    def setUp(self) -> None:
        self.main_sha = "a" * 40
        self.release_sha = "b" * 40

    def test_first_cut_uses_main_and_prepares_next_minor_development(self) -> None:
        self.assertEqual(
            release.plan_release("0.4.0rc1", self.main_sha, "0.4.0.dev0"),
            {
                "version": "0.4.0rc1",
                "branch": "release/databricks-agentbricks/v0.4",
                "tag": "databricks-agentbricks-v0.4.0rc1",
                "base_sha": self.main_sha,
                "branch_exists": False,
                "next_dev_version": "0.5.0.dev0",
            },
        )

    def test_existing_minor_branch_is_reused_for_patch(self) -> None:
        result = release.plan_release("0.4.1", self.main_sha, "0.4.0", self.release_sha)
        self.assertEqual(result["base_sha"], self.release_sha)
        self.assertTrue(result["branch_exists"])
        self.assertEqual(result["next_dev_version"], "0.5.0.dev0")
        self.assertEqual(result["branch"], "release/databricks-agentbricks/v0.4")

    def test_existing_release_version_can_be_planned_again(self) -> None:
        result = release.plan_release("0.4.0", self.main_sha, "0.4.0", self.release_sha)
        self.assertEqual(result["tag"], "databricks-agentbricks-v0.4.0")
        self.assertEqual(result["next_dev_version"], "0.5.0.dev0")

    def test_first_cut_of_patch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "patch releases require"):
            release.plan_release("0.4.1", self.main_sha, "0.4.0")

    def test_regressive_or_cross_minor_releases_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "regress"):
            release.plan_release("0.4.0rc1", self.main_sha, "0.4.0")
        with self.assertRaisesRegex(ValueError, "share the target major and minor"):
            release.plan_release("0.5.0", self.main_sha, "0.4.0", self.release_sha)

    def test_malformed_versions_and_shas_are_rejected(self) -> None:
        for version in ("1.2", "v1.2.3", "1.2.3.dev0", "1.02.3", "1.2.3rc"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "unsupported"):
                release.plan_release(version, self.main_sha, "0.3.0")
        with self.assertRaisesRegex(ValueError, "40-character commit SHA"):
            release.plan_release("0.4.0", "main", "0.4.0.dev0")

    def test_cli_prints_json(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "plan", "--version", "0.4.0",
             "--base-sha", self.main_sha, "--current-version", "0.4.0"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(json.loads(result.stdout)["next_dev_version"], "0.5.0.dev0")


class StampTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.package = self.root / release.PACKAGE_PATH
        self.package.parent.mkdir(parents=True)
        self.package.write_text(
            '[project]\nname = "databricks-agentbricks"\nversion = "0.3.0"  # current\n'
            'dependencies = ["databricks-ai-bridge>=0.22.0"]\n\n'
            '[tool.example]\nversion = "9.9.9"\n'
        )
        self.templates = []
        for name in release.TEMPLATE_NAMES:
            path = self.root / release.TEMPLATE_BASE_PATH / name / "pyproject.toml"
            path.parent.mkdir(parents=True)
            extra = "[langgraph]" if "langgraph" in name and not name.startswith("custom") else (
                "[openai]" if "openai" in name and not name.startswith("custom") else ""
            )
            path.write_text(
                '[project]\nversion = "0.1.0"\n'
                f'dependencies = ["databricks-agentbricks{extra}>=0.3.0", "rich>=13.7"]\n'
            )
            self.templates.append(path)

    def test_release_stamp_updates_package_and_four_template_minimums(self) -> None:
        release.stamp_release(self.root, "0.4.0", templates=True)
        self.assertIn('version = "0.4.0"  # current', self.package.read_text())
        self.assertIn('version = "9.9.9"', self.package.read_text())
        self.assertIn('"databricks-ai-bridge>=0.22.0"', self.package.read_text())
        for path in self.templates:
            self.assertIn('"databricks-agentbricks', path.read_text())
            self.assertIn('>=0.4.0"', path.read_text())
            self.assertIn('version = "0.1.0"', path.read_text())
            self.assertIn('"rich>=13.7"', path.read_text())

    def test_repeated_stamp_keeps_exact_content(self) -> None:
        release.stamp_release(self.root, "0.4.0rc1", templates=True)
        expected = {path: path.read_bytes() for path in (self.package, *self.templates)}
        release.stamp_release(self.root, "0.4.0rc1", templates=True)
        self.assertEqual(expected, {path: path.read_bytes() for path in expected})

    def test_development_stamp_changes_package_only(self) -> None:
        template_contents = [path.read_bytes() for path in self.templates]
        release.stamp_release(self.root, "0.5.0.dev0")
        self.assertIn('version = "0.5.0.dev0"', self.package.read_text())
        self.assertEqual(template_contents, [path.read_bytes() for path in self.templates])

    def test_failed_template_validation_leaves_every_file_untouched(self) -> None:
        self.templates[-1].write_text('[project]\nname = "missing-dependency"\n')
        previous = self.package.read_bytes()
        with self.assertRaisesRegex(ValueError, "exactly one Agent Bricks dependency"):
            release.stamp_release(self.root, "0.4.0", templates=True)
        self.assertEqual(self.package.read_bytes(), previous)


if __name__ == "__main__":
    unittest.main()
