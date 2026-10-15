"""Tests for the Agent Bricks release planner, metadata stamper, and release notes."""

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
            release.plan_release("0.4.0", self.main_sha, "0.4.0.dev0"),
            {
                "version": "0.4.0",
                "branch": "release/databricks-agentbricks/v0.4",
                "tag": "databricks-agentbricks-v0.4.0",
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
            release.plan_release("0.3.0", self.main_sha, "0.4.0")
        with self.assertRaisesRegex(ValueError, "share the target major and minor"):
            release.plan_release("0.5.0", self.main_sha, "0.4.0", self.release_sha)

    def test_malformed_versions_and_shas_are_rejected(self) -> None:
        for version in ("1.2", "v1.2.3", "1.2.3.dev0", "1.02.3", "1.2.3rc1", "1.2.3RC1"):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, "unsupported"):
                release.plan_release(version, self.main_sha, "0.3.0")
        with self.assertRaisesRegex(ValueError, "40-character commit SHA"):
            release.plan_release("0.4.0", "main", "0.4.0.dev0")

    def test_rc_metadata_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported"):
            release.plan_release("0.4.0", self.main_sha, "0.4.0rc1")
        with self.assertRaisesRegex(ValueError, "unsupported"):
            release.parse_version("0.4.0rc1", allow_dev=True)

    def test_cli_prints_json(self) -> None:
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "plan", "--version", "0.4.0",
             "--base-sha", self.main_sha, "--base-version", "0.4.0"],
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
        for name in (
            "agent-langgraph", "agent-openai", "custom-agent-langgraph", "custom-agent-openai"
        ):
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
        release.stamp_release(self.root, "0.4.0", templates=True)
        expected = {path: path.read_bytes() for path in (self.package, *self.templates)}
        release.stamp_release(self.root, "0.4.0", templates=True)
        self.assertEqual(expected, {path: path.read_bytes() for path in expected})

    def test_new_template_receives_release_version(self) -> None:
        path = self.root / release.TEMPLATE_BASE_PATH / "another-framework" / "pyproject.toml"
        path.parent.mkdir(parents=True)
        path.write_text('[project]\ndependencies = ["databricks-agentbricks>=0.3.0"]\n')
        release.stamp_release(self.root, "0.4.0", templates=True)
        self.assertIn('"databricks-agentbricks>=0.4.0"', path.read_text())

    def test_missing_templates_fail_before_package_write(self) -> None:
        for path in self.templates:
            path.unlink()
        previous = self.package.read_bytes()
        with self.assertRaisesRegex(ValueError, "expected Agent Bricks template pyproject.toml"):
            release.stamp_release(self.root, "0.4.0", templates=True)
        self.assertEqual(self.package.read_bytes(), previous)

    def test_cli_stamp_accepts_target_version(self) -> None:
        subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "stamp", "--root", str(self.root),
             "--target-version", "0.4.0", "--templates"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn('version = "0.4.0"', self.package.read_text())

    def test_development_stamp_changes_package_only(self) -> None:
        template_contents = [path.read_bytes() for path in self.templates]
        release.stamp_release(self.root, "0.5.0.dev0")
        self.assertIn('version = "0.5.0.dev0"', self.package.read_text())
        self.assertEqual(template_contents, [path.read_bytes() for path in self.templates])

    def test_rc_and_template_development_stamps_are_rejected(self) -> None:
        invalid_stamps = (("0.4.0rc1", False), ("0.4.0rc1", True), ("0.5.0.dev0", True))
        for version, templates in invalid_stamps:
            with self.subTest(version=version, templates=templates):
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    release.stamp_release(self.root, version, templates=templates)
        self.assertIn('version = "0.3.0"', self.package.read_text())

    def test_failed_template_validation_leaves_every_file_untouched(self) -> None:
        self.templates[-1].write_text('[project]\nname = "missing-dependency"\n')
        previous = self.package.read_bytes()
        with self.assertRaisesRegex(ValueError, "exactly one Agent Bricks dependency"):
            release.stamp_release(self.root, "0.4.0", templates=True)
        self.assertEqual(self.package.read_bytes(), previous)


class ReleaseNotesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scratch = tempfile.TemporaryDirectory()
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        self.git("init", "--quiet", "--initial-branch=main")
        self.git("config", "user.name", "Test")
        self.git("config", "user.email", "test@example.com")
        self.git("config", "commit.gpgsign", "false")

    def git(self, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=self.root, check=True, capture_output=True, text=True
        ).stdout.strip()

    def commit(self, path: str, subject: str) -> str:
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(subject)
        self.git("add", path)
        self.git("commit", "--quiet", "-m", subject)
        return self.git("rev-parse", "HEAD")

    def test_previous_tag_is_the_highest_release_below_the_target(self) -> None:
        tags = [
            "databricks-agentbricks-v0.4.0", "databricks-agentbricks-v0.4.1",
            "databricks-agentbricks-v0.5.0", "databricks-agentbricks-v0.6.0",
            "databricks-agentbricks-vnext", "databricks-langchain-v0.9.0",
        ]
        self.assertEqual(release.previous_release_tag("0.5.0", tags), "databricks-agentbricks-v0.4.1")
        self.assertEqual(release.previous_release_tag("0.5.1", tags), "databricks-agentbricks-v0.5.0")
        self.assertIsNone(release.previous_release_tag("0.4.0", tags))

    def test_notes_list_package_changes_since_previous_tag(self) -> None:
        self.commit("integrations/agentbricks/a.py", "Shipped before (#1)")
        self.git("tag", "databricks-agentbricks-v0.4.0")
        self.commit("integrations/agentbricks/b.py", "[agentbricks] Add feature (#2)")
        self.commit("integrations/langchain/c.py", "Unrelated package change (#3)")
        self.commit("integrations/agentbricks/pyproject.toml", "Release Agent Bricks 0.5.0")
        head = self.git("rev-parse", "HEAD")
        notes = release.release_notes(self.root, "0.5.0", head, "owner/repo")
        self.assertIn("- [agentbricks] Add feature ([#2](https://github.com/owner/repo/pull/2))", notes)
        self.assertNotIn("Shipped before", notes)
        self.assertNotIn("Unrelated package change", notes)
        self.assertNotIn("Release Agent Bricks", notes)
        self.assertIn(
            "https://github.com/owner/repo/compare/databricks-agentbricks-v0.4.0...databricks-agentbricks-v0.5.0",
            notes,
        )

    def test_notes_skip_fixes_already_cherry_picked_into_the_previous_release(self) -> None:
        self.commit("integrations/agentbricks/a.py", "Base")
        self.git("switch", "--quiet", "-c", "release")
        self.git("switch", "--quiet", "main")
        fix = self.commit("integrations/agentbricks/fix.py", "Fix crash (#4)")
        self.commit("integrations/agentbricks/new.py", "New in minor (#5)")
        self.git("switch", "--quiet", "release")
        self.git("cherry-pick", fix)
        self.git("tag", "databricks-agentbricks-v0.4.1")
        self.git("switch", "--quiet", "main")
        notes = release.release_notes(self.root, "0.5.0", self.git("rev-parse", "HEAD"), "owner/repo")
        self.assertIn("New in minor ([#5](https://github.com/owner/repo/pull/5))", notes)
        self.assertNotIn("Fix crash", notes)

    def test_first_release_lists_history_without_a_compare_link(self) -> None:
        head = self.commit("integrations/agentbricks/a.py", "Initial package (#1)")
        notes = release.release_notes(self.root, "0.1.0", head, "owner/repo")
        self.assertIn("- Initial package ([#1](https://github.com/owner/repo/pull/1))", notes)
        self.assertNotIn("compare", notes)

    def test_every_pull_request_reference_is_linked(self) -> None:
        notes = release.render_release_notes(
            "0.5.0", None, ["Fix kwargs (#426, #429)", "Keep owner/repo#7 and C# as written"], "owner/repo"
        )
        self.assertIn(
            "Fix kwargs ([#426](https://github.com/owner/repo/pull/426), "
            "[#429](https://github.com/owner/repo/pull/429))",
            notes,
        )
        self.assertIn("Keep owner/repo#7 and C# as written", notes)

    def test_release_without_package_changes_says_so(self) -> None:
        self.commit("integrations/agentbricks/a.py", "Base")
        self.git("tag", "databricks-agentbricks-v0.4.0")
        head = self.commit("integrations/agentbricks/pyproject.toml", "Release Agent Bricks 0.4.1")
        notes = release.release_notes(self.root, "0.4.1", head, "owner/repo")
        self.assertIn("No package changes", notes)

    def test_cli_prints_notes(self) -> None:
        head = self.commit("integrations/agentbricks/a.py", "Initial package (#1)")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "notes", "--version", "0.1.0", "--release-sha", head,
             "--root", str(self.root), "--repository", "owner/repo"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("## databricks-agentbricks 0.1.0", result.stdout)


if __name__ == "__main__":
    unittest.main()
