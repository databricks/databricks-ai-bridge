"""Exercise the workflow's real label-deletion blocks with GitHub CLI responses."""

import os
from pathlib import Path
import subprocess
import tempfile
import textwrap
import unittest


class LabelCleanupTest(unittest.TestCase):
    def run_cleanup(self, step, response, exit_code):
        workflow = (Path(__file__).resolve().parents[2] / "workflows/agentbricks-reviewer.yml").read_text()
        section = workflow.split(f"      - name: {step}\n", 1)[1]
        section = section.split("      - name:", 1)[0]
        script = textwrap.dedent(section.split("        run: |\n", 1)[1])
        # Execute the actual deletion loop without the unrelated label creation/lookups.
        marker = "for old in risk:low" if step == "Label reviewed change risk" else "for label in risk:low"
        script = "set -euo pipefail\n" + script[script.index(marker):]
        with tempfile.TemporaryDirectory() as directory:
            gh = Path(directory) / "gh"
            gh.write_text('#!/bin/bash\nprintf "%s\\n" "$TEST_RESPONSE"\nexit "$TEST_EXIT"\n')
            gh.chmod(0o755)
            env = dict(
                os.environ, PATH=directory + os.pathsep + os.environ["PATH"],
                TEST_RESPONSE=response, TEST_EXIT=str(exit_code),
                REPO="org/repo", PR_NUMBER="1", label="risk:low",
            )
            return subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True)

    def test_success_and_missing_label_are_allowed(self):
        for step in ("Label reviewed change risk", "Clear risk labels for unreviewed commits"):
            for response, code in (("HTTP/2.0 200 OK", 0), ("HTTP/2.0 404 Not Found", 1)):
                with self.subTest(step=step, response=response):
                    result = self.run_cleanup(step, response, code)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_permission_server_and_network_errors_are_reported(self):
        for step in ("Label reviewed change risk", "Clear risk labels for unreviewed commits"):
            for response in (
                "HTTP/2.0 403 Forbidden", "HTTP/2.0 500 Internal Server Error",
                "error connecting to api.github.com",
            ):
                with self.subTest(step=step, response=response):
                    result = self.run_cleanup(step, response, 1)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(response, result.stderr)


if __name__ == "__main__":
    unittest.main()
