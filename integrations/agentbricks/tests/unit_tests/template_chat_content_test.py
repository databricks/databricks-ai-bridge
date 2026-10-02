"""Run the browser's actual text/Markdown normalization against typed response fixtures."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Browser renderer tests require Node.js")
@pytest.mark.parametrize("script", ["chat-content.test.cjs", "background-polling.test.cjs"])
def test_chat_content_rendering(script):
    test = Path(__file__).parents[1] / "ui" / script
    subprocess.run(["node", "--test", str(test)], check=True, capture_output=True, text=True)
