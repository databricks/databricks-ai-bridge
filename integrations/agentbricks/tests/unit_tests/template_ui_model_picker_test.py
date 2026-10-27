"""Check the shipped chat UIs' model picker label."""

from pathlib import Path

import pytest

TEMPLATES = Path(__file__).parents[2] / "src/databricks_agentbricks/templates/ui"


@pytest.mark.parametrize("template", ["agent-openai", "agent-langgraph"])
def test_default_model_label_fits_the_picker(template):
    # The select is capped at 260px; "(project default)" clipped after the default model's name.
    source = (TEMPLATES / template / "ui/app.js").read_text()

    assert "`${name} (default)`" in source
    assert "(project default)" not in source
