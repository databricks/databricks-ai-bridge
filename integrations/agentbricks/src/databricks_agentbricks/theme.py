"""Semantic color palette for the Agent Bricks CLI.

The palette and — more importantly — the color *rules*. Color carries meaning, never decoration:

    green   success only — the ✔ glyph, success boxes, the prompt caret
    blue    links / URLs only
    cyan    the actionable command — next-step commands, hint code
    amber   warnings
    red     errors
    purple  LLM activity in the run log; the `mcp` tool-kind badge
    steel   the INFO log tag — deliberately *not* green, so it reads as calm/routine

Because URLs are blue, the actionable command in a hint/next-step is cyan, not blue, so the two
never compete.

Terminal reality: the palette hexes are tuned for a dark terminal (a #0d1117 background). A
terminal is not a web page — we cannot set the background, and a forced light-grey
default foreground (`--t-fg`) would vanish on a light terminal. So we leave *default* text
uncolored (the terminal's own foreground) and assign only the semantic accents and the mid-tone
greys, which read on either background. `rich` downsamples these truecolor hexes on limited
terminals and honors NO_COLOR / non-TTY on its own.
"""

from __future__ import annotations

from rich.theme import Theme

# --- palette ------------------------------------------------------------------
# `--t-fg` (#cdd5df) is intentionally omitted: default text uses the terminal's own foreground.
SECONDARY = "#949ca5"  # command descriptions, section labels (SETUP/DEVELOP/SHIP)
DIM = "#6b7580"  # secondary metadata — timestamps, key labels, help preamble, prompt path
FAINT = "#4d565f"  # tertiary — tree branches, input placeholder
GREEN = "#56d497"  # success
BLUE = "#5b69ef"  # links / URLs, and the `note:` severity keyword
CYAN = "#56d0d6"  # the actionable command
AMBER = "#e3b341"  # warning
RED = "#ff7b72"  # error
PURPLE = "#c9a2ff"  # LLM activity, the mcp tool-kind badge
STEEL = "#7e97b8"  # the INFO log-level tag


def rgb(hex_color: str) -> tuple[int, int, int]:
    """Parse a ``#rrggbb`` string into an ``(r, g, b)`` tuple.

    ``rich`` takes the hex strings directly, but ``click.style`` (used to color ``--help`` output)
    wants a truecolor tuple — this bridges the two so both share one palette definition.
    """
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


# rich Theme so styles are referenced by semantic name (`[agentbricks.error]…`) rather than raw hex,
# keeping the color *rules* in one place.
AGENTBRICKS_THEME = Theme(
    {
        "agentbricks.secondary": SECONDARY,
        "agentbricks.dim": DIM,
        "agentbricks.faint": FAINT,
        "agentbricks.success": GREEN,
        "agentbricks.link": BLUE,
        "agentbricks.command": CYAN,
        "agentbricks.warning": AMBER,
        "agentbricks.error": RED,
        "agentbricks.llm": PURPLE,
        "agentbricks.info": STEEL,
        # Severity keywords: `error:` red, `warning:` amber, `note:` blue.
        "agentbricks.sev.error": f"bold {RED}",
        "agentbricks.sev.warning": f"bold {AMBER}",
        "agentbricks.sev.note": f"bold {BLUE}",
    },
    inherit=True,
)
