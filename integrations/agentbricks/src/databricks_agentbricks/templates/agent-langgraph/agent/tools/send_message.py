"""A side-effecting sample tool, gated by human approval.

Unlike ``get_current_time`` (a harmless read), this tool stands in for an action with real
consequences — sending a message. ``agent/agent.py`` lists it in ``REQUIRE_APPROVAL``, so before it
runs the agent pauses and asks a human to approve / edit / reject the call. Swap the body for a real
send; the approval gate is what the template is demonstrating.
"""

from langchain_core.tools import tool


@tool
def send_message(recipient: str, body: str) -> str:
    """Send a message to a recipient. Use when the user asks to notify or message someone."""
    # A real implementation would call an email/Slack/SMS API here. The point of this template is the
    # human-approval gate in front of it (see REQUIRE_APPROVAL in agent/agent.py), not the delivery.
    return f"Message sent to {recipient}: {body}"
