"""A side-effecting sample tool, gated by human approval.

Unlike ``get_current_time`` (a harmless read), this tool stands in for an action with real
consequences — sending a message. ``needs_approval=True`` makes the Agents SDK pause before it runs
and surface a pending approval; the runtime relays that as an ``interrupt`` event and resumes on an
approve/reject decision (its name is also listed in ``REQUIRE_APPROVAL`` in ``agent/agent.py``). Swap
the body for a real send; the approval gate is what the template is demonstrating.
"""

from agents import function_tool


@function_tool(needs_approval=True)
def send_message(recipient: str, body: str) -> str:
    """Send a message to a recipient. Use when the user asks to notify or message someone."""
    # A real implementation would call an email/Slack/SMS API here. The point of this template is the
    # human-approval gate in front of it (needs_approval=True), not the delivery.
    return f"Message sent to {recipient}: {body}"
