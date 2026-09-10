"""A side-effecting sample tool, gated by human approval.

Unlike ``get_current_time`` (a harmless read), this stands in for an action with real consequences.
Its name is listed in ``REQUIRE_APPROVAL`` in ``agent/agent.py``: when the model calls it the agent
stops before running it and emits an ``interrupt``, resuming on an approve/reject decision. Swap the
body for a real send; the approval gate is what the template is demonstrating.
"""

from anthropic import beta_tool


@beta_tool
def send_message(recipient: str, body: str) -> str:
    """Send a message to a recipient. Use when the user asks to notify or message someone.

    Args:
        recipient: Who to send the message to.
        body: The message body.
    """
    return f"Message sent to {recipient}: {body}"
