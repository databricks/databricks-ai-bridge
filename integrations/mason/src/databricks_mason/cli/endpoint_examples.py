"""Copy-pasteable examples for invoking Mason-generated agent endpoints."""

from __future__ import annotations


def agent_invoke_command(target: str, *, uses_runtime_api: bool) -> str:
    """Build a narrow, shell-safe command that fits in Mason's success panel."""
    path = "/api/invocations" if uses_runtime_api else "/invocations"
    lines = ["INVOCATION_ID=$(uuidgen)"] if uses_runtime_api else []
    lines.extend(
        [
            "mason endpoint invoke \\",
            f"  {target} \\",
            f"  --path {path} \\",
            '  --json "{',
        ]
    )
    if uses_runtime_api:
        lines.append('    \\"id\\":\\"$INVOCATION_ID\\",')
    lines.extend(
        [
            '    \\"input\\":[{',
            '      \\"role\\":\\"user\\",',
            '      \\"content\\":\\"hi\\"',
            "    }]",
            '  }"',
        ]
    )
    return "\n".join(lines)
