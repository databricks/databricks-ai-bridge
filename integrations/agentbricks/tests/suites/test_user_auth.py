"""Request-user auth: a deployed App calls `auth = "user"` tools as the user who invoked it."""

from __future__ import annotations

import json
import re

from harness.agentbricks_cli import AgentbricksCli, excerpt, tool_results
from harness.cleanup import Cleanup
from harness.project import Project
from harness.workspace import Workspace
from tool_definitions import web_search_tool

# A user-owned tool reporting which identity each of the request's credentials resolves to.
# Tools reach request-user credentials only through `workspace_client_for`, which the template
# passes to its MCP and Genie tools but not to `all_tools()`, so the test registers this one itself.
IDENTITY_FILE = "agent/request_identity.py"
IDENTITY_SOURCE = '''from langchain_core.tools import tool


def request_identity_tools(workspace_client_for):
    if workspace_client_for is None:
        return []

    @tool
    def request_identity() -> str:
        """Return the identities this request's user and app credentials resolve to."""
        user = workspace_client_for("user").current_user.me()
        app = workspace_client_for("app").current_user.me()
        return f"AGENTBRICKS_IDENTITY user={user.user_name} app={app.user_name}"

    return [request_identity]
'''
IDENTITY_RESULT = re.compile(r"AGENTBRICKS_IDENTITY user=(\S+) app=(\S+)")
IDENTITY_PROMPT = "You must call the request_identity tool. Return its exact result."


def test_user_auth(
    agentbricks_cli: AgentbricksCli,
    project: Project,
    cleanup: Cleanup,
    workspace_client: Workspace,
    subtests,
) -> None:
    agentbricks_cli.run(f"init --framework langgraph {project.path}", timeout=600)
    project.pin_runtime()
    app = project.app_name
    # Before deploy: a deploy that fails partway can still leave the App behind.
    cleanup(f"deployments delete {app} --delete-source --yes")
    web_search = web_search_tool(auth="user")
    agentbricks_cli.run(f"tools add {web_search.add_command} --source {project.path}")
    project.write_file(IDENTITY_FILE, IDENTITY_SOURCE)
    _register_identity_tools(project)

    deployed = json.loads(
        agentbricks_cli.run(
            f"--output json deploy {app} --source {project.path}", timeout=2400
        ).stdout
    )
    cleanup(f"memory stores delete memory-stores/{deployed['memory_store']} --yes")
    cleanup(f"sessions stores delete {deployed['session_store']} --yes")

    with subtests.test(check="user tool"):
        web_search.assert_called(agentbricks_cli.invoke(app, web_search.prompt))
    with subtests.test(check="identity"):
        output = agentbricks_cli.invoke(app, IDENTITY_PROMPT)
        identities = [
            match.groups()
            for result in tool_results(output)
            if (match := IDENTITY_RESULT.search(str(result.get("content"))))
        ]
        assert identities, f"No request_identity result: {excerpt(output)}"
        user, app_identity = identities[0]
        assert user == workspace_client.user_name, excerpt(output)
        assert app_identity != user, excerpt(output)


def _register_identity_tools(project: Project) -> None:
    """Add the identity tools next to `all_tools()` in the generated agent."""
    agent = project.path / "agent" / "agent.py"
    source = agent.read_text(encoding="utf-8")
    import_line = "from agent.tools import all_tools\n"
    tools_line = "        *all_tools(),\n"
    assert import_line in source and tools_line in source, "the template's agent.py changed"
    source = source.replace(
        import_line, import_line + "from agent.request_identity import request_identity_tools\n"
    ).replace(tools_line, tools_line + "        *request_identity_tools(workspace_client_for),\n")
    project.write_file("agent/agent.py", source)
