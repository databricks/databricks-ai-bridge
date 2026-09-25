"""Offline contract tests for the governed-connection E2E harness."""

from __future__ import annotations

import ast
import json
import pathlib
import sys
from argparse import Namespace
from collections.abc import Callable
from typing import cast

import pytest

E2E_DIR = pathlib.Path(__file__).parents[1] / "e2e"
sys.path.insert(0, str(E2E_DIR.parent))

from e2e.connection_matrix import (  # noqa: E402
    DeploymentCase,
    Framework,
    MatrixCase,
    MatrixError,
    MatrixRunner,
    Response,
    assert_markers,
    cleanup_plan,
    connection_command,
    deployment_cases,
    execute_invocation,
    expect_denial,
    matrix_cases,
    parse_args,
    render_probe_files,
    resource_name,
    verify_evidence,
    write_evidence,
)


def test_matrix_is_exact_cartesian_product() -> None:
    cases = matrix_cases()

    assert len(cases) == 8
    assert len(set(cases)) == 8
    assert {case.setup for case in cases} == {"existing"}
    assert {case.transport for case in cases} == {"mcp", "http"}
    assert {case.framework for case in cases} == {"langgraph", "openai"}
    assert {case.execution for case in cases} == {"foreground", "background"}


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (
            MatrixCase("existing", "mcp", "langgraph", "foreground"),
            [
                "/venv/bin/ab",
                "--profile",
                "workspace",
                "auth",
                "connections",
                "bind",
                "existing-mcp",
                "--uc-connection",
                "main.agentbricks_e2e.mcp_fixture",
                "--transport",
                "mcp",
                "--principal",
                "app",
                "--source",
                "/tmp/project",
            ],
        ),
        (
            MatrixCase("existing", "http", "openai", "background"),
            [
                "/venv/bin/ab",
                "--profile",
                "workspace",
                "auth",
                "connections",
                "bind",
                "existing-http",
                "--uc-connection",
                "main.agentbricks_e2e.http_fixture",
                "--transport",
                "http",
                "--principal",
                "app",
                "--source",
                "/tmp/project",
            ],
        ),
    ],
)
def test_connection_command_only_binds_existing_bearer_connections(
    case: MatrixCase, expected: list[str]
) -> None:
    actual = connection_command(
        case,
        ab=pathlib.Path("/venv/bin/ab"),
        profile="workspace",
        project=pathlib.Path("/tmp/project"),
        aliases={"existing:mcp": "existing-mcp", "existing:http": "existing-http"},
        existing={
            "mcp": "main.agentbricks_e2e.mcp_fixture",
            "http": "main.agentbricks_e2e.http_fixture",
        },
    )

    assert actual == expected


def test_background_requires_202_then_polls_terminal_result() -> None:
    calls: list[tuple[str, str]] = []
    responses = iter(
        [
            Response(202, {"status": "queued", "status_url": "/api/invocations/run-1"}),
            Response(200, {"status": "running"}),
            Response(200, {"status": "active"}),
            Response(
                200,
                {
                    "status": "completed",
                    "output": {"provider_marker": "AGENTBRICKS_HTTP_OK", "user": "alice"},
                },
            ),
        ]
    )

    def request(method: str, url: str, body: dict | None) -> Response:
        calls.append((method, url))
        return next(responses)

    result = execute_invocation(
        "background",
        "https://agent.example.test/api/invocations",
        {"id": "run-1", "input": {}},
        request,
        expected_marker="AGENTBRICKS_HTTP_OK",
        expected_user_marker="alice",
        poll_interval=0,
    )

    assert result["status"] == "completed"
    assert calls == [
        ("POST", "https://agent.example.test/api/invocations"),
        ("GET", "https://agent.example.test/api/invocations/run-1"),
        ("GET", "https://agent.example.test/api/invocations/run-1"),
        ("GET", "https://agent.example.test/api/invocations/run-1"),
    ]


def test_foreground_requires_200_and_does_not_poll() -> None:
    calls: list[tuple[str, str]] = []

    def request(method: str, url: str, body: dict | None) -> Response:
        calls.append((method, url))
        return Response(
            200,
            {
                "status": "completed",
                "output": {"provider_marker": "AGENTBRICKS_MCP_OK", "user": "alice"},
            },
        )

    result = execute_invocation(
        "foreground",
        "https://agent.example.test/api/invocations",
        {"id": "run-2", "input": {}},
        request,
        expected_marker="AGENTBRICKS_MCP_OK",
        expected_user_marker="alice",
    )

    assert result["status"] == "completed"
    assert calls == [("POST", "https://agent.example.test/api/invocations")]


@pytest.mark.parametrize(
    "payload",
    [
        {"provider_marker": "wrong", "user": "alice"},
        {"provider_marker": "AGENTBRICKS_HTTP_OK", "user": "bob"},
    ],
)
def test_marker_assertion_rejects_wrong_provider_or_user(payload: dict[str, str]) -> None:
    with pytest.raises(MatrixError, match="marker"):
        assert_markers(payload, "AGENTBRICKS_HTTP_OK", "alice")


@pytest.mark.parametrize(
    "seeded_secret",
    [
        "SENTINEL-CLIENT-SECRET",
        "Authorization: Bearer SENTINEL-TOKEN",
        "Cookie: session=SENTINEL-COOKIE",
    ],
)
def test_evidence_redactor_rejects_seeded_secrets(
    tmp_path: pathlib.Path, seeded_secret: str
) -> None:
    with pytest.raises(MatrixError, match="sensitive"):
        write_evidence(tmp_path, rows=[], controls=[], cleanup=[], scanned_text=seeded_secret)
    assert not (tmp_path / "evidence.json").exists()


def test_evidence_write_is_atomic_valid_json(tmp_path: pathlib.Path) -> None:
    target = write_evidence(
        tmp_path,
        rows=[{"status": "pass", "case": "existing-http-langgraph-foreground"}],
        controls=[{"name": "unknown-alias", "status": "pass"}],
        cleanup=[{"kind": "app", "name": "agent-bricks-cx-run", "status": "deleted"}],
        scanned_text="fixed non-sensitive marker",
    )

    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["rows"][0]["status"] == "pass"
    assert document["controls"][0]["name"] == "unknown-alias"
    assert document["cleanup"][0]["status"] == "deleted"
    assert not (tmp_path / "evidence.json.tmp").exists()


def test_resource_names_are_stable_bounded_and_dimension_specific() -> None:
    foreground = MatrixCase("existing", "http", "openai", "foreground")
    background = MatrixCase("existing", "http", "openai", "background")

    assert resource_name("20260924T010203Z", foreground) == "cx-20260924t010203z-ex-http-oa"
    assert resource_name("20260924T010203Z", foreground) != resource_name(
        "20260924T010203Z", background, include_execution=True
    )
    assert len(resource_name("x" * 100, foreground)) <= 63


def test_parse_args_requires_complete_live_inputs(tmp_path: pathlib.Path) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")

    args = parse_args(
        [
            "--profile",
            "workspace",
            "--app-auth-profile",
            "workspace-oauth",
            "--wheel",
            str(wheel),
            "--output",
            str(tmp_path / "out"),
            "--existing-mcp-connection",
            "main.agentbricks_e2e.mcp_fixture",
            "--existing-http-connection",
            "main.agentbricks_e2e.http_fixture",
            "--mcp-marker",
            "AGENTBRICKS_MCP_OK",
            "--http-marker",
            "AGENTBRICKS_HTTP_OK",
            "--user-marker",
            "alice",
        ]
    )

    assert args.profile == "workspace"
    assert args.app_auth_profile == "workspace-oauth"
    assert args.existing_mcp_connection.endswith("mcp_fixture")


def test_parse_args_allows_evidence_verification_without_live_inputs(
    tmp_path: pathlib.Path,
) -> None:
    evidence = tmp_path / "evidence.json"
    evidence.write_text("{}")

    args = parse_args(["--verify-evidence", str(evidence)])

    assert args.verify_evidence == evidence


def test_wrong_background_submission_status_is_a_failure() -> None:
    with pytest.raises(MatrixError, match="202"):
        execute_invocation(
            "background",
            "https://agent.example.test/api/invocations",
            {"id": "run-3", "input": {}},
            lambda method, url, body: Response(200, {"status": "completed"}),
            expected_marker="AGENTBRICKS_HTTP_OK",
            expected_user_marker="alice",
        )


def test_negative_control_requires_sanitized_http_denial() -> None:
    expect_denial(Response(401, {"error": "authentication required"}), "missing-user-identity")
    expect_denial(Response(500, {"error": "connection request failed"}), "unknown-alias")

    with pytest.raises(MatrixError, match="denial"):
        expect_denial(Response(200, {"status": "completed"}), "unknown-alias")
    with pytest.raises(MatrixError, match="sensitive"):
        expect_denial(
            Response(500, {"error": "Authorization: Bearer SENTINEL-TOKEN"}),
            "forbidden-header",
        )


def test_deployment_cases_group_transport_and_execution_axes() -> None:
    deployments = deployment_cases()

    assert deployments == (
        DeploymentCase("existing", "langgraph"),
        DeploymentCase("existing", "openai"),
    )
    assert all(
        len(
            [
                case
                for case in matrix_cases()
                if case.setup == deployment.setup and case.framework == deployment.framework
            ]
        )
        == 4
        for deployment in deployments
    )


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_rendered_probe_files_are_valid_framework_code(framework: Framework) -> None:
    files = render_probe_files(
        framework,
        aliases={"mcp": "github", "http": "salesforce"},
        http_path="/agentbricks-e2e",
        mcp_tool="agentbricks_connection_probe",
    )

    assert set(files) == {"agent/tools/connection_probe.py", "runtime/adapter.py"}
    for relative_path, source in files.items():
        compile(source, relative_path, "exec")
    assert (
        "from databricks_agentkit.auth import context" in files["agent/tools/connection_probe.py"]
    )
    expected_decorator = "@tool" if framework == "langgraph" else "@function_tool"
    assert expected_decorator in files["agent/tools/connection_probe.py"]


@pytest.mark.parametrize("framework", ["langgraph", "openai"])
def test_rendered_jsonrpc_probe_uses_post_for_both_transports(framework: Framework) -> None:
    files = render_probe_files(
        framework,
        aliases={"mcp": "linear-mcp", "http": "linear-http"},
        http_path="/unused",
        mcp_tool="get_user",
        probe_mode="mcp-jsonrpc",
        mcp_arguments={"query": "me"},
    )

    source = files["agent/tools/connection_probe.py"]
    compile(source, "agent/tools/connection_probe.py", "exec")
    assert '"Accept": "application/json, text/event-stream"' in source
    assert '"method": "initialize"' in source
    assert '"method": "notifications/initialized"' in source
    assert '"method": "tools/call"' in source
    assert (
        source.index('"method": "initialize"')
        < source.index('"method": "notifications/initialized"')
        < source.index('"method": "tools/call"')
    )
    assert "\"name\": 'get_user'" in source
    assert "\"arguments\": {'query': 'me'}" in source
    assert 'client.request(\n            "POST",\n            ""' in source


def test_rendered_jsonrpc_probe_decodes_streamable_http_sse() -> None:
    source = render_probe_files(
        "langgraph",
        aliases={"mcp": "github", "http": "github"},
        http_path="/unused",
        mcp_tool="get_me",
        probe_mode="mcp-jsonrpc",
    )["agent/tools/connection_probe.py"]
    module = ast.parse(source)
    helper = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_decode_provider_response"
    )
    namespace: dict[str, object] = {"json": json}
    exec(
        compile(ast.Module(body=[helper], type_ignores=[]), "<generated-helper>", "exec"), namespace
    )
    decode = cast(Callable[[object], object], namespace["_decode_provider_response"])

    class Response:
        headers = {"content-type": "text/event-stream"}
        text = 'event: message\ndata: {"jsonrpc":"2.0","result":{"login":"junchoi-db"}}\n\n'

        def json(self):
            raise json.JSONDecodeError("SSE", self.text, 0)

    assert decode(Response()) == {
        "jsonrpc": "2.0",
        "result": {"login": "junchoi-db"},
    }

    class ProxyResponse:
        headers = {"content-type": "application/json,text/event-stream"}
        text = '{"jsonrpc":"2.0","result":{"login":"junchoi-db"}}'

        def json(self):
            return json.loads(self.text)

    assert decode(ProxyResponse()) == {
        "jsonrpc": "2.0",
        "result": {"login": "junchoi-db"},
    }


def test_parse_args_accepts_jsonrpc_probe_options(tmp_path: pathlib.Path) -> None:
    wheel = tmp_path / "agentbricks.whl"
    wheel.write_bytes(b"wheel")

    args = parse_args(
        [
            "--profile",
            "workspace",
            "--app-auth-profile",
            "workspace-oauth",
            "--wheel",
            str(wheel),
            "--output",
            str(tmp_path / "out"),
            "--existing-mcp-connection",
            "main.agentbricks_e2e.linear",
            "--existing-http-connection",
            "main.agentbricks_e2e.linear",
            "--mcp-marker",
            "user-id",
            "--http-marker",
            "user-id",
            "--user-marker",
            "Jun Choi",
            "--mcp-tool",
            "get_user",
            "--mcp-arguments",
            '{"query":"me"}',
            "--probe-mode",
            "mcp-jsonrpc",
            "--pause-before-invocations",
        ]
    )

    assert args.probe_mode == "mcp-jsonrpc"
    assert args.mcp_arguments == {"query": "me"}
    assert args.pause_before_invocations is True


def test_runner_pauses_after_preparation_before_matrix_cells(tmp_path: pathlib.Path) -> None:
    events: list[str] = []

    class PausingRunner(MatrixRunner):
        def __init__(self) -> None:
            super().__init__(output=tmp_path, keep_resources=False)

        def bootstrap(self) -> None:
            events.append("bootstrap")

        def prepare_deployment(self, deployment: DeploymentCase) -> str:
            events.append(f"prepare:{deployment.setup}:{deployment.framework}")
            return f"https://{deployment.setup}-{deployment.framework}.example.test"

        def after_prepare(self, app_urls: dict[DeploymentCase, str]) -> None:
            assert len(app_urls) == 2
            events.append("pause")

        def execute_case(self, case: MatrixCase, app_url: str) -> dict[str, object]:
            events.append(f"execute:{case.id}")
            return {"case": case.id, "status": "pass"}

        def execute_controls(self, app_urls: dict[DeploymentCase, str]) -> list[dict[str, object]]:
            return [
                {"name": name, "status": "pass"}
                for name in (
                    "exact-primary-matrix",
                    "unknown-alias",
                    "forbidden-header",
                    "missing-user-identity",
                )
            ]

        def cleanup(self) -> list[dict[str, object]]:
            return [{"kind": "app", "name": "offline", "status": "deleted"}]

    assert PausingRunner().run() == 0
    pause_index = events.index("pause")
    assert all(event.startswith("prepare:") for event in events[1:pause_index])
    assert events[pause_index + 1].startswith("execute:")


def test_cleanup_plan_includes_every_created_resource_once() -> None:
    assert cleanup_plan(
        apps=["agent-bricks-one", "agent-bricks-one", "agent-bricks-two"],
        connections=["main.e2e.one", "main.e2e.two", "main.e2e.one"],
    ) == (
        ("app", "agent-bricks-one"),
        ("app", "agent-bricks-two"),
        ("connection", "main.e2e.one"),
        ("connection", "main.e2e.two"),
    )


def test_verify_evidence_requires_8_passes_controls_and_cleanup(tmp_path: pathlib.Path) -> None:
    rows = [{"case": case.id, "status": "pass"} for case in matrix_cases()]
    controls = [
        {"name": name, "status": "pass"}
        for name in (
            "exact-primary-matrix",
            "unknown-alias",
            "forbidden-header",
            "missing-user-identity",
        )
    ]
    target = write_evidence(
        tmp_path,
        rows=rows,
        controls=controls,
        cleanup=[{"kind": "app", "name": "one", "status": "deleted"}],
        scanned_text="",
    )
    assert verify_evidence(target) == 0

    rows[-1]["case"] = rows[-2]["case"]
    target = write_evidence(
        tmp_path,
        rows=rows,
        controls=controls,
        cleanup=[{"kind": "app", "name": "one", "status": "deleted"}],
        scanned_text="",
    )
    assert verify_evidence(target) == 1


def test_verify_evidence_rejects_missing_negative_controls(tmp_path: pathlib.Path) -> None:
    target = write_evidence(
        tmp_path,
        rows=[{"case": case.id, "status": "pass"} for case in matrix_cases()],
        controls=[
            {"name": name, "status": "pass"}
            for name in (
                "exact-primary-matrix",
                "unknown-alias",
                "forbidden-header",
            )
        ],
        cleanup=[{"kind": "app", "name": "one", "status": "deleted"}],
        scanned_text="",
    )

    assert verify_evidence(target) == 1


def test_deployment_is_registered_for_cleanup_before_deploy(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = Namespace(
        output=tmp_path,
        keep_resources=False,
        profile="workspace",
        existing_mcp_connection="main.agentbricks_e2e.existing_mcp",
        existing_http_connection="main.agentbricks_e2e.existing_http",
        existing_mcp_alias="existing-mcp",
        existing_http_alias="existing-http",
        http_path="/agentbricks-e2e",
        mcp_tool="probe",
        probe_mode="fixture",
        mcp_arguments={},
    )
    from e2e.connection_matrix import LiveMatrixRunner

    runner = LiveMatrixRunner(args)
    monkeypatch.setattr(runner, "_install_wheel_source", lambda project: None)

    def run(argv, **kwargs):
        label = kwargs.get("label", "")
        if label == "deploy-existing-langgraph":
            assert runner.created_apps == [
                f"agent-bricks-{runner._deployment_name(DeploymentCase('existing', 'langgraph'))}"
            ]
            raise MatrixError("simulated deploy failure")
        return __import__("subprocess").CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(runner, "_run", run)

    with pytest.raises(MatrixError, match="deploy failure"):
        runner.prepare_deployment(DeploymentCase("existing", "langgraph"))


def test_subprocess_secret_is_rejected_before_log_write(tmp_path: pathlib.Path) -> None:
    script = tmp_path / "emit_secret.py"
    script.write_text("print('client_secret=TOPSECRET')\n", encoding="utf-8")
    args = Namespace(output=tmp_path / "out", keep_resources=False)
    from e2e.connection_matrix import LiveMatrixRunner

    runner = LiveMatrixRunner(args)
    with pytest.raises(MatrixError, match="sensitive"):
        runner._run([sys.executable, str(script)], label="secret-probe")

    assert not (args.output / "logs" / "secret-probe.log").exists()


def test_cleanup_treats_missing_candidate_app_as_success(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = Namespace(output=tmp_path, keep_resources=False, profile="workspace")
    from e2e.connection_matrix import LiveMatrixRunner

    runner = LiveMatrixRunner(args)
    runner.created_apps = ["agent-bricks-candidate"]
    monkeypatch.setattr(
        runner,
        "_run",
        lambda argv, **kwargs: __import__("subprocess").CompletedProcess(
            argv, 1, "Error: deployment does not exist", ""
        ),
    )

    assert runner.cleanup() == [
        {"kind": "app", "name": "agent-bricks-candidate", "status": "not_found"}
    ]


def test_runner_executes_all_cells_controls_and_cleanup(tmp_path: pathlib.Path) -> None:
    class OfflineRunner(MatrixRunner):
        def __init__(self) -> None:
            super().__init__(output=tmp_path, keep_resources=False)
            self.prepared: list[DeploymentCase] = []
            self.executed: list[MatrixCase] = []
            self.cleaned = False

        def bootstrap(self) -> None:
            pass

        def prepare_deployment(self, deployment: DeploymentCase) -> str:
            self.prepared.append(deployment)
            return f"https://{deployment.setup}-{deployment.framework}.example.test"

        def execute_case(self, case: MatrixCase, app_url: str) -> dict[str, object]:
            self.executed.append(case)
            assert f"{case.setup}-{case.framework}" in app_url
            return {"case": case.id, "status": "pass"}

        def execute_controls(self, app_urls: dict[DeploymentCase, str]) -> list[dict[str, object]]:
            assert len(app_urls) == 2
            return [
                {"name": name, "status": "pass"}
                for name in (
                    "exact-primary-matrix",
                    "unknown-alias",
                    "forbidden-header",
                    "missing-user-identity",
                )
            ]

        def cleanup(self) -> list[dict[str, object]]:
            self.cleaned = True
            return [{"kind": "app", "name": "offline", "status": "deleted"}]

        def evidence_metadata(self) -> dict[str, object]:
            return {"mode": "offline"}

    runner = OfflineRunner()
    assert runner.run() == 0
    assert runner.prepared == list(deployment_cases())
    assert runner.executed == list(matrix_cases())
    assert runner.cleaned is True
    assert json.loads((tmp_path / "evidence.json").read_text())["metadata"] == {"mode": "offline"}
