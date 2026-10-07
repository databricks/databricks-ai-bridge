"""Exercise a deployed OBO chat, then reopen it without resubmitting the prompt."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


def _content_text(message: Mapping[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part if isinstance(part, str) else str(part.get("text", ""))
            for part in content
            if isinstance(part, (str, dict))
        )
    return ""


def run_case(
    *,
    url: str,
    headers: Mapping[str, str],
    prompt: str,
    model: str,
    tool: str,
    answer_marker: str,
    expected: str,
    output: Path,
    framework: str = "langgraph",
) -> dict[str, Any]:
    """Run the same user journey against an unfixed or fixed deployed project."""
    sync_playwright = importlib.import_module("playwright.sync_api").sync_playwright

    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not parsed.hostname.endswith(".databricksapps.com")
    ):
        raise ValueError("Expected a deployed HTTPS Databricks App URL")
    if set(headers) != {"Authorization"}:
        raise ValueError("Use OAuth at Apps ingress, never caller-supplied forwarded headers")
    url = url.rstrip("/")
    output.mkdir(parents=True, exist_ok=True)
    evidence: dict[str, Any] = {
        "url": url,
        "expected": expected,
        "prompt": prompt,
        "model": model,
        "framework": framework,
        "verdict": "FAIL",
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome")
        try:
            context = browser.new_context(viewport={"width": 1440, "height": 1000})
            # Restrict authorization to the intended app, not every browser destination.
            context.route(
                url + "/**",
                lambda route: route.continue_(headers={**route.request.headers, **headers}),
            )
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            initial = page.goto(url, wait_until="networkidle", timeout=60000)
            assert initial and initial.status == 200
            page.wait_for_function(
                """model => {
                    const select = document.querySelector('#model-select');
                    return Array.from(select.options).some(option => option.value === model)
                        && !document.querySelector('#prompt-input').disabled;
                }""",
                arg=model,
                timeout=60000,
            )
            if page.locator("#model-select").input_value() != model:
                page.locator("#model-select").select_option(model)
            evidence["model_picker"] = page.locator("#model-select").evaluate(
                """select => ({
                    selected: select.value,
                    disabled: select.disabled,
                    options: Array.from(select.options).map(option => option.value)
                })"""
            )
            page.locator("#prompt-input").fill(prompt)
            with page.expect_response(
                lambda response: response.request.method == "POST"
                and response.url == url + "/api/invocations",
                timeout=600000,
            ) as invocation:
                page.locator("#send-button").click()
            response = invocation.value
            assert response.status == 200, response.text()
            request = response.request.post_data_json
            assert isinstance(request, dict), "Expected a JSON invocation request"
            page.wait_for_function(
                "!document.querySelector('#prompt-input').disabled", timeout=600000
            )
            result = page.request.get(
                url + "/api/invocations/" + request["id"],
                headers={**headers, "X-Routing-Key": request["session_id"]},
                timeout=60000,
            )
            assert result.status == 200, result.text()
            body = result.json()
            evidence.update(request=request, invocation=body)
            assert body["status"] == "completed", body
            messages = body["output"]["output"]
            tools = [item for item in messages if (item.get("type") or item.get("role")) == "tool"]
            if framework == "openai":
                called_tools = [
                    call["name"] for item in messages for call in item.get("tool_calls", [])
                ]
                assert tool in called_tools, called_tools
                # OpenAI's template output omits call IDs and can omit tool-result names. Verify
                # the requested query-result call and grounded result separately; retain raw items.
                results = tools
                evidence["called_tools"] = called_tools
            else:
                results = [item for item in tools if item.get("name") == tool]
                assert any(item.get("status") != "error" for item in results), results
            assert any(answer_marker in json.dumps(item) for item in results), results
            assistant = "\n".join(
                _content_text(item)
                for item in messages
                if (item.get("type") or item.get("role")) in ("ai", "assistant")
            )
            assert answer_marker in assistant, assistant
            history_before = page.request.get(
                url + "/api/demo/session/items",
                params={"session_id": request["session_id"]},
                headers={**headers, "X-Routing-Key": request["session_id"]},
                timeout=60000,
            )
            evidence["history_before_reopen"] = history_before.json()
            page.screenshot(path=str(output / "before-reopen.png"))
            submissions: list[str] = []
            page.on(
                "request",
                lambda request: submissions.append(request.url)
                if request.method == "POST" and request.url == url + "/api/invocations"
                else None,
            )
            with page.expect_response(
                lambda response: "/api/demo/session/items?" in response.url,
                timeout=60000,
            ) as restored:
                page.reload(wait_until="networkidle", timeout=60000)
            history_response = restored.value
            history = history_response.json()
            evidence.update(
                request=request,
                invocation=body,
                assistant=assistant,
                history_http_status=history_response.status,
                history=history,
            )
            if expected == "auth-error":
                assert history_response.status == 401, history
                assert history["error"]["code"] == "MCP_USER_AUTHORIZATION_MISSING", history
            elif expected == "empty-history":
                assert history_response.status == 200, history
                assert history["session_items"] == [], history
            else:
                assert history_response.status == 200, history
                assert history["session_id"] == request["session_id"]
                restored_messages = [item["data"] for item in history["session_items"]]
                assert restored_messages[0]["content"] == prompt, restored_messages
                if framework == "openai":
                    assert history == evidence["history_before_reopen"]
                    restored_assistant = "\n".join(
                        _content_text(item)
                        for item in restored_messages
                        if item.get("role") == "assistant"
                    )
                    assert restored_assistant == assistant, restored_messages
                else:
                    assert restored_messages[1:] == messages, restored_messages
                page.locator("#chat-log").get_by_text(answer_marker, exact=False).first.wait_for(
                    timeout=60000
                )
                assert prompt in page.locator("#chat-log").inner_text()
                assert history["interrupts"] == []
            assert submissions == [], "Reopening must not execute another agent invocation"
            assert errors == [], errors
            page.screenshot(path=str(output / "after-reopen.png"))
            evidence.update(no_invocation_on_reopen=True, page_errors=errors, verdict="PASS")
        finally:
            browser.close()
            (output / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    return evidence


def main() -> None:
    from databricks.sdk import WorkspaceClient

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--app-auth-profile", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--answer-marker", required=True)
    parser.add_argument("--framework", choices=("langgraph", "openai"), default="langgraph")
    parser.add_argument(
        "--expect", choices=("auth-error", "empty-history", "restored"), default="restored"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = WorkspaceClient(profile=args.app_auth_profile).config
    if config.auth_type == "pat":
        raise ValueError("An OAuth profile is required for genuine OBO testing")
    authorization = config.authenticate().get("Authorization")
    if not authorization:
        raise ValueError("No OAuth authorization available")
    run_case(
        url=args.url,
        headers={"Authorization": authorization},
        prompt=args.prompt,
        model=args.model,
        tool=args.tool,
        answer_marker=args.answer_marker,
        expected=args.expect,
        output=args.output,
        framework=args.framework,
    )
    logging.basicConfig(level=logging.INFO)
    logging.info("PASS: %s; evidence: %s", args.expect, args.output / "evidence.json")


if __name__ == "__main__":
    main()
