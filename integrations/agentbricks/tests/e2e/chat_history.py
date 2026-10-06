"""Exercise a deployed OBO chat, then reopen it without resubmitting the prompt."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


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
) -> dict[str, Any]:
    """Run the same user journey against an unfixed or fixed deployed project."""
    from playwright.sync_api import sync_playwright

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
    evidence: dict[str, Any] = {"url": url, "expected": expected, "prompt": prompt, "model": model}
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
                "!document.querySelector('#model-select').disabled", timeout=60000
            )
            page.locator("#model-select").select_option(model)
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
            result = page.request.get(
                url + "/api/invocations/" + request["id"], headers=dict(headers), timeout=60000
            )
            assert result.status == 200, result.text()
            body = result.json()
            assert body["status"] == "completed", body
            messages = body["output"]["output"]
            tools = [item for item in messages if item.get("type") == "tool"]
            assert any(
                item.get("name") == tool and item.get("status") != "error" for item in tools
            ), tools
            assistant = "\n".join(
                str(item.get("content", "")) for item in messages if item.get("type") == "ai"
            )
            assert answer_marker in assistant, assistant
            page.wait_for_function(
                "!document.querySelector('#prompt-input').disabled", timeout=60000
            )
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
            else:
                assert history_response.status == 200, history
                assert history["session_id"] == request["session_id"]
                restored_messages = [item["data"] for item in history["session_items"]]
                assert restored_messages[0]["content"] == prompt, restored_messages
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
    parser.add_argument("--expect", choices=("auth-error", "restored"), default="restored")
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
    )
    logging.basicConfig(level=logging.INFO)
    logging.info("PASS: %s; evidence: %s", args.expect, args.output / "evidence.json")


if __name__ == "__main__":
    main()
