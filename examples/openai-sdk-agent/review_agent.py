"""OpenAI Agents SDK loop for the PR-review example."""

import asyncio
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

from agents import Agent, Runner, function_tool

MODEL = "gpt-5.6-luna"
SHELL_TIMEOUT_SECONDS = 15 * 60
SHELL_OUTPUT_LIMIT = 60_000
SENSITIVE_ENVIRONMENT_MARKERS = (
    "CREDENTIAL",
    "DATABRICKS",
    "KEY",
    "LAKEBASE",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)
CUJS = (
    ("quality", "discover and run the repository's formatting, lint, and type checks"),
    ("package-tests", "run all PR-relevant package test suites and report each separately"),
    ("build-install", "build distributions, install them cleanly, and verify imports"),
    ("agent-e2e", "launch an affected example agent, invoke it, and retain useful logs"),
)


def _model() -> str:
    configured = os.getenv("OPENAI_MODEL", MODEL)
    if configured != MODEL:
        raise RuntimeError(f"This cost-controlled example only allows {MODEL}; got {configured}")
    return configured


def _shell_environment(workspace: Path) -> dict[str, str]:
    environment = {
        name: value
        for name, value in os.environ.items()
        if not any(marker in name.upper() for marker in SENSITIVE_ENVIRONMENT_MARKERS)
    }
    environment["HOME"] = str(workspace)
    return environment


def _limit_output(output: str) -> str:
    if len(output) <= SHELL_OUTPUT_LIMIT:
        return output
    half_limit = SHELL_OUTPUT_LIMIT // 2
    return f"{output[:half_limit]}\n... output truncated ...\n{output[-half_limit:]}"


def create_agent(workspace: Path) -> Agent:
    """Create an SDK agent with a shell backed by the Databricks App process."""

    @function_tool
    async def run_shell(command: str) -> str:
        """Run one shell command in the review workspace and return its status and output."""
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=workspace,
            env=_shell_environment(workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=SHELL_TIMEOUT_SECONDS)
        except TimeoutError:
            process.kill()
            stdout, _ = await process.communicate()
            output = stdout.decode(errors="replace")
            return _limit_output(
                f"Command timed out after {SHELL_TIMEOUT_SECONDS} seconds.\n{output}"
            )
        except asyncio.CancelledError:
            process.kill()
            await process.communicate()
            raise
        output = stdout.decode(errors="replace")
        return _limit_output(f"Exit status: {process.returncode}\n{output}")

    return Agent(
        name="Sequential PR CUJ reviewer",
        model=_model(),
        instructions=(
            "Review the supplied public GitHub PR by actually executing every CUJ in order. "
            "Use run_shell for checkout, package installation, tests, and app execution. "
            "Do not claim a check passed without command output. Keep changes inside the review "
            "environment and finish with a concise evidence-based Markdown report."
        ),
        tools=[run_shell],
    )


def build_prompt(pr_url: str, iteration: int, workspace: Path, recovery_note: str = "") -> str:
    steps = "\n".join(f"{index}. {name}: {goal}" for index, (name, goal) in enumerate(CUJS, 1))
    return f"""PR: {pr_url}
Iteration: {iteration}
Workspace: {workspace}

First clone the repository into {workspace}/repo and check out the PR head. Each run_shell call
starts in {workspace}, so use an explicit `cd {workspace}/repo` for repository commands. Then
execute these CUJs sequentially in the same environment:
{steps}

If a CUJ fails, investigate the cause and continue with the remaining CUJs. Include exact commands,
exit status, and important output in the final report. {recovery_note}
"""


async def execute_review(
    pr_url: str, minimum_minutes: float, session, recovery_note: str = ""
) -> str:
    """Run complete CUJ iterations sequentially until the minimum wall time is met."""
    started = time.monotonic()
    iteration = 0
    reports: list[str] = []
    with TemporaryDirectory(prefix="openai-sdk-agent-") as temporary_directory:
        workspace = Path(temporary_directory)
        while iteration == 0 or time.monotonic() - started < minimum_minutes * 60:
            iteration += 1
            result = await Runner.run(
                create_agent(workspace),
                build_prompt(pr_url, iteration, workspace, recovery_note if iteration == 1 else ""),
                session=session,
                max_turns=100,
            )
            reports.append(f"## Iteration {iteration}\n\n{result.final_output}")
            if minimum_minutes <= 0:
                break
    return "\n\n".join(reports)


async def resume_review(session, recovery_note: str) -> str:
    """Resume using only the SDK session history and a fixed recovery note."""
    with TemporaryDirectory(prefix="openai-sdk-agent-") as temporary_directory:
        workspace = Path(temporary_directory)
        result = await Runner.run(
            create_agent(workspace),
            recovery_note,
            session=session,
            max_turns=100,
        )
        return str(result.final_output)
