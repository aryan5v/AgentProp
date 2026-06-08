"""Harbor custom agent that runs Gemini through AgentProp control."""

from __future__ import annotations

import os
import shlex
import tempfile
from pathlib import Path
from typing import Any, cast

try:  # Harbor is optional for normal AgentProp installs/tests.
    from harbor.agents.installed.gemini_cli import GeminiCli
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError:  # pragma: no cover - exercised only without Harbor installed.

    class GeminiCli:  # type: ignore[no-redef]
        async def exec_as_agent(self, environment: Any, command: str, **kwargs: Any) -> None:
            del environment, command, kwargs
            raise RuntimeError("Harbor is required to execute AgentPropGeminiAgent")

    BaseEnvironment = Any  # type: ignore[misc,assignment]
    AgentContext = Any  # type: ignore[misc,assignment]


class AgentPropGeminiAgent(GeminiCli):  # type: ignore[misc]
    """Harbor agent that runs Gemini CLI through AgentProp's terminal controller."""

    @staticmethod
    def name() -> str:
        return "agentprop-gemini"

    def version(self) -> str | None:
        return os.environ.get("AGENTPROP_GEMINI_AGENT_VERSION", "0.1")

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        install_spec = os.environ.get("AGENTPROP_INSTALL_SPEC", "agentprop[mcp]")
        local_source = os.environ.get("AGENTPROP_LOCAL_SOURCE", "").strip()
        if local_source:
            source_root = Path(local_source).expanduser().resolve()
            if not source_root.is_dir():
                raise FileNotFoundError(f"AGENTPROP_LOCAL_SOURCE is not a directory: {source_root}")
            await environment.upload_dir(source_root, "/opt/agentprop-src")
            pip_install = "/opt/agentprop-venv/bin/pip install -e /opt/agentprop-src"
        else:
            pip_install = (
                f"/opt/agentprop-venv/bin/pip install --upgrade {shlex.quote(install_spec)}"
            )
        await self.exec_as_root(
            environment,
            command="apt-get update && apt-get install -y python3 python3-pip python3-venv git",
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail\n"
                "python3 -m venv /opt/agentprop-venv\n"
                "/opt/agentprop-venv/bin/pip install --upgrade pip\n"
                f"{pip_install}\n"
                'export PATH="/opt/agentprop-venv/bin:$HOME/.local/bin:$PATH"\n'
                "python -m agentprop.cli doctor --tier graph\n"
            ),
        )

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        task_id = _context_value(context, "task_name") or _context_value(context, "trial_name")
        task_id = task_id or "terminal-bench-task"
        auth_path = await _upload_gemini_auth_file(environment, self)
        command = (
            "set -euo pipefail\n"
            f"trap 'rm -f {shlex.quote(auth_path)}' EXIT\n"
            f". {shlex.quote(auth_path)}\n"
            "export AGENTPROP_HARBOR_SCORE_ONLY=1\n"
            'export PATH="/opt/agentprop-venv/bin:$HOME/.local/bin:$PATH"\n'
            'export NVM_DIR="$HOME/.nvm"\n'
            '. "$NVM_DIR/nvm.sh"\n'
            "python -m agentprop.benchmarks.gemini_terminal_agent "
            f"--instruction {shlex.quote(instruction)} "
            f"--task-id {shlex.quote(str(task_id))} "
            "--category terminal-bench "
            "--workspace . "
            '--model "${GEMINI_MODEL:-gemini-3.1-pro-preview}" '
            '--max-steps "${AGENTPROP_MAX_STEPS:-64}" '
            '--trace-dir ".agentprop/gemini-terminal-bench"'
        )
        await self.exec_as_agent(environment, command=command)

    def populate_context_post_run(self, context: AgentContext) -> None:
        del context


async def _upload_gemini_auth_file(
    environment: BaseEnvironment,
    agent: AgentPropGeminiAgent,
) -> str:
    remote_path = "/tmp/agentprop-gemini-auth.env"
    env = _gemini_auth_env()
    lines = [f"export {key}={shlex.quote(value)}" for key, value in sorted(env.items())]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write("\n".join(lines))
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    try:
        await environment.upload_file(tmp_path, remote_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    if environment.default_user is not None:
        await agent.exec_as_root(
            environment,
            command=f"chown {environment.default_user} {shlex.quote(remote_path)}",
        )
    await agent.exec_as_agent(
        environment,
        command=f"chmod 600 {shlex.quote(remote_path)}",
    )
    return remote_path


def _gemini_auth_env() -> dict[str, str]:
    env = {"GEMINI_CLI_TRUST_WORKSPACE": "true"}
    for var in (
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "GOOGLE_GENAI_USE_VERTEXAI",
    ):
        value = os.environ.get(var)
        if value:
            env[var] = value
    return env


def _context_value(context: Any, name: str) -> object | None:
    if hasattr(context, name):
        return cast(object, getattr(context, name))
    if isinstance(context, dict):
        return cast(object | None, context.get(name))
    return None
