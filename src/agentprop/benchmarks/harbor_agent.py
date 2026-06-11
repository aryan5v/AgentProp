"""Harbor custom agents backed by AgentProp runtime control."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import tempfile
from pathlib import Path
from typing import Any, cast

try:  # Harbor is optional for normal AgentProp installs/tests.
    from harbor.agents.installed.base import BaseInstalledAgent, with_prompt_template
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext
except ModuleNotFoundError:  # pragma: no cover - exercised only without Harbor installed.

    class BaseInstalledAgent:  # type: ignore[no-redef]
        async def exec_as_agent(self, environment: Any, command: str, **kwargs: Any) -> None:
            del environment, command, kwargs
            raise RuntimeError("Harbor is required to execute AgentPropCursorAgent")

    def with_prompt_template(fn: Any) -> Any:
        return fn

    BaseEnvironment = Any
    AgentContext = Any


class AgentPropCursorAgent(BaseInstalledAgent):  # type: ignore[misc]
    """Harbor agent that runs Cursor through AgentProp's per-command controller."""

    @staticmethod
    def name() -> str:
        return "agentprop-cursor"

    def version(self) -> str | None:
        return os.environ.get("AGENTPROP_CURSOR_AGENT_VERSION", "0.1")

    def get_version_command(self) -> str | None:
        return 'export PATH="$HOME/.local/bin:$PATH"; cursor-agent --version'

    def _agent_env(self, name: str, default: str = "") -> str:
        value = self._extra_env.get(name) or os.environ.get(name, default)
        return value.strip() if isinstance(value, str) else default

    async def install(self, environment: BaseEnvironment) -> None:
        install_spec = self._agent_env("AGENTPROP_INSTALL_SPEC", "agentprop")
        local_source = self._agent_env("AGENTPROP_LOCAL_SOURCE")
        wheel_path = self._agent_env("AGENTPROP_WHEEL_PATH")
        if wheel_path:
            wheel = Path(wheel_path).expanduser().resolve()
            if not wheel.is_file():
                raise FileNotFoundError(f"AGENTPROP_WHEEL_PATH is not a file: {wheel}")
            remote_wheel = f"/tmp/{wheel.name}"
            await environment.upload_file(wheel, remote_wheel)
            pip_install = f"/opt/agentprop-venv/bin/pip install {shlex.quote(remote_wheel)}"
        elif local_source:
            source_root = Path(local_source).expanduser().resolve()
            if not source_root.is_dir():
                raise FileNotFoundError(f"AGENTPROP_LOCAL_SOURCE is not a directory: {source_root}")
            staging = Path(tempfile.mkdtemp(prefix="agentprop-staging-"))
            try:
                for name in ("src", "pyproject.toml", "README.md", "LICENSE"):
                    src = source_root / name
                    if not src.exists():
                        continue
                    dest = staging / name
                    if src.is_dir():
                        shutil.copytree(src, dest)
                    else:
                        shutil.copy2(src, dest)
                await environment.upload_dir(staging, "/opt/agentprop-src")
            finally:
                shutil.rmtree(staging, ignore_errors=True)
            pip_install = "/opt/agentprop-venv/bin/pip install -e /opt/agentprop-src"
        else:
            pip_install = (
                f"/opt/agentprop-venv/bin/pip install --upgrade {shlex.quote(install_spec)}"
            )
        await self.exec_as_root(
            environment,
            command=(
                "apt-get update && "
                "apt-get install -y curl git python3 python3-pip python3-venv"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "set -euo pipefail\n"
                "python3 -m venv /opt/agentprop-venv\n"
                "/opt/agentprop-venv/bin/pip install --upgrade pip\n"
                f"{pip_install}\n"
                "if ! command -v cursor-agent >/dev/null 2>&1; then\n"
                "  curl https://cursor.com/install -fsS | bash\n"
                "fi\n"
                'export PATH="/opt/agentprop-venv/bin:$HOME/.local/bin:$PATH"\n'
                "cursor-agent --version\n"
                "/opt/agentprop-venv/bin/python -m agentprop.cli doctor --tier graph\n"
                "/opt/agentprop-venv/bin/python -c "
                "\"import agentprop.benchmarks.cursor_terminal_agent\"\n"
            ),
        )

    @with_prompt_template  # type: ignore[untyped-decorator,misc,unused-ignore]
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        task_id = _context_value(context, "task_name") or _context_value(context, "trial_name")
        task_id = task_id or "terminal-bench-task"
        extra_args: list[str] = []
        if _env_bool(self._agent_env("AGENTPROP_USE_SYSTEM_PYTHON", "1")):
            extra_args.append("--use-system-python")
        fast_path = self._agent_env("AGENTPROP_FAST_PATH", "yolo-until-verifier-miss")
        if fast_path and fast_path != "off":
            extra_args.extend(["--fast-path", shlex.quote(fast_path)])
        fast_path_timeout = self._agent_env("AGENTPROP_FAST_PATH_TIMEOUT_S", "900")
        if fast_path_timeout:
            extra_args.extend(["--fast-path-timeout-s", shlex.quote(fast_path_timeout)])
        command = (
            "set -euo pipefail\n"
            'export PATH="/opt/agentprop-venv/bin:$HOME/.local/bin:$PATH"\n'
            'export AGENTPROP_HARBOR_LOGS_DIR="/logs/agent"\n'
            'export AGENTPROP_HARBOR_SCORE_ONLY="${AGENTPROP_HARBOR_SCORE_ONLY:-1}"\n'
            'export AGENTPROP_USE_SYSTEM_PYTHON="${AGENTPROP_USE_SYSTEM_PYTHON:-1}"\n'
            'export AGENTPROP_FAST_PATH="${AGENTPROP_FAST_PATH:-yolo-until-verifier-miss}"\n'
            'export AGENTPROP_FAST_PATH_TIMEOUT_S="${AGENTPROP_FAST_PATH_TIMEOUT_S:-900}"\n'
            'test -n "${CURSOR_API_KEY:-}" || '
            '(echo "CURSOR_API_KEY is required for agentprop-cursor" >&2; exit 2)\n'
            "/opt/agentprop-venv/bin/python -m agentprop.benchmarks.cursor_terminal_agent "
            f"--instruction {shlex.quote(instruction)} "
            f"--task-id {shlex.quote(str(task_id))} "
            "--category terminal-bench "
            "--workspace . "
            '--model "${CURSOR_MODEL:-composer-2.5}" '
            '--max-steps "${AGENTPROP_MAX_STEPS:-64}" '
            '--trace-dir ".agentprop/cursor-terminal-bench" '
            + " ".join(extra_args)
        )
        await self.exec_as_agent(environment, command=command)

    def populate_context_post_run(self, context: AgentContext) -> None:
        usage_path = self.logs_dir / "agentprop-cursor-usage.json"
        if not usage_path.exists():
            return
        try:
            payload = json.loads(usage_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        if not isinstance(payload, dict):
            return
        context.n_input_tokens = int(payload.get("n_input_tokens") or 0)
        context.n_cache_tokens = int(payload.get("n_cache_tokens") or 0)
        context.n_output_tokens = int(payload.get("n_output_tokens") or 0)
        cost_usd = payload.get("cost_usd")
        if isinstance(cost_usd, int | float):
            context.cost_usd = float(cost_usd)


def _context_value(context: Any, name: str) -> object | None:
    if hasattr(context, name):
        return cast(object, getattr(context, name))
    if isinstance(context, dict):
        return cast(object | None, context.get(name))
    return None


def _env_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
