import shlex
import sys
from pathlib import Path

from agentprop.evaluation import run_verification_command, verification_row_fields


def test_run_verification_command_captures_success(tmp_path: Path) -> None:
    result = run_verification_command(
        f"{shlex.quote(sys.executable)} -c 'print(\"ok\")'",
        cwd=tmp_path,
        timeout_s=5,
    )

    assert result.passed
    assert result.status == "passed"
    assert result.returncode == 0
    assert result.cwd == str(tmp_path.resolve())
    assert result.stdout.strip() == "ok"
    assert verification_row_fields(result)["verification_passed"] is True


def test_run_verification_command_captures_failure() -> None:
    result = run_verification_command(
        f"{shlex.quote(sys.executable)} -c 'import sys; sys.exit(3)'",
        timeout_s=5,
    )

    assert not result.passed
    assert result.status == "failed"
    assert result.returncode == 3


def test_run_verification_command_truncates_output() -> None:
    result = run_verification_command(
        f"{shlex.quote(sys.executable)} -c 'print(\"abcdef\")'",
        timeout_s=5,
        output_limit=5,
    )

    assert len(result.stdout) > 5
    assert "truncated" in result.stdout
