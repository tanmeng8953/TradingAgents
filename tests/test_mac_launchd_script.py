import os
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "mac_install_web_launchd.sh"


def _run(tmp_path, *args, available_kib):
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "TRADINGAGENTS_TEST_AVAILABLE_KIB": str(available_kib),
        "TRADINGAGENTS_SKIP_LAUNCHCTL": "1",
    }
    return subprocess.run(
        ["/bin/zsh", str(SCRIPT), *args],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )


@pytest.mark.unit
def test_launchd_install_stops_below_five_gib(tmp_path):
    result = _run(tmp_path, "install", available_kib=4 * 1024 * 1024)

    assert result.returncode != 0
    assert "5 GiB" in result.stderr
    assert not list(tmp_path.rglob("*.plist"))


@pytest.mark.unit
def test_launchd_install_is_local_only_and_contains_no_credentials(tmp_path):
    result = _run(tmp_path, "install", available_kib=6 * 1024 * 1024)

    assert result.returncode == 0, result.stderr
    plist = (
        tmp_path / "Library" / "LaunchAgents" / "com.tanmeng.tradingagents-web.plist"
    ).read_text(encoding="utf-8")
    assert "<string>127.0.0.1</string>" in plist
    assert "<string>8502</string>" in plist
    assert ".venv/bin/tradingagents" in plist
    assert "TRADINGAGENTS_WORKBENCH_ENV_FILE" in plist
    assert "/Users/tanmeng/GitHub/AI/ai-workbench/.env" in plist
    assert "API_KEY" not in plist
    assert "TOKEN" not in plist
    assert "PASSWORD" not in plist
