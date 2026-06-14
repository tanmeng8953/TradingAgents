from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from cli.main import app


@pytest.mark.unit
def test_web_command_starts_local_only_server():
    runner = CliRunner()

    with patch("uvicorn.run") as run:
        result = runner.invoke(
            app,
            ["web", "--host", "127.0.0.1", "--port", "8502"],
        )

    assert result.exit_code == 0
    run.assert_called_once()
    assert run.call_args.kwargs["host"] == "127.0.0.1"
    assert run.call_args.kwargs["port"] == 8502
