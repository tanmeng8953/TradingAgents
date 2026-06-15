import copy

import pytest

import tradingagents.default_config as default_config
from tradingagents.agents.utils import fundamental_data_tools
from tradingagents.dataflows import config as config_module
from tradingagents.dataflows.config import set_config


@pytest.fixture(autouse=True)
def reset_dataflow_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.mark.unit
def test_financial_statement_limits_rows_and_periods_for_small_context_models(
    monkeypatch,
):
    payload = "\n".join(
        [
            "# Balance Sheet data for AAPL (annual)",
            "# Data retrieved on: 2026-06-16",
            "",
            ",2025-09-30,2024-09-30,2023-09-30,2022-09-30",
            "Net Debt,10,20,30,40",
            "Total Debt,11,21,31,41",
            "Working Capital,12,22,32,42",
        ]
    )
    set_config(
        {
            "fundamental_statement_output_rows": 2,
            "fundamental_statement_output_periods": 2,
        }
    )
    monkeypatch.setattr(
        fundamental_data_tools,
        "route_to_vendor",
        lambda *_args, **_kwargs: payload,
    )

    result = fundamental_data_tools.get_balance_sheet.invoke(
        {
            "ticker": "AAPL",
            "freq": "annual",
            "curr_date": "2026-06-12",
        }
    )

    assert "Showing 2 of 3 metrics across the 2 most recent periods" in result
    assert ",2025-09-30,2024-09-30" in result
    assert "2023-09-30" not in result
    assert "Net Debt,10,20" in result
    assert "Total Debt,11,21" in result
    assert "Working Capital" not in result
