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


@pytest.mark.unit
def test_financial_statement_summarizes_full_table_before_limits(monkeypatch):
    payload = "\n".join(
        [
            "# Balance Sheet data for NVDA (quarterly)",
            "# Data retrieved on: 2026-06-16",
            "",
            ",2026-04-30,2026-01-31,2025-10-31,2025-07-31",
            "Cash And Cash Equivalents,40,35,30,25",
            "Total Debt,10,12,14,16",
            "Working Capital,60,58,55,52",
        ]
    )
    set_config(
        {
            "full_data_summary_mode": True,
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
            "ticker": "NVDA",
            "freq": "quarterly",
            "curr_date": "2026-06-15",
        }
    )

    assert "Full financial statement summary generated from all 3 metrics" in result
    assert "across all 4 periods" in result
    assert "Period range: 2026-04-30 to 2025-07-31" in result
    assert "Showing 2 of 3 metrics across the 2 most recent periods" in result
    assert "Cash And Cash Equivalents,40,35" in result
    assert "Working Capital" not in result
