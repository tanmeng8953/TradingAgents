import copy

import pytest

import tradingagents.default_config as default_config
from tradingagents.agents.utils import technical_indicators_tools
from tradingagents.dataflows import config as config_module
from tradingagents.dataflows.config import set_config


@pytest.fixture(autouse=True)
def reset_dataflow_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.mark.unit
def test_get_indicators_limits_values_for_small_context_models(monkeypatch):
    payload = "\n".join(
        [
            "## rsi values from 2026-06-01 to 2026-06-05:",
            "",
            "2026-06-05: 55",
            "2026-06-04: 54",
            "2026-06-03: 53",
            "2026-06-02: 52",
            "2026-06-01: 51",
        ]
    )
    set_config({"indicator_output_rows": 2})
    monkeypatch.setattr(
        technical_indicators_tools,
        "route_to_vendor",
        lambda *_args, **_kwargs: payload,
    )

    result = technical_indicators_tools.get_indicators.invoke(
        {
            "symbol": "AAPL",
            "indicator": "rsi",
            "curr_date": "2026-06-05",
            "look_back_days": 30,
        }
    )

    assert "Showing the 2 most recent values out of 5" in result
    assert "2026-06-05: 55" in result
    assert "2026-06-04: 54" in result
    assert "2026-06-03" not in result


@pytest.mark.unit
def test_get_indicators_summarizes_full_series_before_value_limit(monkeypatch):
    payload = "\n".join(
        [
            "## rsi values from 2026-06-01 to 2026-06-05:",
            "",
            "2026-06-05: 65",
            "2026-06-04: 60",
            "2026-06-03: 55",
            "2026-06-02: 50",
            "2026-06-01: 45",
        ]
    )
    set_config(
        {
            "full_data_summary_mode": True,
            "indicator_output_rows": 2,
        }
    )
    monkeypatch.setattr(
        technical_indicators_tools,
        "route_to_vendor",
        lambda *_args, **_kwargs: payload,
    )

    result = technical_indicators_tools.get_indicators.invoke(
        {
            "symbol": "NVDA",
            "indicator": "rsi",
            "curr_date": "2026-06-05",
            "look_back_days": 30,
        }
    )

    assert "Full indicator summary generated from all 5 dated values" in result
    assert "Date range: 2026-06-01 to 2026-06-05" in result
    assert "Latest value: 65" in result
    assert "Change over full series: 20" in result
    assert "Showing the 2 most recent values out of 5" in result
    assert "2026-06-03" not in result
