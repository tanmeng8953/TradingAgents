import copy

import pytest

import tradingagents.default_config as default_config
from tradingagents.agents.utils import core_stock_tools
from tradingagents.dataflows import config as config_module
from tradingagents.dataflows.config import set_config


@pytest.fixture(autouse=True)
def reset_dataflow_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


@pytest.mark.unit
def test_get_stock_data_limits_rows_for_small_context_models(monkeypatch):
    payload = "\n".join(
        [
            "Date,Close,High,Low,Open,Volume",
            "2026-06-01,1,2,0,1,10",
            "2026-06-02,2,3,1,2,20",
            "2026-06-03,3,4,2,3,30",
            "2026-06-04,4,5,3,4,40",
            "2026-06-05,5,6,4,5,50",
        ]
    )
    set_config({"stock_data_output_rows": 3})
    monkeypatch.setattr(
        core_stock_tools,
        "route_to_vendor",
        lambda *_args, **_kwargs: payload,
    )

    result = core_stock_tools.get_stock_data.invoke(
        {
            "symbol": "AAPL",
            "start_date": "2026-06-01",
            "end_date": "2026-06-05",
        }
    )

    assert "Showing the 3 most recent rows out of 5" in result
    assert "Date,Close,High,Low,Open,Volume" in result
    assert "2026-06-01" not in result
    assert "2026-06-02" not in result
    assert "2026-06-03" in result
    assert "2026-06-05" in result
