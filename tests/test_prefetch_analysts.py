import copy
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

import tradingagents.default_config as default_config
from tradingagents.agents.analysts import fundamentals_analyst, market_analyst, news_analyst
from tradingagents.dataflows import config as config_module
from tradingagents.dataflows.config import set_config


@pytest.fixture(autouse=True)
def reset_dataflow_config():
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)
    yield
    config_module._config = copy.deepcopy(default_config.DEFAULT_CONFIG)


class NoToolLLM:
    def __init__(self, response: str = "prefetched report"):
        self.response = response
        self.messages = []

    def bind_tools(self, _tools):
        raise AssertionError("local prefetch mode should not bind tools")

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=self.response)


def _state():
    return {
        "company_of_interest": "NVDA",
        "trade_date": "2026-06-15",
        "asset_type": "stock",
        "messages": [HumanMessage(content="Analyze NVDA")],
    }


@pytest.mark.unit
def test_market_analyst_prefetch_mode_avoids_tool_calling(monkeypatch):
    set_config({"prefetch_analyst_data": True})
    monkeypatch.setattr(
        market_analyst,
        "get_stock_data",
        SimpleNamespace(func=lambda *_args: "PRICE_BLOCK"),
    )
    monkeypatch.setattr(
        market_analyst,
        "get_indicators",
        SimpleNamespace(func=lambda *_args: "INDICATOR_BLOCK"),
    )
    monkeypatch.setattr(
        market_analyst,
        "get_verified_market_snapshot",
        SimpleNamespace(func=lambda *_args: "SNAPSHOT_BLOCK"),
    )
    llm = NoToolLLM("market report")

    result = market_analyst.create_market_analyst(llm)(_state())

    assert result["market_report"] == "market report"
    assert result["messages"][0].tool_calls == []
    assert "PRICE_BLOCK" in llm.messages[0].content
    assert "INDICATOR_BLOCK" in llm.messages[0].content
    assert "SNAPSHOT_BLOCK" in llm.messages[0].content


@pytest.mark.unit
def test_fundamentals_analyst_prefetch_mode_avoids_tool_calling(monkeypatch):
    set_config({"prefetch_analyst_data": True})
    monkeypatch.setattr(
        fundamentals_analyst,
        "get_fundamentals",
        SimpleNamespace(func=lambda *_args: "PROFILE_BLOCK"),
    )
    monkeypatch.setattr(
        fundamentals_analyst,
        "get_balance_sheet",
        SimpleNamespace(func=lambda *_args: "BALANCE_BLOCK"),
    )
    monkeypatch.setattr(
        fundamentals_analyst,
        "get_cashflow",
        SimpleNamespace(func=lambda *_args: "CASHFLOW_BLOCK"),
    )
    monkeypatch.setattr(
        fundamentals_analyst,
        "get_income_statement",
        SimpleNamespace(func=lambda *_args: "INCOME_BLOCK"),
    )
    llm = NoToolLLM("fundamentals report")

    result = fundamentals_analyst.create_fundamentals_analyst(llm)(_state())

    assert result["fundamentals_report"] == "fundamentals report"
    assert result["messages"][0].tool_calls == []
    assert "PROFILE_BLOCK" in llm.messages[0].content
    assert "BALANCE_BLOCK" in llm.messages[0].content
    assert "CASHFLOW_BLOCK" in llm.messages[0].content
    assert "INCOME_BLOCK" in llm.messages[0].content


@pytest.mark.unit
def test_news_analyst_prefetch_mode_avoids_tool_calling(monkeypatch):
    set_config({"prefetch_analyst_data": True})
    monkeypatch.setattr(
        news_analyst,
        "get_news",
        SimpleNamespace(func=lambda *_args: "TICKER_NEWS_BLOCK"),
    )
    monkeypatch.setattr(
        news_analyst,
        "get_global_news",
        SimpleNamespace(func=lambda *_args: "GLOBAL_NEWS_BLOCK"),
    )
    monkeypatch.setattr(
        news_analyst,
        "get_macro_indicators",
        SimpleNamespace(func=lambda *_args: "MACRO_BLOCK"),
    )
    monkeypatch.setattr(
        news_analyst,
        "get_prediction_markets",
        SimpleNamespace(func=lambda *_args: "PREDICTION_BLOCK"),
    )
    llm = NoToolLLM("news report")

    result = news_analyst.create_news_analyst(llm)(_state())

    assert result["news_report"] == "news report"
    assert result["messages"][0].tool_calls == []
    assert "TICKER_NEWS_BLOCK" in llm.messages[0].content
    assert "GLOBAL_NEWS_BLOCK" in llm.messages[0].content
    assert "MACRO_BLOCK" in llm.messages[0].content
    assert "PREDICTION_BLOCK" in llm.messages[0].content
