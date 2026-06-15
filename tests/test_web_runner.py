import pytest
import requests

from tradingagents.web import runner
from tradingagents.web.runner import (
    build_structured_result,
    execute_run,
    resolve_local_model_settings,
)
from tradingagents.web.store import RunStore

FINAL_STATE = {
    "market_report": "Market report",
    "sentiment_report": "Sentiment report",
    "news_report": "News report",
    "fundamentals_report": "Fundamentals report",
    "investment_debate_state": {
        "bull_history": "Bull case",
        "bear_history": "Bear case",
        "judge_decision": "Research manager decision",
    },
    "investment_plan": "**Recommendation**: Buy",
    "trader_investment_plan": "\n".join(
        [
            "**Action**: Buy",
            "",
            "**Reasoning**: Verified market snapshot supports entry.",
            "",
            "**Entry Price**: 190.5",
            "",
            "**Stop Loss**: 180.0",
            "",
            "**Position Sizing**: 0.5% of portfolio",
        ]
    ),
    "risk_debate_state": {
        "aggressive_history": "Aggressive case",
        "conservative_history": "Conservative case",
        "neutral_history": "Neutral case",
        "judge_decision": "Portfolio manager decision",
    },
    "final_trade_decision": "\n".join(
        [
            "**Rating**: Buy",
            "",
            "**Executive Summary**: Enter with a small position.",
            "",
            "**Investment Thesis**: Earnings and verified price data support the thesis.",
            "",
            "**Price Target**: 215.0",
            "",
            "**Time Horizon**: 3-6 months",
        ]
    ),
}


class SuccessfulEngine:
    def run(self, request, on_progress):
        on_progress("analysts", {"market_report": "Market report"})
        on_progress("research", {"investment_plan": "Buy"})
        on_progress("trader", {"trader_investment_plan": "Buy"})
        on_progress("risk", {"risk_debate_state": {"history": "Risk"}})
        on_progress("portfolio_manager", {"final_trade_decision": "Buy"})
        return FINAL_STATE, "Buy", {"core_stock_apis": "yfinance"}


class FailingEngine:
    def run(self, request, on_progress):
        _ = (request, on_progress)
        raise RuntimeError("5090 endpoint unavailable")


@pytest.mark.unit
def test_build_structured_result_extracts_typed_decisions():
    result = build_structured_result(
        final_state=FINAL_STATE,
        decision="Buy",
        request={"model_route": "local_5090"},
        data_sources={"core_stock_apis": "yfinance"},
    )

    assert result["trader_proposal"] == {
        "action": "Buy",
        "reasoning": "Verified market snapshot supports entry.",
        "entry_price": 190.5,
        "stop_loss": 180.0,
        "position_sizing": "0.5% of portfolio",
    }
    assert result["portfolio_decision"]["rating"] == "Buy"
    assert result["portfolio_decision"]["price_target"] == 215.0
    assert result["portfolio_decision"]["time_horizon"] == "3-6 months"
    assert result["model_route"] == "local_5090"
    assert result["broker_execution"] is False


@pytest.mark.unit
def test_execute_run_updates_progress_and_persists_success(tmp_path):
    store = RunStore(tmp_path)
    run = store.create(
        {
            "ticker": "AAPL",
            "analysis_date": "2026-06-12",
            "model_route": "local_5090",
        }
    )

    execute_run(store, run["run_id"], run["request"], engine=SuccessfulEngine())

    completed = store.get(run["run_id"])
    assert completed["status"] == "succeeded"
    assert completed["progress"]["current_stage"] == "completed"
    assert set(completed["progress"]["stages"].values()) == {"completed"}
    assert completed["result"]["trader_proposal"]["action"] == "Buy"
    assert completed["result"]["portfolio_decision"]["rating"] == "Buy"


@pytest.mark.unit
def test_execute_run_records_failure_without_request_secrets(tmp_path):
    store = RunStore(tmp_path)
    run = store.create(
        {
            "ticker": "AAPL",
            "analysis_date": "2026-06-12",
            "model_route": "local_5090",
            "API_TOKEN": "hidden",
        }
    )

    execute_run(store, run["run_id"], run["request"], engine=FailingEngine())

    failed = store.get(run["run_id"])
    assert failed["status"] == "failed"
    assert failed["error"] == "5090 endpoint unavailable"
    assert "hidden" not in str(failed)


@pytest.mark.unit
def test_local_5090_settings_can_be_loaded_from_workbench_env(tmp_path, monkeypatch):
    env_file = tmp_path / "workbench.env"
    env_file.write_text(
        "\n".join(
            [
                "AI_WORKBENCH_LOCAL_LLM_BASE_URL=http://100.64.0.10:8001/v1",
                "AI_WORKBENCH_LOCAL_LLM_MODEL=qwen-local",
                "AI_WORKBENCH_LOCAL_LLM_API_KEY=not-persisted",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("TRADINGAGENTS_WORKBENCH_ENV_FILE", str(env_file))

    resolved = resolve_local_model_settings({})

    assert resolved == {
        "backend_url": "http://100.64.0.10:8001/v1",
        "quick_think_llm": "qwen-local",
        "deep_think_llm": "qwen-local",
    }


@pytest.mark.unit
def test_local_model_readiness_reports_timeout_without_leaking_credentials(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_LOCAL_LLM_BASE_URL", "http://100.64.0.10:8001/v1")
    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "must-not-leak")

    def timeout_get(*_args, **_kwargs):
        raise requests.Timeout("Bearer must-not-leak timed out")

    readiness = runner.local_model_readiness({}, request_get=timeout_get, timeout_seconds=0.1)

    assert readiness["status"] == "unavailable"
    assert readiness["reachable"] is False
    assert readiness["model_route"] == "local_5090"
    assert "did not respond within" in readiness["message"]
    assert "must-not-leak" not in str(readiness)


@pytest.mark.unit
def test_analysis_engine_stops_before_graph_when_local_model_is_unavailable(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(
        runner,
        "local_model_readiness",
        lambda _request: {
            "status": "unavailable",
            "reachable": False,
            "message": "Configured 5090 endpoint is offline.",
        },
    )
    engine = runner.TradingAgentsAnalysisEngine(tmp_path)

    with pytest.raises(RuntimeError, match="Configured 5090 endpoint is offline"):
        engine.run(
            {
                "ticker": "AAPL",
                "analysis_date": "2026-06-15",
                "model_route": "local_5090",
            },
            lambda *_args: None,
        )
