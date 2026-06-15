import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage

from tradingagents.graph.trading_graph import TradingAgentsGraph


def test_log_state_serializes_langchain_messages(tmp_path):
    graph = object.__new__(TradingAgentsGraph)
    graph.ticker = "AAPL"
    graph.config = {"results_dir": str(tmp_path)}
    graph.log_states_dict = {}
    final_state = {
        "company_of_interest": "AAPL",
        "trade_date": "2026-06-16",
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
        "fundamentals_report": "Fundamentals report.",
        "investment_debate_state": {
            "bull_history": AIMessage(content="Bull case."),
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
        },
        "trader_investment_plan": "Hold.",
        "risk_debate_state": {
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "history": "",
            "judge_decision": "",
        },
        "investment_plan": "Hold.",
        "final_trade_decision": "Rating: Hold",
    }

    graph._log_state("2026-06-16", final_state)

    log_path = (
        tmp_path
        / "AAPL"
        / "TradingAgentsStrategy_logs"
        / "full_states_log_2026-06-16.json"
    )
    persisted = json.loads(log_path.read_text(encoding="utf-8"))
    assert persisted["investment_debate_state"]["bull_history"]["content"] == (
        "Bull case."
    )


def test_run_graph_recovers_completed_terminal_checkpoint(tmp_path, monkeypatch):
    final_state = {
        "company_of_interest": "AAPL",
        "trade_date": "2026-06-16",
        "market_report": "Market report.",
        "sentiment_report": "Sentiment report.",
        "news_report": "News report.",
        "fundamentals_report": "Fundamentals report.",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
        },
        "trader_investment_plan": "Hold.",
        "risk_debate_state": {
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "history": "",
            "judge_decision": "",
        },
        "investment_plan": "Hold.",
        "final_trade_decision": "Rating: Hold",
    }
    stale_state = {
        **final_state,
        "final_trade_decision": "Stale inherited decision. " * 100,
        "market_report": "",
        "sentiment_report": "",
        "news_report": "",
        "fundamentals_report": "",
        "investment_debate_state": {
            "bull_history": "",
            "bear_history": "",
            "history": "",
            "current_response": "",
            "judge_decision": "",
        },
        "investment_plan": "",
        "trader_investment_plan": "",
        "risk_debate_state": {
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "history": "",
            "judge_decision": "",
        },
    }
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {
        "checkpoint_enabled": True,
        "data_cache_dir": str(tmp_path / "cache"),
        "results_dir": str(tmp_path / "results"),
    }
    graph.graph = MagicMock()
    graph.graph.get_state.return_value = SimpleNamespace(
        values=stale_state,
        metadata={"writes": {"Market Analyst": {}}},
    )
    graph.graph.get_state_history.return_value = [
        SimpleNamespace(
            values=final_state,
            metadata={"writes": {"Portfolio Manager": {}}},
        ),
    ]
    graph.propagator = MagicMock()
    graph.propagator.create_initial_state.return_value = {}
    graph.propagator.get_graph_args.return_value = {}
    graph.memory_log = MagicMock()
    graph.signal_processor = MagicMock()
    graph.signal_processor.process_signal.return_value = "Hold"
    graph.log_states_dict = {}
    graph.ticker = "AAPL"
    graph.curr_state = None
    graph.debug = False
    graph.resolve_instrument_context = MagicMock(return_value="AAPL context")
    monkeypatch.setattr(
        "tradingagents.graph.trading_graph.clear_checkpoint",
        MagicMock(),
    )

    state, decision = graph._run_graph("AAPL", "2026-06-16")

    assert state == final_state
    assert decision == "Hold"
    graph.graph.invoke.assert_not_called()
    graph.memory_log.store_decision.assert_called_once()
