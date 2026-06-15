from tradingagents.agents.analysts.market_analyst import (
    build_compact_market_system_message,
)


def test_compact_market_prompt_preserves_grounding_contract():
    compact = build_compact_market_system_message()

    assert len(compact) < 1200
    assert "get_stock_data" in compact
    assert "get_indicators" in compact
    assert "get_verified_market_snapshot" in compact
    assert "close_50_sma" in compact
    assert "Markdown table" in compact
