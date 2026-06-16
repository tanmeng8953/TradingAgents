from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.prefetch import call_tool, invoke_report, truncate_block
from tradingagents.agents.utils.agent_utils import (
    get_indicators,
    get_instrument_context_from_state,
    get_language_instruction,
    get_stock_data,
    get_verified_market_snapshot,
)
from tradingagents.dataflows.config import get_config


def build_compact_market_system_message() -> str:
    return (
        """You are a technical market analyst. Use tools in this order:
1. Call get_stock_data.
2. Call get_indicators for at most six diverse, non-redundant indicators chosen only from: close_50_sma, close_200_sma, close_10_ema, macd, macds, macdh, rsi, boll, boll_ub, boll_lb, atr, vwma.
3. Before the final report, call get_verified_market_snapshot.

The verified snapshot is the source of truth for exact OHLCV, price levels, and indicator values. Flag conflicts instead of reconciling them yourself. Do not invent historical validation, support or resistance reactions, or percentage moves without dated tool evidence. Return a detailed but concise, actionable trend report and end with a Markdown table of the key evidence."""
        + get_language_instruction()
    )


def create_market_analyst(llm):

    def market_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        instrument_context = get_instrument_context_from_state(state)
        config = get_config()

        if config.get("prefetch_analyst_data"):
            max_chars = int(config.get("prefetch_data_block_char_limit") or 2000)
            price_lookback_days = int(config.get("prefetch_price_lookback_days") or 1500)
            indicator_lookback_days = int(
                config.get("prefetch_indicator_lookback_days") or 365
            )
            start_date = (
                datetime.strptime(current_date, "%Y-%m-%d")
                - timedelta(days=price_lookback_days)
            ).strftime("%Y-%m-%d")
            indicators = [
                "close_50_sma",
                "close_200_sma",
                "macd",
                "rsi",
                "boll",
                "atr",
            ]
            price_block = truncate_block(
                "price data",
                call_tool(get_stock_data, "price data", ticker, start_date, current_date),
                max_chars,
            )
            indicator_block = truncate_block(
                "technical indicators",
                "\n\n".join(
                    call_tool(
                        get_indicators,
                        f"{indicator} indicator",
                        ticker,
                        indicator,
                        current_date,
                        indicator_lookback_days,
                    )
                    for indicator in indicators
                ),
                max_chars,
            )
            snapshot_block = truncate_block(
                "verified market snapshot",
                call_tool(
                    get_verified_market_snapshot,
                    "verified market snapshot",
                    ticker,
                    current_date,
                    30,
                ),
                max_chars,
            )
            system_message = _build_prefetched_market_system_message(
                ticker=ticker,
                start_date=start_date,
                current_date=current_date,
                price_block=price_block,
                indicator_block=indicator_block,
                snapshot_block=snapshot_block,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a technical market analyst. The data below has "
                        "already been collected and compacted locally; do not call "
                        "tools. Write a concise, evidence-grounded trend report and "
                        "end with a Markdown table of key evidence.\n{system_message}"
                        "For your reference, the current date is {current_date}. "
                        "{instrument_context}",
                    ),
                    MessagesPlaceholder(variable_name="messages"),
                ]
            )
            formatted_messages = prompt.partial(
                system_message=system_message,
                current_date=current_date,
                instrument_context=instrument_context,
            ).format_messages(messages=state["messages"])
            report = invoke_report(llm, formatted_messages)
            return {
                "messages": [AIMessage(content=report)],
                "market_report": report,
            }

        tools = [
            get_stock_data,
            get_indicators,
            get_verified_market_snapshot,
        ]

        system_message = (
            """You are a trading assistant tasked with analyzing financial markets. Your role is to select the **most relevant indicators** for a given market condition or trading strategy from the following list. The goal is to choose up to **8 indicators** that provide complementary insights without redundancy. Categories and each category's indicators are:

Moving Averages:
- close_50_sma: 50 SMA: A medium-term trend indicator. Usage: Identify trend direction and serve as dynamic support/resistance. Tips: It lags price; combine with faster indicators for timely signals.
- close_200_sma: 200 SMA: A long-term trend benchmark. Usage: Confirm overall market trend and identify golden/death cross setups. Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries.
- close_10_ema: 10 EMA: A responsive short-term average. Usage: Capture quick shifts in momentum and potential entry points. Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals.

MACD Related:
- macd: MACD: Computes momentum via differences of EMAs. Usage: Look for crossovers and divergence as signals of trend changes. Tips: Confirm with other indicators in low-volatility or sideways markets.
- macds: MACD Signal: An EMA smoothing of the MACD line. Usage: Use crossovers with the MACD line to trigger trades. Tips: Should be part of a broader strategy to avoid false positives.
- macdh: MACD Histogram: Shows the gap between the MACD line and its signal. Usage: Visualize momentum strength and spot divergence early. Tips: Can be volatile; complement with additional filters in fast-moving markets.

Momentum Indicators:
- rsi: RSI: Measures momentum to flag overbought/oversold conditions. Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis.

Volatility Indicators:
- boll: Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. Usage: Acts as a dynamic benchmark for price movement. Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals.
- boll_ub: Bollinger Upper Band: Typically 2 standard deviations above the middle line. Usage: Signals potential overbought conditions and breakout zones. Tips: Confirm signals with other tools; prices may ride the band in strong trends.
- boll_lb: Bollinger Lower Band: Typically 2 standard deviations below the middle line. Usage: Indicates potential oversold conditions. Tips: Use additional analysis to avoid false reversal signals.
- atr: ATR: Averages true range to measure volatility. Usage: Set stop-loss levels and adjust position sizes based on current market volatility. Tips: It's a reactive measure, so use it as part of a broader risk management strategy.

Volume-Based Indicators:
- vwma: VWMA: A moving average weighted by volume. Usage: Confirm trends by integrating price action with volume data. Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses.

- Select indicators that provide diverse and complementary information. Avoid redundancy (e.g., do not select both rsi and stochrsi). Also briefly explain why they are suitable for the given market context. When you tool call, please use the exact name of the indicators provided above as they are defined parameters, otherwise your call will fail. Please make sure to call get_stock_data first to retrieve the CSV that is needed to generate indicators. Then use get_indicators with the specific indicator names.

Before writing the final report, call get_verified_market_snapshot for this ticker and the current date, and treat it as the source of truth for any exact OHLCV, price-level, or indicator-value claim. If another tool's output conflicts with the verified snapshot, flag the discrepancy rather than inventing a reconciled number. Do not claim historical validation, support/resistance bounces, or exact percentage moves unless they are directly supported by tool output with concrete dates and prices.

Write a very detailed and nuanced report of the trends you observe. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."""
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )
        if get_config().get("compact_prompts"):
            system_message = build_compact_market_system_message()

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "market_report": report,
        }

    return market_analyst_node


def _build_prefetched_market_system_message(
    *,
    ticker: str,
    start_date: str,
    current_date: str,
    price_block: str,
    indicator_block: str,
    snapshot_block: str,
) -> str:
    return (
        f"Analyze {ticker} market action from {start_date} to {current_date}.\n\n"
        "## Full price-data summary and recent sample\n"
        f"{price_block}\n\n"
        "## Technical indicator summaries\n"
        f"{indicator_block}\n\n"
        "## Verified market snapshot\n"
        f"{snapshot_block}\n\n"
        "Use the verified snapshot as the source of truth for exact prices and "
        "indicator values. If blocks conflict, flag the conflict instead of "
        "inventing a reconciled number."
        + get_language_instruction()
    )
