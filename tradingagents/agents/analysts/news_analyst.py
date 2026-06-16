from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.prefetch import call_tool, invoke_report, truncate_block
from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)
        config = get_config()

        if config.get("prefetch_analyst_data"):
            max_chars = int(config.get("prefetch_data_block_char_limit") or 2000)
            start_date = (
                datetime.strptime(current_date, "%Y-%m-%d") - timedelta(days=7)
            ).strftime("%Y-%m-%d")
            ticker_news_block = truncate_block(
                "ticker news",
                call_tool(get_news, "ticker news", ticker, start_date, current_date),
                max_chars,
            )
            global_news_block = truncate_block(
                "global news",
                call_tool(get_global_news, "global news", current_date, 7, 5),
                max_chars,
            )
            macro_block = truncate_block(
                "macro indicators",
                "\n\n".join(
                    call_tool(
                        get_macro_indicators,
                        f"{indicator} macro indicator",
                        indicator,
                        current_date,
                        365,
                    )
                    for indicator in [
                        "cpi",
                        "unemployment",
                        "fed_funds_rate",
                        "10y_treasury",
                    ]
                ),
                max_chars,
            )
            prediction_block = truncate_block(
                "prediction markets",
                call_tool(
                    get_prediction_markets,
                    "prediction markets",
                    f"{ticker} market catalysts",
                    5,
                ),
                max_chars,
            )
            system_message = _build_prefetched_news_system_message(
                ticker=ticker,
                start_date=start_date,
                current_date=current_date,
                ticker_news_block=ticker_news_block,
                global_news_block=global_news_block,
                macro_block=macro_block,
                prediction_block=prediction_block,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a news and macro analyst. The data below has "
                        "already been collected and compacted locally; do not call "
                        "tools. Write a concise, evidence-grounded catalyst and "
                        "risk report with a Markdown table.\n{system_message}"
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
                "news_report": report,
            }

        tools = [
            get_news,
            get_global_news,
            get_macro_indicators,
            get_prediction_markets,
        ]

        system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for {asset_label}-specific or targeted news searches, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events). Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + get_language_instruction()
        )

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
            "news_report": report,
        }

    return news_analyst_node


def _build_prefetched_news_system_message(
    *,
    ticker: str,
    start_date: str,
    current_date: str,
    ticker_news_block: str,
    global_news_block: str,
    macro_block: str,
    prediction_block: str,
) -> str:
    return (
        f"Analyze {ticker} news and macro context from {start_date} to {current_date}.\n\n"
        "## Ticker news\n"
        f"{ticker_news_block}\n\n"
        "## Global and macro news\n"
        f"{global_news_block}\n\n"
        "## Macro indicators\n"
        f"{macro_block}\n\n"
        "## Prediction markets\n"
        f"{prediction_block}\n\n"
        "Separate confirmed events from market opinions. Flag missing or unavailable "
        "sources instead of filling gaps."
        + get_language_instruction()
    )
