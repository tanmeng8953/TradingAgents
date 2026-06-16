from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.analysts.prefetch import call_tool, invoke_report, truncate_block
from tradingagents.agents.utils.agent_utils import (
    get_balance_sheet,
    get_cashflow,
    get_fundamentals,
    get_income_statement,
    get_instrument_context_from_state,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        ticker = state["company_of_interest"]
        current_date = state["trade_date"]
        instrument_context = get_instrument_context_from_state(state)
        config = get_config()

        if config.get("prefetch_analyst_data"):
            max_chars = int(config.get("prefetch_data_block_char_limit") or 2000)
            profile_block = truncate_block(
                "fundamental profile",
                call_tool(get_fundamentals, "fundamental profile", ticker, current_date),
                max_chars,
            )
            statements_block = truncate_block(
                "financial statements",
                "\n\n".join(
                    [
                        "### Balance sheet\n"
                        + call_tool(
                            get_balance_sheet,
                            "balance sheet",
                            ticker,
                            "quarterly",
                            current_date,
                        ),
                        "### Cash flow\n"
                        + call_tool(
                            get_cashflow,
                            "cash flow",
                            ticker,
                            "quarterly",
                            current_date,
                        ),
                        "### Income statement\n"
                        + call_tool(
                            get_income_statement,
                            "income statement",
                            ticker,
                            "quarterly",
                            current_date,
                        ),
                    ]
                ),
                max_chars * 2,
            )
            system_message = _build_prefetched_fundamentals_system_message(
                ticker=ticker,
                current_date=current_date,
                profile_block=profile_block,
                statements_block=statements_block,
            )
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        "You are a fundamentals analyst. The data below has "
                        "already been collected and compacted locally; do not call "
                        "tools. Write a concise, evidence-grounded report with a "
                        "Markdown table of key financial evidence.\n{system_message}"
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
                "fundamentals_report": report,
            }

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements."
            + get_language_instruction(),
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node


def _build_prefetched_fundamentals_system_message(
    *,
    ticker: str,
    current_date: str,
    profile_block: str,
    statements_block: str,
) -> str:
    return (
        f"Analyze {ticker} fundamentals as of {current_date}.\n\n"
        "## Company and valuation profile\n"
        f"{profile_block}\n\n"
        "## Financial statement summaries\n"
        f"{statements_block}\n\n"
        "Focus on profitability, growth, balance-sheet strength, cash flow, "
        "valuation risk, and concrete caveats. Do not invent missing line items."
        + get_language_instruction()
    )
