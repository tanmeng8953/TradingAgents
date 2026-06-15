import re

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.interface import route_to_vendor


_DATED_VALUE = re.compile(r"^\d{4}-\d{2}-\d{2}:")


def _limit_indicator_values(payload: str, max_rows: int | None) -> str:
    if not max_rows or max_rows < 1:
        return payload

    lines = payload.splitlines()
    dated_lines = [line for line in lines if _DATED_VALUE.match(line)]
    if len(dated_lines) <= max_rows:
        return payload

    first_data_index = next(
        index for index, line in enumerate(lines) if _DATED_VALUE.match(line)
    )
    prefix = "\n".join(lines[:first_data_index]).rstrip()
    recent = sorted(dated_lines, reverse=True)[:max_rows]
    return "\n".join(
        [
            prefix,
            "",
            (
                f"# Showing the {max_rows} most recent values out of "
                f"{len(dated_lines)} to fit the configured model context."
            ),
            *recent,
        ]
    ).strip()


@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve a single technical indicator for a given ticker symbol.
    Uses the configured technical_indicators vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        indicator (str): A single technical indicator name, e.g. 'rsi', 'macd'. Call this tool once per indicator.
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        look_back_days (int): How many days to look back, default is 30
    Returns:
        str: A formatted dataframe containing the technical indicators for the specified ticker symbol and indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            payload = route_to_vendor(
                "get_indicators",
                symbol,
                ind,
                curr_date,
                look_back_days,
            )
            results.append(
                _limit_indicator_values(
                    payload,
                    get_config().get("indicator_output_rows"),
                )
            )
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)
