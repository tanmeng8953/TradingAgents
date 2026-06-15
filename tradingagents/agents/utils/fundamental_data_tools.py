import csv
import io

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.interface import route_to_vendor


def _limit_statement(
    payload: str,
    max_rows: int | None,
    max_periods: int | None,
) -> str:
    if not max_rows or max_rows < 1 or not max_periods or max_periods < 1:
        return payload

    lines = payload.splitlines()
    try:
        header_index = next(
            index for index, line in enumerate(lines) if line.startswith(",")
        )
    except StopIteration:
        return payload

    table = list(csv.reader(io.StringIO("\n".join(lines[header_index:]))))
    if len(table) < 2:
        return payload

    header = table[0]
    metrics = table[1:]
    if len(metrics) <= max_rows and len(header) - 1 <= max_periods:
        return payload

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header[: max_periods + 1])
    for row in metrics[:max_rows]:
        writer.writerow(row[: max_periods + 1])

    metadata = "\n".join(lines[:header_index]).rstrip()
    return "\n".join(
        [
            metadata,
            "",
            (
                f"# Showing {min(max_rows, len(metrics))} of {len(metrics)} metrics "
                f"across the {min(max_periods, len(header) - 1)} most recent periods "
                "to fit the configured model context."
            ),
            output.getvalue().rstrip(),
        ]
    ).strip()


def _statement_result(method: str, ticker: str, freq: str, curr_date: str) -> str:
    payload = route_to_vendor(method, ticker, freq, curr_date)
    config = get_config()
    return _limit_statement(
        payload,
        config.get("fundamental_statement_output_rows"),
        config.get("fundamental_statement_output_periods"),
    )


@tool
def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing comprehensive fundamental data
    """
    return route_to_vendor("get_fundamentals", ticker, curr_date)


@tool
def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve balance sheet data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing balance sheet data
    """
    return _statement_result("get_balance_sheet", ticker, freq, curr_date)


@tool
def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve cash flow statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing cash flow statement data
    """
    return _statement_result("get_cashflow", ticker, freq, curr_date)


@tool
def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "reporting frequency: annual/quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"] = None,
) -> str:
    """
    Retrieve income statement data for a given ticker symbol.
    Uses the configured fundamental_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
        freq (str): Reporting frequency: annual/quarterly (default quarterly)
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A formatted report containing income statement data
    """
    return _statement_result("get_income_statement", ticker, freq, curr_date)
