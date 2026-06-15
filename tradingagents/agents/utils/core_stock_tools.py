import csv
import io

from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.config import get_config


def _limit_csv_rows(payload: str, max_rows: int | None) -> str:
    if not max_rows or max_rows < 1:
        return payload

    rows = list(csv.reader(io.StringIO(payload)))
    if len(rows) <= max_rows + 1 or not rows:
        return payload

    data_rows = rows[1:]
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(rows[0])
    writer.writerows(data_rows[-max_rows:])
    return (
        f"# Showing the {max_rows} most recent rows out of {len(data_rows)} "
        "to fit the configured model context.\n"
        f"{output.getvalue().rstrip()}"
    )


@tool
def get_stock_data(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
) -> str:
    """
    Retrieve stock price data (OHLCV) for a given ticker symbol.
    Uses the configured core_stock_apis vendor.
    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted dataframe containing the stock price data for the specified ticker symbol in the specified date range.
    """
    payload = route_to_vendor("get_stock_data", symbol, start_date, end_date)
    return _limit_csv_rows(payload, get_config().get("stock_data_output_rows"))
