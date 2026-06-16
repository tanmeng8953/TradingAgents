import csv
import io
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.interface import route_to_vendor


def _as_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except ValueError:
        return None


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}".rstrip("0").rstrip(".")


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


def _summarize_csv_rows(payload: str, max_rows: int | None) -> str:
    sample_rows = max_rows if max_rows and max_rows > 0 else 50
    rows = list(csv.reader(io.StringIO(payload)))
    if len(rows) < 2:
        return _limit_csv_rows(payload, sample_rows)

    header = rows[0]
    data_rows = rows[1:]
    columns = {name.strip().lower(): index for index, name in enumerate(header)}

    def column(name: str) -> int | None:
        return columns.get(name.lower())

    def cell(row: list[str], index: int | None) -> str | None:
        if index is None or index >= len(row):
            return None
        return row[index]

    date_index = column("date")
    close_index = column("close")
    high_index = column("high")
    low_index = column("low")
    volume_index = column("volume")

    first_row = data_rows[0]
    latest_row = data_rows[-1]
    first_close = _as_float(cell(first_row, close_index))
    latest_close = _as_float(cell(latest_row, close_index))
    close_return = None
    if first_close not in (None, 0) and latest_close is not None:
        close_return = ((latest_close / first_close) - 1) * 100

    highs = [
        value
        for value in (_as_float(cell(row, high_index)) for row in data_rows)
        if value is not None
    ]
    lows = [
        value
        for value in (_as_float(cell(row, low_index)) for row in data_rows)
        if value is not None
    ]
    volumes = [
        value
        for value in (_as_float(cell(row, volume_index)) for row in data_rows)
        if value is not None
    ]

    summary = [
        (
            f"# Full price-data summary generated from all {len(data_rows)} rows "
            "before prompt compaction."
        ),
        (
            f"# Date range: {cell(first_row, date_index) or 'n/a'} to "
            f"{cell(latest_row, date_index) or 'n/a'}"
        ),
        (
            f"# First close: {_format_number(first_close)}; "
            f"latest close: {_format_number(latest_close)}; "
            f"Close return: {close_return:.2f}%"
            if close_return is not None
            else (
                f"# First close: {_format_number(first_close)}; "
                f"latest close: {_format_number(latest_close)}; Close return: n/a"
            )
        ),
        (
            f"# Period high: {_format_number(max(highs) if highs else None)}; "
            f"period low: {_format_number(min(lows) if lows else None)}; "
            f"average volume: "
            f"{_format_number(sum(volumes) / len(volumes) if volumes else None)}"
        ),
    ]
    return "\n".join([*summary, "", _limit_csv_rows(payload, sample_rows)])


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
    config = get_config()
    max_rows = config.get("stock_data_output_rows")
    if config.get("full_data_summary_mode"):
        return _summarize_csv_rows(payload, max_rows)
    return _limit_csv_rows(payload, max_rows)
