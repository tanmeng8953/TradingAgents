import re
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.interface import route_to_vendor

_DATED_VALUE = re.compile(r"^\d{4}-\d{2}-\d{2}:")
_NUMERIC_VALUE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def _as_float(value: str) -> float | None:
    match = _NUMERIC_VALUE.search(value)
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def _format_number(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}".rstrip("0").rstrip(".")


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


def _summarize_indicator_values(payload: str, max_rows: int | None) -> str:
    sample_rows = max_rows if max_rows and max_rows > 0 else 3
    lines = payload.splitlines()
    dated_lines = [line for line in lines if _DATED_VALUE.match(line)]
    if not dated_lines:
        return _limit_indicator_values(payload, sample_rows)

    first_data_index = next(
        index for index, line in enumerate(lines) if _DATED_VALUE.match(line)
    )
    prefix = "\n".join(lines[:first_data_index]).rstrip()
    entries = []
    for line in dated_lines:
        date, value = line.split(":", 1)
        entries.append(
            {
                "date": date,
                "line": line,
                "value": _as_float(value),
            }
        )
    chronological = sorted(entries, key=lambda item: item["date"])
    oldest = chronological[0]
    latest = chronological[-1]
    latest_value = latest["value"]
    oldest_value = oldest["value"]
    full_change = None
    if latest_value is not None and oldest_value is not None:
        full_change = latest_value - oldest_value

    recent = [
        entry["line"]
        for entry in sorted(entries, key=lambda item: item["date"], reverse=True)[
            :sample_rows
        ]
    ]
    blocks = []
    if prefix:
        blocks.extend([prefix, ""])
    blocks.extend(
        [
            (
                "# Full indicator summary generated from all "
                f"{len(dated_lines)} dated values before prompt compaction."
            ),
            f"# Date range: {oldest['date']} to {latest['date']}",
            f"# Latest value: {_format_number(latest_value)}",
            f"# Change over full series: {_format_number(full_change)}",
            "",
            (
                f"# Showing the {min(sample_rows, len(dated_lines))} most recent "
                f"values out of {len(dated_lines)} to fit the configured model context."
            ),
            *recent,
        ]
    )
    return "\n".join(blocks).strip()


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
            config = get_config()
            max_rows = config.get("indicator_output_rows")
            results.append(
                _summarize_indicator_values(
                    payload,
                    max_rows,
                )
                if config.get("full_data_summary_mode")
                else _limit_indicator_values(
                    payload,
                    max_rows,
                )
            )
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)
