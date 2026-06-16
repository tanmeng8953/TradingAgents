from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage


def call_tool(tool: Any, label: str, *args: Any) -> str:
    try:
        func = getattr(tool, "func", None)
        if func is not None:
            return str(func(*args))
        return str(tool.invoke(args))
    except Exception as exc:
        return f"<{label} unavailable: {exc.__class__.__name__}>"


def truncate_block(label: str, value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if not text:
        return f"<{label} empty>"
    if len(text) <= max_chars:
        return text
    omitted = len(text) - max_chars
    return (
        text[:max_chars].rstrip()
        + f"\n\n# {label} truncated to {max_chars} chars; {omitted} chars omitted."
    )


def invoke_report(llm: Any, messages: list[Any]) -> str:
    result = llm.invoke(messages)
    if isinstance(result, AIMessage):
        return str(result.content)
    return str(getattr(result, "content", result))
