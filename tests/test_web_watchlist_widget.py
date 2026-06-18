from __future__ import annotations

from pathlib import Path


def test_static_console_embeds_research_only_watchlist_widget():
    html = (Path(__file__).resolve().parents[1] / "tradingagents" / "web" / "static" / "index.html").read_text(
        encoding="utf-8"
    )

    assert "自选股票实时价格" in html
    assert "research_only" in html
    assert "每只股票下方直接显示自己的 K线图" in html
    assert "按钮增加/删除自选股" in html
    assert "height: 1080px" in html
    assert "http://127.0.0.1:8765/workbench/crawler-widget/trading" in html
