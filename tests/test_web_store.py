import json

import pytest
from langchain_core.messages import AIMessage

from tradingagents.web.store import RunStore


@pytest.mark.unit
def test_run_store_persists_sanitized_requests_and_updates(tmp_path):
    store = RunStore(tmp_path)

    created = store.create(
        {
            "ticker": "AAPL",
            "analysis_date": "2026-06-12",
            "model_route": "local_5090",
            "OPENAI_API_KEY": "must-not-be-written",
        }
    )

    assert created["status"] == "queued"
    assert created["request"]["ticker"] == "AAPL"
    assert "OPENAI_API_KEY" not in created["request"]

    updated = store.update(
        created["run_id"],
        status="running",
        progress={"current_stage": "analysts"},
    )
    assert updated["progress"]["current_stage"] == "analysts"

    persisted = json.loads((tmp_path / created["run_id"] / "run.json").read_text(encoding="utf-8"))
    assert persisted["status"] == "running"
    assert "must-not-be-written" not in json.dumps(persisted)


@pytest.mark.unit
def test_run_store_lists_newest_first(tmp_path):
    store = RunStore(tmp_path)
    first = store.create({"ticker": "AAPL", "analysis_date": "2026-06-11"})
    second = store.create({"ticker": "MSFT", "analysis_date": "2026-06-12"})

    runs = store.list()

    assert [run["run_id"] for run in runs] == [second["run_id"], first["run_id"]]


@pytest.mark.unit
def test_run_store_serializes_langchain_messages(tmp_path):
    store = RunStore(tmp_path)
    created = store.create({"ticker": "AAPL", "analysis_date": "2026-06-12"})

    updated = store.update(
        created["run_id"],
        reports={"debate": AIMessage(content="Bull case.")},
    )

    assert updated["reports"]["debate"]["content"] == "Bull case."
