from __future__ import annotations

import os
from importlib.resources import files
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

from tradingagents.web.runner import ThreadedRunDispatcher
from tradingagents.web.store import RunStore


class RunRequest(BaseModel):
    ticker: str = Field(min_length=1, max_length=32)
    analysis_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    analysts: list[Literal["market", "social", "news", "fundamentals"]] = Field(
        default_factory=lambda: ["market", "social", "news", "fundamentals"]
    )
    model_route: Literal["local_5090", "cloud"] = "local_5090"
    llm_provider: str = "openai_compatible"
    backend_url: str = ""
    quick_think_llm: str = ""
    deep_think_llm: str = ""
    debate_rounds: int = Field(default=1, ge=1, le=3)
    output_language: str = "Chinese"
    checkpoint_enabled: bool = True
    cloud_confirmed: bool = False

    @model_validator(mode="after")
    def validate_cloud_confirmation(self):
        if self.model_route == "cloud" and not self.cloud_confirmed:
            raise ValueError("cloud model route requires explicit confirmation")
        return self


def default_runs_dir() -> Path:
    return Path(
        os.getenv(
            "TRADINGAGENTS_RUNS_DIR",
            str(Path.home() / ".tradingagents" / "runs"),
        )
    ).expanduser()


def create_app(
    *,
    run_store: RunStore | None = None,
    dispatcher=None,
) -> FastAPI:
    store = run_store or RunStore(default_runs_dir())
    run_dispatcher = dispatcher or ThreadedRunDispatcher(store)
    app = FastAPI(title="TradingAgents Local Research Console")
    app.state.run_store = store
    app.state.dispatcher = run_dispatcher

    @app.get("/health")
    def health():
        return {
            "ok": True,
            "service": "tradingagents-web",
            "broker_execution": False,
            "paper_only": True,
        }

    @app.get("/", response_class=HTMLResponse)
    def home():
        return files("tradingagents.web").joinpath("static/index.html").read_text(encoding="utf-8")

    @app.post("/api/v1/runs", status_code=202)
    def create_run(payload: RunRequest):
        request = payload.model_dump()
        request["ticker"] = request["ticker"].strip().upper()
        run = store.create(request)
        run_dispatcher.submit(run["run_id"], run["request"])
        return run

    @app.get("/api/v1/runs")
    def list_runs(limit: int = Query(default=50, ge=1, le=200)):
        return {"runs": store.list(limit=limit)}

    @app.get("/api/v1/runs/{run_id}")
    def get_run(run_id: str):
        try:
            return store.get(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc

    return app
