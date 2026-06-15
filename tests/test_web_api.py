import pytest
from fastapi.testclient import TestClient

from tradingagents.web.app import create_app
from tradingagents.web.store import RunStore


class RecordingDispatcher:
    def __init__(self):
        self.submissions = []

    def submit(self, run_id, request):
        self.submissions.append((run_id, request))


@pytest.fixture()
def web_client(tmp_path):
    dispatcher = RecordingDispatcher()
    app = create_app(run_store=RunStore(tmp_path), dispatcher=dispatcher)
    return TestClient(app), dispatcher


@pytest.mark.unit
def test_health_declares_research_only_boundary(web_client):
    client, _ = web_client

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "ok": True,
        "service": "tradingagents-web",
        "broker_execution": False,
        "paper_only": True,
    }


@pytest.mark.unit
def test_create_list_and_get_run(web_client):
    client, dispatcher = web_client

    created_response = client.post(
        "/api/v1/runs",
        json={
            "ticker": "AAPL",
            "analysis_date": "2026-06-12",
            "analysts": ["market", "news"],
            "model_route": "local_5090",
            "debate_rounds": 1,
        },
    )

    assert created_response.status_code == 202
    created = created_response.json()
    assert created["status"] == "queued"
    assert created["request"]["ticker"] == "AAPL"
    assert dispatcher.submissions == [(created["run_id"], created["request"])]

    listed = client.get("/api/v1/runs").json()["runs"]
    assert listed[0]["run_id"] == created["run_id"]

    fetched = client.get(f"/api/v1/runs/{created['run_id']}")
    assert fetched.status_code == 200
    assert fetched.json()["request"]["model_route"] == "local_5090"


@pytest.mark.unit
def test_cloud_route_requires_explicit_confirmation(web_client):
    client, dispatcher = web_client

    response = client.post(
        "/api/v1/runs",
        json={
            "ticker": "AAPL",
            "analysis_date": "2026-06-12",
            "model_route": "cloud",
            "llm_provider": "openai",
        },
    )

    assert response.status_code == 422
    assert dispatcher.submissions == []


@pytest.mark.unit
def test_home_page_exposes_primary_research_workflow(web_client):
    client, _ = web_client

    response = client.get("/")

    assert response.status_code == 200
    assert "TradingAgents Local Research Console" in response.text
    assert 'name="ticker"' in response.text
    assert 'name="model_route"' in response.text
    assert "Analysts" in response.text
    assert "Bull vs Bear" in response.text
    assert "Final decision" in response.text
    assert "run.reports?.final_trade_decision" in response.text
    assert "reportForTab" in response.text
    assert "No standalone analyst reports were persisted." in response.text
    assert 'querySelectorAll(".tab")' in response.text
    assert "Research only" in response.text


@pytest.mark.unit
def test_readiness_endpoint_reports_local_model_state(tmp_path):
    expected = {
        "status": "unavailable",
        "reachable": False,
        "model_route": "local_5090",
        "message": "Configured 5090 endpoint is offline.",
    }
    app = create_app(
        run_store=RunStore(tmp_path),
        dispatcher=RecordingDispatcher(),
        readiness_probe=lambda: expected,
    )

    response = TestClient(app).get("/api/v1/readiness")

    assert response.status_code == 200
    assert response.json() == expected


@pytest.mark.unit
def test_home_page_surfaces_model_readiness(web_client):
    client, _ = web_client

    response = client.get("/")

    assert 'id="model-readiness"' in response.text
    assert 'id="model-readiness-detail"' in response.text
    assert 'fetch("/api/v1/readiness")' in response.text
    assert "This run timed out before local model readiness checks were added." in response.text
