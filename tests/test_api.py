import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from tests.test_predictions import GOOD  # noqa: E402


@pytest.fixture()
def client(env, monkeypatch):
    from api import main
    from src import app_utils
    monkeypatch.setattr(main, "_bundle", app_utils.load_bundle(env["artifacts"], env["cfg"]))
    return TestClient(main.app)


def test_health(client):
    assert client.get("/health").json()["model_loaded"] is True


def test_predict_and_validation(client):
    body = {**GOOD, "TotalCharges": None}
    r = client.post("/predict", json=body)
    assert r.status_code == 200 and 0 <= r.json()["churn_probability"] <= 1
    assert client.post("/predict", json={**body, "Contract": "Weekly"}).status_code == 422


def test_predict_batch(client):
    body = {**GOOD, "TotalCharges": None}
    r = client.post("/predict-batch", json={"records": [body, body]})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 2
