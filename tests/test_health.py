from starlette.testclient import TestClient

from ecoface_lite.api.main import create_app


def test_health():
    with TestClient(create_app()) as client:
        r = client.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
