from fastapi.testclient import TestClient

from pdserve.nixl_proxy import create_proxy


def test_nixl_proxy_health_identifies_connector():
    app = create_proxy("http://prefill.invalid", "http://decode.invalid")
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "connector": "NixlConnector"}
