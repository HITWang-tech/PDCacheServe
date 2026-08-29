from fastapi.testclient import TestClient

from pdserve.api import create_app
from pdserve.control import ControlPlane


def worker(worker_id, role):
    return {
        "worker_id": worker_id,
        "role": role,
        "endpoint": f"http://{worker_id}:8000",
        "gpu_model": "RTX-4090-24GB",
        "layout": {"model_id": "qwen"},
    }


def test_api_registers_workers_routes_and_completes_in_simulation():
    client = TestClient(create_app(ControlPlane.create()))
    assert client.post("/v1/workers", json=worker("p0", "prefill")).status_code == 200
    assert client.post("/v1/workers", json=worker("d0", "decode")).status_code == 200
    request = {
        "model_id": "qwen",
        "input_tokens": 256,
        "max_new_tokens": 32,
        "ttft_slo_ms": 2000,
        "tpot_slo_ms": 80,
        "payload": {"mock_text": "hello"},
    }
    route = client.post("/v1/routes", json=request)
    assert route.status_code == 200
    assert route.json()["admitted"] is True
    completion = client.post("/v1/completions", json=request)
    assert completion.status_code == 200
    assert completion.json()["choices"][0]["text"] == "hello"


def test_api_rejects_request_without_healthy_pair():
    client = TestClient(create_app(ControlPlane.create()))
    response = client.post(
        "/v1/completions",
        json={"model_id": "qwen", "input_tokens": 10, "max_new_tokens": 10},
    )
    assert response.status_code == 429


def test_api_heartbeat_rejects_unknown_worker():
    client = TestClient(create_app(ControlPlane.create()))
    assert client.post("/v1/workers/missing/heartbeat", json={}).status_code == 404
