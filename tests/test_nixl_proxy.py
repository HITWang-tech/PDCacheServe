from fastapi.testclient import TestClient

from pdserve.nixl_proxy import create_proxy


class FakeResponse:
    is_error = False
    status_code = 200
    text = ""

    def json(self):
        return {"kv_transfer_params": {"remote_engine_id": "prefill-0"}}

    async def aiter_bytes(self):
        yield b"data: {\"choices\":[]}\n\n"

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None


class FakeClient:
    def __init__(self):
        self.stream_call = None

    async def post(self, *_args, **_kwargs):
        return FakeResponse()

    def stream(self, method, url, **_kwargs):
        self.stream_call = (method, url)
        return FakeResponse()

    async def aclose(self):
        return None


def test_nixl_proxy_health_identifies_connector():
    app = create_proxy("http://prefill.invalid", "http://decode.invalid")
    with TestClient(app) as client:
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "connector": "NixlConnector"}


def test_streaming_decode_uses_post_method():
    app = create_proxy("http://prefill.invalid", "http://decode.invalid")
    with TestClient(app) as client:
        app.state.prefill = FakeClient()
        decoder = FakeClient()
        app.state.decode = decoder
        with client.stream(
            "POST",
            "/v1/completions",
            json={"model": "model", "prompt": "hello", "max_tokens": 2, "stream": True},
        ) as response:
            assert response.status_code == 200
            assert b"data:" in response.read()

    assert decoder.stream_call == ("POST", "/v1/completions")
