"""OpenAI-compatible 1P1D proxy for vLLM's native NIXL connector.

The handshake follows vLLM's Apache-2.0 licensed NixlConnector integration
proxy: prefill one token, forward the returned ``kv_transfer_params`` with the
same request ID, and stream the decoder response back to the caller.
"""

from __future__ import annotations

import argparse
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

try:
    import httpx
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
except ImportError as exc:  # pragma: no cover - optional runtime dependencies
    raise RuntimeError("install pdcacheserve[api]") from exc


def create_proxy(prefill_url: str, decode_url: str) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        app.state.prefill = httpx.AsyncClient(
            base_url=prefill_url.rstrip("/"), timeout=None, limits=limits
        )
        app.state.decode = httpx.AsyncClient(
            base_url=decode_url.rstrip("/"), timeout=None, limits=limits
        )
        try:
            yield
        finally:
            await app.state.prefill.aclose()
            await app.state.decode.aclose()

    app = FastAPI(title="PDCacheServe native NIXL proxy", lifespan=lifespan)

    async def prepare(request: Request, path: str) -> tuple[dict, str, bool]:
        original = await request.json()
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        prefill_body = dict(original)
        prefill_body["stream"] = False
        prefill_body["max_tokens"] = 1
        if "max_completion_tokens" in prefill_body:
            prefill_body["max_completion_tokens"] = 1
        prefill_body.pop("stream_options", None)
        prefill_body["kv_transfer_params"] = {
            "do_remote_decode": True,
            "do_remote_prefill": False,
            "remote_engine_id": None,
            "remote_block_ids": None,
            "remote_host": None,
            "remote_port": None,
        }
        response = await app.state.prefill.post(
            path, json=prefill_body, headers={"X-Request-Id": request_id}
        )
        if response.is_error:
            raise HTTPException(response.status_code, response.text)
        transfer = response.json().get("kv_transfer_params")
        if not transfer:
            raise HTTPException(502, "prefiller did not return kv_transfer_params")
        decode_body = dict(original)
        decode_body["kv_transfer_params"] = transfer
        return decode_body, request_id, bool(original.get("stream", False))

    async def forward(request: Request, path: str):
        body, request_id, requested_stream = await prepare(request, path)
        headers = {"X-Request-Id": request_id}
        if not requested_stream:
            response = await app.state.decode.post(path, json=body, headers=headers)
            if response.is_error:
                raise HTTPException(response.status_code, response.text)
            return JSONResponse(response.json())

        async def generate() -> AsyncIterator[bytes]:
            async with app.state.decode.stream(
                "POST", path, json=body, headers=headers
            ) as response:
                if response.is_error:
                    content = await response.aread()
                    raise HTTPException(response.status_code, content.decode())
                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(generate(), media_type="text/event-stream")

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok", "connector": "NixlConnector"}

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await forward(request, "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await forward(request, "/v1/chat/completions")

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:8100")
    parser.add_argument("--decode-url", default="http://127.0.0.1:8200")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        create_proxy(args.prefill_url, args.decode_url), host=args.host, port=args.port
    )


if __name__ == "__main__":
    main()
