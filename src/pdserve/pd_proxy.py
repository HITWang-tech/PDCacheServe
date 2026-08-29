from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict

try:
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import Response, StreamingResponse
except ImportError as exc:  # pragma: no cover - optional dependency
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
        yield
        await app.state.prefill.aclose()
        await app.state.decode.aclose()

    app = FastAPI(title="PDCacheServe P/D Proxy", lifespan=lifespan)

    async def forward(path: str, body: Dict[str, object]):
        prefill_body = dict(body)
        prefill_body["stream"] = False
        prefill_body.pop("stream_options", None)
        prefill_body["max_tokens"] = 1
        if "max_completion_tokens" in prefill_body:
            prefill_body["max_completion_tokens"] = 1
        prefill_response = await app.state.prefill.post(path, json=prefill_body)
        prefill_response.raise_for_status()

        if not body.get("stream", False):
            decode_response = await app.state.decode.post(path, json=body)
            return Response(
                content=decode_response.content,
                status_code=decode_response.status_code,
                media_type=decode_response.headers.get("content-type", "application/json"),
            )

        async def decode_stream() -> AsyncIterator[bytes]:
            async with app.state.decode.stream("POST", path, json=body) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes():
                    yield chunk

        return StreamingResponse(decode_stream(), media_type="text/event-stream")

    @app.get("/healthz")
    async def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await forward("/v1/completions", await request.json())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        return await forward("/v1/chat/completions", await request.json())

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
