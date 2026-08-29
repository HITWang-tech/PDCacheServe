"""OpenAI-compatible 1P1D proxy for LMCache's NIXL transfer channel.

The request protocol follows LMCache 0.5.x's Apache-2.0 licensed
``examples/disagg_prefill/disagg_proxy_server.py`` implementation. This module
keeps the deployment intentionally small (one prefiller and one decoder) while
preserving the production-critical transfer handshake.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

try:
    import httpx
    import msgspec
    import zmq
    import zmq.asyncio
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    from lmcache.v1.storage_backend.pd_backend import PDMsg, ProxyNotif
except ImportError as exc:  # pragma: no cover - optional runtime dependencies
    raise RuntimeError("install pdcacheserve[api] and LMCache") from exc


class WeightedSemaphore:
    """Limit concurrent requests by the KV-transfer chunks they occupy."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.available = capacity
        self._condition = asyncio.Condition()

    async def acquire(self, slots: int) -> None:
        if slots > self.capacity:
            raise ValueError(f"request needs {slots} transfer slots; capacity is {self.capacity}")
        async with self._condition:
            await self._condition.wait_for(lambda: self.available >= slots)
            self.available -= slots

    async def release(self, slots: int) -> None:
        async with self._condition:
            self.available = min(self.capacity, self.available + slots)
            self._condition.notify_all()


def kv_bytes_per_token(model: str) -> int:
    """Calculate bytes occupied by all K/V tensors for one token."""
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model)
    layers = int(config.num_hidden_layers)
    kv_heads = int(getattr(config, "num_key_value_heads", config.num_attention_heads))
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    dtype_bytes = 4 if "float32" in str(getattr(config, "torch_dtype", "")) else 2
    return 2 * layers * kv_heads * head_dim * dtype_bytes


@dataclass(frozen=True)
class ProxyConfig:
    prefill_url: str
    decode_url: str
    decoder_host: str
    decoder_init_port: int
    decoder_alloc_port: int
    notification_host: str
    notification_port: int
    model: str
    pd_buffer_size: int = 2 * 1024 * 1024 * 1024
    chunk_size: int = 256
    transfer_timeout_seconds: float = 120.0


def create_proxy(config: ProxyConfig) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limits = httpx.Limits(max_connections=None, max_keepalive_connections=None)
        app.state.prefill = httpx.AsyncClient(
            base_url=config.prefill_url.rstrip("/"), timeout=None, limits=limits
        )
        app.state.decode = httpx.AsyncClient(
            base_url=config.decode_url.rstrip("/"), timeout=None, limits=limits
        )
        app.state.notifications = {}
        app.state.request_counter = 0
        bytes_per_chunk = kv_bytes_per_token(config.model) * config.chunk_size
        app.state.buffer = WeightedSemaphore(config.pd_buffer_size // bytes_per_chunk)
        app.state.zmq_context = zmq.asyncio.Context()
        app.state.zmq_task = asyncio.create_task(notification_server(app))
        try:
            yield
        finally:
            app.state.zmq_task.cancel()
            await asyncio.gather(app.state.zmq_task, return_exceptions=True)
            app.state.zmq_context.term()
            await app.state.prefill.aclose()
            await app.state.decode.aclose()

    app = FastAPI(title="PDCacheServe P/D Proxy", lifespan=lifespan)

    async def notification_server(current_app: FastAPI) -> None:
        socket = current_app.state.zmq_context.socket(zmq.PULL)
        socket.bind(f"tcp://{config.notification_host}:{config.notification_port}")
        try:
            while True:
                payload = await socket.recv()
                message = msgspec.msgpack.decode(payload, type=PDMsg)
                if isinstance(message, ProxyNotif):
                    event = current_app.state.notifications.get(message.req_id)
                    if event is not None:
                        event.set()
        finally:
            socket.close(linger=0)

    async def post(client: httpx.AsyncClient, path: str, body: dict) -> httpx.Response:
        response = await client.post(path, json=body)
        if response.is_error:
            raise HTTPException(response.status_code, response.text)
        return response

    async def stream(client: httpx.AsyncClient, path: str, body: dict) -> AsyncIterator[bytes]:
        async with client.stream("POST", path, json=body) as response:
            if response.is_error:
                content = await response.aread()
                raise HTTPException(response.status_code, content.decode())
            async for chunk in response.aiter_bytes():
                yield chunk

    async def prepare(request: Request, chat: bool) -> tuple:
        body = await request.json()
        original_stream = bool(body.get("stream", False))
        max_tokens_key = (
            "max_completion_tokens" if "max_completion_tokens" in body else "max_tokens"
        )
        original_max_tokens = int(body.get(max_tokens_key, 16))
        if original_max_tokens < 2:
            raise HTTPException(400, "P/D requests require max_tokens >= 2")

        tokenize_body = {"messages": body["messages"]} if chat else {"prompt": body["prompt"]}
        tokenized = (await post(app.state.prefill, "/tokenize", tokenize_body)).json()
        tokens = tokenized["tokens"]
        slots = math.ceil(len(tokens) / config.chunk_size)
        await app.state.buffer.acquire(slots)

        app.state.request_counter += 1
        request_id = str(app.state.request_counter)
        ready = asyncio.Event()
        app.state.notifications[request_id] = ready

        prefill_body = dict(body)
        prefill_body.pop("messages", None)
        prefill_body["prompt"] = tokens
        prefill_body["max_tokens"] = 1
        if max_tokens_key == "max_completion_tokens":
            prefill_body["max_completion_tokens"] = 1
        prefill_body["stream"] = False
        stream_options = prefill_body.pop("stream_options", None)
        prefill_body["kv_transfer_params"] = {
            "ret_first_tok": True,
            "disagg_spec": {
                "req_id": request_id,
                "receiver_host": config.decoder_host,
                "receiver_init_port": [config.decoder_init_port],
                "receiver_alloc_port": [config.decoder_alloc_port],
            },
        }
        try:
            first = (await post(app.state.prefill, "/v1/completions", prefill_body)).json()
        except Exception:
            app.state.notifications.pop(request_id, None)
            await app.state.buffer.release(slots)
            raise

        decode_body = dict(prefill_body)
        decode_body.pop("kv_transfer_params")
        decode_body["prompt"] = tokens + [first["kv_transfer_params"]["first_tok"]]
        decode_body["max_tokens"] = original_max_tokens - 1
        if max_tokens_key == "max_completion_tokens":
            decode_body["max_completion_tokens"] = original_max_tokens - 1
        decode_body["stream"] = original_stream
        if stream_options is not None:
            decode_body["stream_options"] = stream_options
        return first, decode_body, ready, request_id, slots, original_stream

    async def wait_and_release(ready: asyncio.Event, request_id: str, slots: int) -> None:
        try:
            await asyncio.wait_for(ready.wait(), config.transfer_timeout_seconds)
        except TimeoutError as exc:
            raise HTTPException(504, "timed out waiting for KV transfer") from exc
        finally:
            app.state.notifications.pop(request_id, None)
            await app.state.buffer.release(slots)

    def first_completion_chunk(first: dict) -> bytes:
        chunk = {
            "id": first["id"],
            "object": "text_completion",
            "created": first["created"],
            "model": first["model"],
            "choices": [
                {
                    "index": 0,
                    "text": first["choices"][0]["text"],
                    "logprobs": None,
                    "finish_reason": None,
                    "stop_reason": None,
                }
            ],
            "usage": None,
        }
        return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n".encode()

    @app.get("/healthz")
    async def healthz() -> dict:
        return {
            "status": "ok",
            "transfer_slots_available": app.state.buffer.available,
            "transfer_slots_total": app.state.buffer.capacity,
        }

    @app.post("/v1/completions")
    async def completions(request: Request):
        first, decode_body, ready, request_id, slots, requested_stream = await prepare(
            request, chat=False
        )
        if requested_stream:

            async def generate() -> AsyncIterator[bytes]:
                yield first_completion_chunk(first)
                await wait_and_release(ready, request_id, slots)
                async for chunk in stream(app.state.decode, "/v1/completions", decode_body):
                    yield chunk

            return StreamingResponse(generate(), media_type="text/event-stream")

        await wait_and_release(ready, request_id, slots)
        decoded = (await post(app.state.decode, "/v1/completions", decode_body)).json()
        decoded["choices"][0]["text"] = first["choices"][0]["text"] + decoded["choices"][0]["text"]
        return JSONResponse(decoded)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        first, decode_body, ready, request_id, slots, requested_stream = await prepare(
            request, chat=True
        )
        await wait_and_release(ready, request_id, slots)
        if requested_stream:

            async def generate() -> AsyncIterator[bytes]:
                yield first_completion_chunk(first)
                async for chunk in stream(app.state.decode, "/v1/completions", decode_body):
                    yield chunk

            return StreamingResponse(generate(), media_type="text/event-stream")
        decoded = (await post(app.state.decode, "/v1/completions", decode_body)).json()
        text = first["choices"][0]["text"] + decoded["choices"][0]["text"]
        return JSONResponse(
            {
                "id": decoded["id"],
                "object": "chat.completion",
                "created": decoded["created"],
                "model": decoded["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": decoded["choices"][0].get("finish_reason"),
                    }
                ],
                "usage": decoded.get("usage"),
            }
        )

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prefill-url", default="http://127.0.0.1:8100")
    parser.add_argument("--decode-url", default="http://127.0.0.1:8200")
    parser.add_argument("--decoder-host", default="127.0.0.1")
    parser.add_argument("--decoder-init-port", type=int, default=7300)
    parser.add_argument("--decoder-alloc-port", type=int, default=7400)
    parser.add_argument("--notification-host", default="127.0.0.1")
    parser.add_argument("--notification-port", type=int, default=7500)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pd-buffer-size", type=int, default=2 * 1024 * 1024 * 1024)
    parser.add_argument("--chunk-size", type=int, default=256)
    args = parser.parse_args()

    import uvicorn

    config = ProxyConfig(
        prefill_url=args.prefill_url,
        decode_url=args.decode_url,
        decoder_host=args.decoder_host,
        decoder_init_port=args.decoder_init_port,
        decoder_alloc_port=args.decoder_alloc_port,
        notification_host=args.notification_host,
        notification_port=args.notification_port,
        model=args.model,
        pd_buffer_size=args.pd_buffer_size,
        chunk_size=args.chunk_size,
    )
    uvicorn.run(create_proxy(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
