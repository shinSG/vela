from __future__ import annotations

import uuid
from pathlib import Path
import re
from typing import Any, Literal

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


class ProviderInput(BaseModel):
    name: str
    url: str
    api_key: str
    protocol: Literal["openai", "anthropic"]
    timeout_seconds: float = Field(default=30, ge=1, le=120)


class Provider(ProviderInput):
    id: str


class ModelInput(BaseModel):
    name: str
    provider_id: str
    upstream_model: str
    fallback_model_ids: list[str] = Field(default_factory=list)
    enabled: bool = True


class ModelConfig(ModelInput):
    id: str


class GatewayRequest(BaseModel):
    model: str
    messages: list[dict[str, Any]]
    temperature: float | None = None
    max_tokens: int | None = None


class GatewayAttempt(BaseModel):
    model_id: str
    provider_id: str
    success: bool
    reason: str | None = None


class GatewayResult(BaseModel):
    selected_model_id: str
    attempts: list[GatewayAttempt]
    response: dict[str, Any]


app = FastAPI(title="Vela LLM Router")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

providers: dict[str, Provider] = {}
models: dict[str, ModelConfig] = {}
MAX_ERROR_DETAIL_LENGTH = 500


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _validate_provider_url(data: ProviderInput) -> None:
    lowered = data.url.lower()
    if data.protocol == "openai" and "openai" not in lowered:
        raise HTTPException(status_code=400, detail="openai provider URL must contain 'openai'.")
    if data.protocol == "anthropic" and "anthropic" not in lowered:
        raise HTTPException(status_code=400, detail="anthropic provider URL must contain 'anthropic'.")


def _seed_data() -> None:
    openai_provider = Provider(
        id="provider_openai",
        name="OpenAI",
        url="https://api.openai.com/v1",
        api_key="replace-with-key",
        protocol="openai",
        timeout_seconds=30,
    )
    anthropic_provider = Provider(
        id="provider_anthropic",
        name="Anthropic",
        url="https://api.anthropic.com/v1",
        api_key="replace-with-key",
        protocol="anthropic",
        timeout_seconds=30,
    )
    providers[openai_provider.id] = openai_provider
    providers[anthropic_provider.id] = anthropic_provider

    gpt4o = ModelConfig(
        id="model_gpt_4o",
        name="GPT-4o",
        provider_id=openai_provider.id,
        upstream_model="gpt-4o",
        fallback_model_ids=["model_claude_sonnet"],
        enabled=True,
    )
    claude = ModelConfig(
        id="model_claude_sonnet",
        name="Claude Sonnet 4",
        provider_id=anthropic_provider.id,
        upstream_model="claude-sonnet-4-20250514",
        fallback_model_ids=["model_gpt_4o"],
        enabled=True,
    )
    models[gpt4o.id] = gpt4o
    models[claude.id] = claude


_seed_data()


@app.get("/api/providers", response_model=list[Provider])
def list_providers() -> list[Provider]:
    return list(providers.values())


@app.post("/api/providers", response_model=Provider)
def create_provider(payload: ProviderInput) -> Provider:
    _validate_provider_url(payload)
    provider = Provider(id=_new_id("provider"), **payload.model_dump())
    providers[provider.id] = provider
    return provider


@app.put("/api/providers/{provider_id}", response_model=Provider)
def update_provider(provider_id: str, payload: ProviderInput) -> Provider:
    if provider_id not in providers:
        raise HTTPException(status_code=404, detail="Provider not found")
    _validate_provider_url(payload)
    provider = Provider(id=provider_id, **payload.model_dump())
    providers[provider_id] = provider
    return provider


@app.delete("/api/providers/{provider_id}")
def delete_provider(provider_id: str) -> dict[str, bool]:
    if provider_id not in providers:
        raise HTTPException(status_code=404, detail="Provider not found")
    for model in models.values():
        if model.provider_id == provider_id:
            raise HTTPException(status_code=400, detail="Provider still used by models")
    del providers[provider_id]
    return {"ok": True}


@app.get("/api/models", response_model=list[ModelConfig])
def list_models() -> list[ModelConfig]:
    return list(models.values())


@app.post("/api/models", response_model=ModelConfig)
def create_model(payload: ModelInput) -> ModelConfig:
    if payload.provider_id not in providers:
        raise HTTPException(status_code=400, detail="provider_id does not exist")
    model = ModelConfig(id=_new_id("model"), **payload.model_dump())
    models[model.id] = model
    return model


@app.put("/api/models/{model_id}", response_model=ModelConfig)
def update_model(model_id: str, payload: ModelInput) -> ModelConfig:
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    if payload.provider_id not in providers:
        raise HTTPException(status_code=400, detail="provider_id does not exist")
    model = ModelConfig(id=model_id, **payload.model_dump())
    models[model_id] = model
    return model


@app.delete("/api/models/{model_id}")
def delete_model(model_id: str) -> dict[str, bool]:
    if model_id not in models:
        raise HTTPException(status_code=404, detail="Model not found")
    del models[model_id]
    for cfg in models.values():
        cfg.fallback_model_ids = [item for item in cfg.fallback_model_ids if item != model_id]
    return {"ok": True}


def _is_switchable_failure(status_code: int | None, response_text: str | None, exc: Exception | None) -> bool:
    if exc is not None:
        return True
    if status_code in {408, 429, 500, 502, 503, 504}:
        return True
    if response_text and re.search(r"\b(tpm|tokens per minute|rate limit)\b", response_text.lower()):
        return True
    return False


def _openai_call(provider: Provider, model: ModelConfig, payload: GatewayRequest) -> dict[str, Any]:
    with httpx.Client(timeout=provider.timeout_seconds) as client:
        response = client.post(
            f"{provider.url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {provider.api_key}"},
            json={
                "model": model.upstream_model,
                "messages": payload.messages,
                "temperature": payload.temperature,
                "max_tokens": payload.max_tokens,
            },
        )
    response.raise_for_status()
    return response.json()


def _anthropic_call(provider: Provider, model: ModelConfig, payload: GatewayRequest) -> dict[str, Any]:
    system_messages = [m.get("content", "") for m in payload.messages if m.get("role") == "system"]
    user_messages = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in payload.messages
        if m.get("role") != "system"
    ]
    req_body: dict[str, Any] = {
        "model": model.upstream_model,
        "messages": user_messages,
        "max_tokens": payload.max_tokens or 1024,
    }
    if system_messages:
        req_body["system"] = "\n".join(system_messages)
    if payload.temperature is not None:
        req_body["temperature"] = payload.temperature

    with httpx.Client(timeout=provider.timeout_seconds) as client:
        response = client.post(
            f"{provider.url.rstrip('/')}/messages",
            headers={
                "x-api-key": provider.api_key,
                "anthropic-version": "2023-06-01",
            },
            json=req_body,
        )
    response.raise_for_status()
    raw = response.json()
    text = ""
    for block in raw.get("content", []):
        if block.get("type") == "text":
            text += block.get("text", "")
    return {
        "id": raw.get("id", "anthropic-response"),
        "object": "chat.completion",
        "model": model.id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": raw.get("stop_reason", "stop"),
            }
        ],
    }


def _perform_model_call(model: ModelConfig, payload: GatewayRequest) -> dict[str, Any]:
    provider = providers.get(model.provider_id)
    if not provider:
        raise HTTPException(status_code=400, detail=f"Provider {model.provider_id} for model {model.id} is missing")
    if provider.protocol == "openai":
        return _openai_call(provider, model, payload)
    return _anthropic_call(provider, model, payload)


def _fallback_chain(first_model_id: str) -> list[str]:
    if first_model_id not in models:
        raise HTTPException(status_code=404, detail="Requested model not found")
    order: list[str] = []
    seen: set[str] = set()
    pending = [first_model_id]
    while pending:
        mid = pending.pop(0)
        if mid in seen or mid not in models:
            continue
        seen.add(mid)
        order.append(mid)
        for fallback_id in models[mid].fallback_model_ids:
            if fallback_id not in seen:
                pending.append(fallback_id)
    return order


@app.post("/api/gateway/chat/completions", response_model=GatewayResult)
@app.post("/v1/chat/completions", response_model=GatewayResult)
def gateway_chat_completions(payload: GatewayRequest) -> GatewayResult:
    attempts: list[GatewayAttempt] = []
    last_error: str | None = None
    for model_id in _fallback_chain(payload.model):
        model = models[model_id]
        if not model.enabled:
            attempts.append(
                GatewayAttempt(
                    model_id=model_id,
                    provider_id=model.provider_id,
                    success=False,
                    reason="model disabled",
                )
            )
            continue

        try:
            response = _perform_model_call(model, payload)
            attempts.append(
                GatewayAttempt(
                    model_id=model_id,
                    provider_id=model.provider_id,
                    success=True,
                )
            )
            return GatewayResult(selected_model_id=model_id, attempts=attempts, response=response)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text
            trimmed_detail = detail[:MAX_ERROR_DETAIL_LENGTH]
            attempts.append(
                GatewayAttempt(
                    model_id=model_id,
                    provider_id=model.provider_id,
                    success=False,
                    reason=f"HTTP {exc.response.status_code}: {trimmed_detail}",
                )
            )
            if _is_switchable_failure(exc.response.status_code, detail, None):
                last_error = attempts[-1].reason
                continue
            raise HTTPException(status_code=exc.response.status_code, detail=trimmed_detail) from exc
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            attempts.append(
                GatewayAttempt(
                    model_id=model_id,
                    provider_id=model.provider_id,
                    success=False,
                    reason=f"network error: {type(exc).__name__}",
                )
            )
            last_error = attempts[-1].reason
            continue

    raise HTTPException(status_code=503, detail=f"All candidate models failed. Last error: {last_error}")


@app.get("/")
def serve_frontend() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
