from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

from app import app


client = TestClient(app)


def test_provider_url_validation():
    response = client.post(
        "/api/providers",
        json={
            "name": "bad-openai",
            "url": "https://example.com/v1",
            "api_key": "k",
            "protocol": "openai",
            "timeout_seconds": 30,
        },
    )
    assert response.status_code == 400
    assert "openai" in response.text


def test_gateway_fallback_on_connect_error():
    def fake_perform(model, payload):
        if model.id == "model_gpt_4o":
            raise httpx.ConnectError("unreachable", request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"))
        return {
            "id": "ok",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
        }

    with patch("app._perform_model_call", side_effect=fake_perform):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "model_gpt_4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_model_id"] == "model_claude_sonnet"
    assert len(payload["attempts"]) == 2
