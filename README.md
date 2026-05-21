# vela

A simple router for llm service.

## Features

- Provider management (`name`, `url`, `api_key`, `protocol`) for OpenAI and Anthropic endpoints.
- Model management per provider, with backup/fallback model configuration.
- Unified gateway endpoint (`/v1/chat/completions`, also `/api/gateway/chat/completions`) for OpenAI-style access.
- Automatic failover when upstream model/provider is unreachable, times out, or hits TPM/rate limits.
- Built-in frontend for provider/model/fallback configuration and gateway testing.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8000

## Test

```bash
python -m pytest tests/test_app.py
```
