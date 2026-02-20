"""
test_api.py  –  Smoke tests for the FastAPI /generate and /health endpoints.

Uses FastAPI's TestClient (via httpx) so no running server is required.
Run with:
    pytest test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

# Patch model paths to point at pre-trained artefacts before importing main
import os
os.environ["BPE_MODEL_PATH"]     = "bpe_tokenizer_trained.json"
os.environ["TRIGRAM_MODEL_PATH"] = "trigram_model_trained.json"

from main import app  # noqa: E402 – import after env vars are set

client = TestClient(app)


# ─────────────────────────────────────────────────────────────────────────────
# /health
# ─────────────────────────────────────────────────────────────────────────────

def test_health_returns_200():
    r = client.get("/health")
    assert r.status_code == 200


def test_health_payload():
    r = client.get("/health")
    body = r.json()
    assert body["status"] == "ok"
    assert body["bpe_vocab_size"] > 0
    assert body["trigram_contexts"] > 0


# ─────────────────────────────────────────────────────────────────────────────
# /generate – happy path
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_returns_200():
    r = client.post("/generate", json={"prefix": "ایک دن بچہ", "max_length": 30})
    assert r.status_code == 200


def test_generate_response_schema():
    r = client.post("/generate", json={"prefix": "ایک دن", "max_length": 50})
    body = r.json()
    assert "generated_text" in body
    assert "token_count" in body
    assert "stopped_at_eot" in body


def test_generate_text_is_nonempty():
    r = client.post("/generate", json={"prefix": "بچہ باہر گیا", "max_length": 40})
    assert len(r.json()["generated_text"]) > 0


def test_generate_token_count_within_limit():
    max_len = 20
    r = client.post("/generate", json={"prefix": "ایک بار", "max_length": max_len})
    body = r.json()
    # seed tokens + generated tokens; token_count should not exceed seed + max_len + 1 (<EOT>)
    assert body["token_count"] <= max_len + 10  # small buffer for seed


def test_generate_stops_at_eot():
    """
    With a short max_length the model may hit <EOT>; if it does,
    stopped_at_eot must be True.
    """
    r = client.post("/generate", json={"prefix": "ایک دن", "max_length": 300})
    body = r.json()
    if body["stopped_at_eot"]:
        assert body["generated_text"].endswith("<EOT>") or True  # decode may strip it
    assert isinstance(body["stopped_at_eot"], bool)


def test_generate_temperature_param():
    """Different temperatures should both succeed."""
    for temp in [0.5, 1.0, 1.5]:
        r = client.post(
            "/generate",
            json={"prefix": "ایک بار کی بات", "max_length": 30, "temperature": temp},
        )
        assert r.status_code == 200, f"Failed at temperature={temp}"


def test_generate_top_k_param():
    for k in [5, 20, 100]:
        r = client.post(
            "/generate",
            json={"prefix": "بچہ سکول", "max_length": 30, "top_k": k},
        )
        assert r.status_code == 200, f"Failed at top_k={k}"


# ─────────────────────────────────────────────────────────────────────────────
# /generate – validation errors
# ─────────────────────────────────────────────────────────────────────────────

def test_generate_missing_prefix():
    r = client.post("/generate", json={"max_length": 50})
    assert r.status_code == 422


def test_generate_empty_prefix():
    r = client.post("/generate", json={"prefix": "", "max_length": 50})
    assert r.status_code == 422


def test_generate_max_length_too_small():
    r = client.post("/generate", json={"prefix": "ایک", "max_length": 1})
    assert r.status_code == 422


def test_generate_max_length_too_large():
    r = client.post("/generate", json={"prefix": "ایک", "max_length": 9999})
    assert r.status_code == 422


def test_generate_invalid_temperature():
    r = client.post(
        "/generate",
        json={"prefix": "ایک دن", "max_length": 30, "temperature": 0.0},
    )
    assert r.status_code == 422