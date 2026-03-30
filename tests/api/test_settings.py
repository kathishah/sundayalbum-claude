"""Tests 22–23: Settings API — user API key storage."""

from __future__ import annotations

import hashlib
import json

import pytest

from helpers import make_event, make_session

import settings as settings_module


def _hash(email: str) -> str:
    return hashlib.sha256(email.encode()).hexdigest()


def _auth(email: str = "user@example.com") -> tuple[str, str, str]:
    user_hash = _hash(email)
    token = make_session(email, user_hash)
    return email, user_hash, token


# ── 22. store and retrieve API keys ──────────────────────────────────────────


def test_store_and_retrieve_api_keys():
    """PUT then GET returns presence flags only — never the raw key values."""
    _, _, token = _auth()

    # Store keys
    put_event = make_event(
        "PUT /settings/api-keys",
        body={"anthropic_api_key": "sk-ant-test", "openai_api_key": "sk-openai-test"},
        token=token,
    )
    put_resp = settings_module.handler(put_event, None)
    assert put_resp["statusCode"] == 200

    # Retrieve — should see flags, not values
    get_event = make_event("GET /settings/api-keys", token=token)
    get_resp = settings_module.handler(get_event, None)
    assert get_resp["statusCode"] == 200

    data = json.loads(get_resp["body"])
    assert data["has_anthropic_key"] is True
    assert data["has_openai_key"] is True
    # Raw key values must never appear in the response
    assert "sk-ant-test" not in get_resp["body"]
    assert "sk-openai-test" not in get_resp["body"]


# ── 23. delete API keys ───────────────────────────────────────────────────────


def test_delete_api_keys():
    """DELETE clears stored keys; subsequent GET shows both flags false."""
    _, _, token = _auth()

    # Store then delete
    put_event = make_event(
        "PUT /settings/api-keys",
        body={"anthropic_api_key": "sk-ant-test", "openai_api_key": "sk-openai-test"},
        token=token,
    )
    settings_module.handler(put_event, None)

    del_event = make_event("DELETE /settings/api-keys", token=token)
    del_resp = settings_module.handler(del_event, None)
    assert del_resp["statusCode"] == 200

    get_event = make_event("GET /settings/api-keys", token=token)
    get_resp = settings_module.handler(get_event, None)
    data = json.loads(get_resp["body"])
    assert data["has_anthropic_key"] is False
    assert data["has_openai_key"] is False
