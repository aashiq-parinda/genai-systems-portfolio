"""Tests: Authentication — valid/invalid/revoked/missing key scenarios."""

import pytest


@pytest.mark.asyncio
async def test_valid_key_allows_access(client):
    ac, raw_key = client
    resp = await ac.get("/v1/models", headers={"Authorization": f"Bearer {raw_key}"})
    assert resp.status_code == 200
    assert "data" in resp.json()


@pytest.mark.asyncio
async def test_missing_auth_header_returns_401(client):
    ac, _ = client
    resp = await ac.get("/v1/models")
    assert resp.status_code == 401
    assert "Authorization" in resp.json()["detail"] or "missing" in resp.json()["error"]


@pytest.mark.asyncio
async def test_invalid_key_returns_401(client):
    ac, _ = client
    resp = await ac.get("/v1/models", headers={"Authorization": "Bearer llm_totallyfakekey12345"})
    assert resp.status_code == 401
    assert resp.json()["error"] == "invalid_api_key"


@pytest.mark.asyncio
async def test_malformed_auth_header_returns_401(client):
    ac, _ = client
    # Missing "Bearer " prefix
    resp = await ac.get("/v1/models", headers={"Authorization": "Token somethingelse"})
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_health_endpoint_requires_no_auth(client):
    """Health endpoints must be reachable without a key — for load balancer probes."""
    ac, _ = client
    resp = await ac.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_request_id_injected_in_response(client):
    ac, raw_key = client
    resp = await ac.get("/v1/models", headers={"Authorization": f"Bearer {raw_key}"})
    assert "x-request-id" in resp.headers


@pytest.mark.asyncio
async def test_create_and_list_keys(client, db_session, test_user):
    ac, raw_key = client
    headers = {"Authorization": f"Bearer {raw_key}"}

    # Create a new key
    create_resp = await ac.post(
        "/v1/auth/keys",
        json={"name": "my-new-key", "scopes": ["chat"]},
        headers=headers,
    )
    assert create_resp.status_code == 201
    data = create_resp.json()
    assert data["raw_key"].startswith("llm_")
    assert "raw_key" in data  # shown once

    # List keys — raw_key must NOT appear in list
    list_resp = await ac.get("/v1/auth/keys", headers=headers)
    assert list_resp.status_code == 200
    keys = list_resp.json()
    assert all("raw_key" not in k for k in keys)


@pytest.mark.asyncio
async def test_revoke_key(client, db_session, test_user):
    ac, raw_key = client
    headers = {"Authorization": f"Bearer {raw_key}"}

    # Create a key to revoke
    create_resp = await ac.post(
        "/v1/auth/keys",
        json={"name": "to-revoke"},
        headers=headers,
    )
    key_id = create_resp.json()["id"]

    # Revoke it
    revoke_resp = await ac.delete(f"/v1/auth/keys/{key_id}", headers=headers)
    assert revoke_resp.status_code == 204

    # List and confirm it's inactive
    list_resp = await ac.get("/v1/auth/keys", headers=headers)
    revoked = next(k for k in list_resp.json() if k["id"] == key_id)
    assert revoked["is_active"] is False
