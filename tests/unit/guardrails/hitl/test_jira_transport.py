import secrets
from uuid import uuid4

import httpx
import pytest

from lisa.guardrails.hitl.jira import JiraAdapter, capability, verify_capability
from lisa.guardrails.hitl.policy import GuardError


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setenv("LISA_JIRA_SIGNING_KEY", secrets.token_hex(32))
    monkeypatch.setenv("NET_TOOLS_SERVICE_TOKEN", secrets.token_hex(32))
    monkeypatch.setenv("NET_TOOLS_API_URL", "http://tools")
    return monkeypatch


def test_capability_is_signed_and_tampering_is_denied(configured):
    claim = uuid4()
    token = capability(claim, "approved-hash")
    assert verify_capability(token)["claim_id"] == str(claim)
    for invalid in ["unsigned", token + "changed", ".".join(token.split(".")[::-1])]:
        with pytest.raises(GuardError, match="invalid_execution_capability"):
            verify_capability(invalid)


def test_expired_capability_is_denied(configured):
    token = capability(uuid4(), "approved-hash")
    configured.setattr("lisa.guardrails.hitl.jira.time.time", lambda: float("inf"))
    with pytest.raises(GuardError, match="invalid_execution_capability"):
        verify_capability(token)


async def test_transport_sanitizes_remote_errors(configured):
    sensitive = secrets.token_urlsafe(32)
    original = httpx.AsyncClient
    configured.setattr(httpx, "AsyncClient", lambda **keywords: original(**keywords,
        transport=httpx.MockTransport(lambda request: httpx.Response(502, text=sensitive))))
    with pytest.raises(GuardError) as raised:
        await JiraAdapter().prepare("close_ticket", {"ticket_id": "123"})
    assert raised.value.code == "jira_preparation_failed"
    assert sensitive not in str(raised.value)


@pytest.mark.parametrize("url", ["http://user:password@tools", "http://tools?token=hidden", "file:///tools", "http://tools:bad"])
async def test_invalid_transport_configuration_fails_closed(configured, url):
    configured.setenv("NET_TOOLS_API_URL", url)
    with pytest.raises(GuardError, match="jira_transport_unavailable"):
        await JiraAdapter().prepare("close_ticket", {"ticket_id": "123"})