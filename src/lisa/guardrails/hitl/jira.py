import base64
import hashlib
import hmac
import json
import os
import time
from urllib.parse import urlsplit

import httpx

from lisa.guardrails.hitl.policy import GuardError

JIRA_TOOLS = frozenset({"create_jira_ticket", "update_jira_ticket", "close_ticket"})


def signing_key():
    key = os.getenv("LISA_JIRA_SIGNING_KEY", "")
    if len(key) < 32:
        raise GuardError("jira_signing_unavailable", 503)
    return key.encode()


def capability(claim_id, payload_hash):
    body = base64.urlsafe_b64encode(json.dumps({"claim_id": str(claim_id), "hash": payload_hash,
        "expires": int(time.time()) + 90}, sort_keys=True, separators=(",", ":")).encode()).decode()
    signature = hmac.new(signing_key(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{signature}"


def verify_capability(value):
    key = signing_key()
    try:
        if not isinstance(value, str) or len(value) > 2048:
            raise ValueError
        body, signature = value.split(".")
        expected = hmac.new(key, body.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError
        claims = json.loads(base64.urlsafe_b64decode(body))
        if type(claims["expires"]) is not int or not time.time() < claims["expires"] <= time.time() + 90:
            raise ValueError
        return claims
    except (ValueError, KeyError, TypeError):
        raise GuardError("invalid_execution_capability", 403) from None


class JiraAdapter:
    async def request(self, phase, name, arguments, execution_capability=None):
        url = os.getenv("NET_TOOLS_API_URL", "").rstrip("/")
        token = os.getenv("NET_TOOLS_SERVICE_TOKEN", "")
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and parsed.hostname and not (
                parsed.username or parsed.password or parsed.query or parsed.fragment)
            parsed.port
        except ValueError:
            valid = False
        if not valid or not token.strip():
            raise GuardError("jira_transport_unavailable", 503)
        headers = {"Authorization": f"Bearer {token}"}
        if execution_capability:
            headers["X-LISA-Execution"] = execution_capability
        try:
            async with httpx.AsyncClient(timeout=60, trust_env=False, follow_redirects=False) as client:
                response = await client.post(f"{url}/internal/jira/{phase}", headers=headers,
                    json={"tool_name": name, "arguments": arguments})
            if response.status_code != 200:
                raise GuardError("jira_preparation_failed" if phase == "prepare" else "jira_execution_unconfirmed",
                    422 if phase == "prepare" and response.status_code in {400, 404, 409, 422} else 503)
            result = response.json()
            if not isinstance(result, dict):
                raise ValueError
            return result
        except (httpx.HTTPError, ValueError):
            raise GuardError("jira_transport_unavailable", 503) from None

    async def prepare(self, name, arguments):
        signing_key()
        result = await self.request("prepare", name, arguments)
        if (result.get("tool_name") != name or not isinstance(result.get("arguments"), dict)
                or not isinstance(result.get("payload"), dict) or not isinstance(result.get("destination"), str)):
            raise GuardError("invalid_jira_preparation", 503)
        return result

    async def execute(self, name, arguments, claim_id, payload_hash):
        result = await self.request("execute", name, arguments, capability(claim_id, payload_hash))
        if result.get("success") is not True:
            raise GuardError("jira_execution_unconfirmed", 503)
        return result