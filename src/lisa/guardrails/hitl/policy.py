from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lisa.memory.models import MemoryScope, MemoryType

POLICY_VERSION = "hitl-v1"


class GuardError(Exception):
    def __init__(self, code="not_authorized", status=403):
        self.code = code
        self.status = status
        super().__init__(code)

    def result(self):
        return {"accepted": False, "error": self.code}


class MemoryArguments(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    content: str = Field(min_length=1, max_length=32000)
    memory_type: MemoryType
    scope: MemoryScope


@dataclass(frozen=True)
class ToolPolicy:
    name: str
    namespace: str
    operation: str
    write: bool = False
    member: bool = False
    editable: tuple[str, ...] = ()
    schema_version: str = "1"


REGISTRY = {
    policy.name: policy for policy in (
        ToolPolicy("web_search", "mcp", "web.search", member=True),
        ToolPolicy("web_search_with_content", "mcp", "web.search_content", member=True),
        ToolPolicy("search_knowledge", "mcp", "knowledge.search"),
        ToolPolicy("health_check", "mcp", "health.read"),
        ToolPolicy("onair_finder", "mcp", "inventory.read"),
        ToolPolicy("search_jira_tickets", "mcp", "jira.search"),
        ToolPolicy("create_jira_ticket", "mcp", "jira.create", write=True,
            editable=("subject", "description", "core", "side", "onair_core")),
        ToolPolicy("update_jira_ticket", "mcp", "jira.comment", write=True, editable=("comment",)),
        ToolPolicy("close_ticket", "mcp", "jira.close", write=True),
        ToolPolicy("save_memory", "native", "memory.create", member=True),
    )
}


def authorize(role: Literal["admin", "member"], name: str) -> ToolPolicy:
    policy = REGISTRY.get(name)
    if policy is None or role not in ("admin", "member") or (role == "member" and not policy.member):
        raise GuardError()
    return policy