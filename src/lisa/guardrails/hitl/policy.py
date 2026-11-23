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


# LG TV tools served by lg-mcp-server. They act on a device on the local
# network rather than on business records, so they run directly for admins
# instead of going through the approval workflow.
TV_READ_TOOLS = frozenset({
    "tv_health_check", "tv_get_status", "tv_get_info", "tv_get_picture_settings",
    "tv_list_apps", "tv_list_inputs",
})
TV_CONTROL_TOOLS = frozenset({
    "tv_power_off", "tv_power_on", "tv_set_screen", "tv_set_volume", "tv_adjust_volume",
    "tv_set_mute", "tv_set_sound_output", "tv_media_control", "tv_launch_app",
    "tv_open_url", "tv_switch_input", "tv_show_notification", "tv_show_alert",
    "tv_close_alert",
    "tv_youtube_seek",
})


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
        ToolPolicy("search_jira_tickets", "mcp", "jira.search"),
        ToolPolicy("create_jira_ticket", "mcp", "jira.create", write=True,
            editable=("subject", "description", "project_key", "issue_type", "priority")),
        ToolPolicy("update_jira_ticket", "mcp", "jira.comment", write=True, editable=("comment",)),
        ToolPolicy("close_ticket", "mcp", "jira.close", write=True),
        ToolPolicy("save_memory", "native", "memory.create", member=True),
        *(ToolPolicy(name, "mcp", f"tv.read.{name.removeprefix('tv_')}")
          for name in sorted(TV_READ_TOOLS)),
        *(ToolPolicy(name, "mcp", f"tv.control.{name.removeprefix('tv_')}")
          for name in sorted(TV_CONTROL_TOOLS)),
    )
}


def authorize(role: Literal["admin", "member"], name: str) -> ToolPolicy:
    policy = REGISTRY.get(name)
    if policy is None or role not in ("admin", "member") or (role == "member" and not policy.member):
        raise GuardError()
    return policy