from lisa.agents.tool_access import (
    KNOWLEDGE_TOOLS,
    RESTRICTED_AGENT_TOOLS,
    TROUBLESHOOTING_TOOLS,
)
from lisa.guardrails.hitl.policy import TV_CONTROL_TOOLS, TV_READ_TOOLS


def test_knowledge_assistant_can_read_but_not_control_the_tv():
    assert TV_READ_TOOLS <= KNOWLEDGE_TOOLS
    assert not TV_CONTROL_TOOLS & KNOWLEDGE_TOOLS


def test_troubleshooting_agent_can_read_and_control_the_tv():
    assert TV_READ_TOOLS | TV_CONTROL_TOOLS <= TROUBLESHOOTING_TOOLS


def test_restricted_agents_never_get_jira_writes():
    for allowed in RESTRICTED_AGENT_TOOLS.values():
        assert not {"create_jira_ticket", "update_jira_ticket", "close_ticket"} & allowed
