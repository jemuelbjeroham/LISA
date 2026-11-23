from lisa.guardrails.hitl.policy import TV_CONTROL_TOOLS, TV_READ_TOOLS

KNOWLEDGE_TOOLS = frozenset({
    "handoff", "web_search", "web_search_with_content", "search_knowledge",
}) | TV_READ_TOOLS

TROUBLESHOOTING_TOOLS = KNOWLEDGE_TOOLS | TV_CONTROL_TOOLS | frozenset({
    "search_jira_tickets",
})

# The TV controller works only on the LG TV: the TV tools, handoff, and
# web_search to find a YouTube link when the user names a song or video.
TV_CONTROLLER_TOOLS = frozenset({"handoff", "web_search"}) | TV_READ_TOOLS | TV_CONTROL_TOOLS

RESTRICTED_AGENT_TOOLS = {
    "knowledge_assistant": KNOWLEDGE_TOOLS,
    "troubleshooting_agent": TROUBLESHOOTING_TOOLS,
    "tv_controller": TV_CONTROLLER_TOOLS,
}