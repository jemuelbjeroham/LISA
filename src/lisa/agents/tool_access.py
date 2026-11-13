KNOWLEDGE_TOOLS = frozenset({
    "handoff", "web_search", "web_search_with_content", "search_knowledge",
})

TROUBLESHOOTING_TOOLS = KNOWLEDGE_TOOLS | frozenset({
    "search_jira_tickets", "onair_finder",
})

RESTRICTED_AGENT_TOOLS = {
    "knowledge_assistant": KNOWLEDGE_TOOLS,
    "troubleshooting_agent": TROUBLESHOOTING_TOOLS,
}