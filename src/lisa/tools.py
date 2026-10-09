from langchain_core.tools import tool


@tool
def handoff(reason: str) -> str:
    """Handoff the current request back to the orchestrator"""
    return reason