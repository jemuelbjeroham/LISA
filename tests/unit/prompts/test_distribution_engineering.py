import ast
from pathlib import Path

import pytest

from lisa.prompts.loader import load_prompt

PROMPT_DIR = Path(load_prompt.__code__.co_filename).parent

APPLICATION_PROMPTS = {
    "orchestrator/routing_distribution_engineering_v2.txt",
    "orchestrator/jev_routing_v1.json",
    "Identity.txt",
    "general_enquiry/general_enquiry_v2.txt",
    "knowledge_assistant/knowledge_assistant_v1.txt",
    "troubleshooting_agent/troubleshooting_agent_v1.txt",
    "memory/memory_agent_v2.txt",
}


def test_no_prompt_uses_the_old_delisa_name():
    offenders = [p.relative_to(PROMPT_DIR) for p in PROMPT_DIR.rglob("*.txt") if "DELISA" in p.read_text()]
    assert offenders == []


def test_identity_names_lisa_and_distribution_engineering():
    identity = load_prompt("Identity.txt")
    assert "You are LISA" in identity
    assert "Distribution Engineering" in identity


def test_application_loads_the_three_agent_prompts():
    from lisa import application

    tree = ast.parse(Path(application.__file__).read_text(encoding="utf-8"))
    paths = {node.args[0].value for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
             and node.func.id == "load_prompt" and node.args and isinstance(node.args[0], ast.Constant)}
    assert paths == APPLICATION_PROMPTS
    for path in paths:
        assert load_prompt(path).strip()


@pytest.mark.parametrize("path", ["orchestrator/routing_distribution_engineering_v2.txt", "orchestrator/routing_v2.txt"])
def test_routing_prompt_describes_the_three_routes(path):
    routing = load_prompt(path)
    assert "LISA" in routing
    for route in ("general_enquiry", "knowledge_assistant", "troubleshooting_agent"):
        assert f"`{route}`" in routing
    assert "technical_clarification" not in routing


def test_memory_agent_purpose_is_unchanged():
    memory = load_prompt("memory/memory_agent_v2.txt")
    assert "Only user messages can justify a memory." in memory
    assert "Return only a JSON object matching `MemoryAgentResult`." in memory
