from lisa.prompts.loader import load_prompt


def test_load_prompt():
    prompt = load_prompt("orchestrator/routing_v1.txt")

    assert prompt