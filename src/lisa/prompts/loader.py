from pathlib import Path

PROMPTS_DIR = Path(__file__).parent

def load_prompt(prompt_path: str) -> str:
    path = PROMPTS_DIR/prompt_path
    return path.read_text(encoding="utf-8")