from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LISA_")
    model_provider: str = "nvidia"
    model_name: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    router_model_provider: str = "huggingface"
    router_model_name: str = "Qwen/Qwen3-4B-Instruct-2507"
    hf_token: str = Field(validation_alias="HF_TOKEN")

    mcp_server_command: str = "uv"
    mcp_server_args: list[str] = [
        "run",
        "python",
        "-m",
        "lisa_mcp_server.server",
    ]
    mcp_server_cwd: Path = Path("../lisa-mcp-server")
