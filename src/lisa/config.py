from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LISA_")
    #Primary Model
    model_provider: str = "nvidia"
    model_name: str = "nvidia/nemotron-3.5-lightning-30b-a3b"

    #Fallback Model
    fallback_model_provider: str = "openai"
    fallback_model_name: str = "gpt-5-nano"

    #Router Model
    router_model_provider: str = "openai"
    router_model_name: str = "gpt-5-nano"
    openai_token: str = Field(validation_alias="OPENAI_API_KEY")

    #Technical Clarification Model
    technical_model_provider: str = "openai"
    technical_model_name: str = "gpt-5-nano"

    mcp_server_command: str = "uv"
    mcp_server_args: list[str] = [
        "run",
        "python",
        "-m",
        "lisa_mcp_server.server",
    ]
    mcp_server_cwd: Path = Path("../lisa-mcp-server")
