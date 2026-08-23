from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LISA_")
    model_provider: str = "nvidia"
    model_name: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
