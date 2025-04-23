from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    embeddings_openai_deployment: str
    embeddings_openai_endpoint: str
    embeddings_openai_key: str
    openai_api_version: str
    model_config = SettingsConfigDict(env_file=".env")


settings: Settings = Settings()