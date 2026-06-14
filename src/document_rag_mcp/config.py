import os
from pathlib import Path
from typing import Any
import yaml
from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    SettingsConfigDict,
    PydanticBaseSettingsSource,
)

class CollectionConfig(BaseModel):
    name: str                           # e.g., "project-docs"
    paths: list[Path]                   # folders to watch/scan
    file_patterns: list[str] = ["*.txt", "*.md", "*.pdf"]


class EmbeddingConfig(BaseModel):
    base_url: str = "http://localhost:8080/v1"   # lemonade default
    api_key: str = "unused"
    model: str = "embed-gemma-300m-FLM"
    dimensions: int = 768               # gemma embedding dims
    batch_size: int = 32


class VisionConfig(BaseModel):
    enabled: bool = False
    base_url: str = "http://localhost:8080/v1"
    api_key: str = "unused"
    model: str = "gpt-4o"


class ChunkingConfig(BaseModel):
    max_chunk_size: int = 512          # tokens
    similarity_threshold: float = 0.5  # for semantic boundary detection
    local_model: str = "all-MiniLM-L6-v2"  # overridable via --chunking-model CLI flag


class StorageConfig(BaseModel):
    data_dir: Path = Path("./data")    # ChromaDB + SQLite storage


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="DOCRAG_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    collections: list[CollectionConfig] = Field(default_factory=list)
    embedding: EmbeddingConfig = EmbeddingConfig()
    vision: VisionConfig = VisionConfig()
    chunking: ChunkingConfig = ChunkingConfig()
    storage: StorageConfig = StorageConfig()
    server: ServerConfig = ServerConfig()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Prioritize environment variables over initialization arguments (YAML data)
        return env_settings, init_settings, dotenv_settings, file_secret_settings


def load_config(config_path: Path | str | None = None) -> AppConfig:
    """Loads application configuration from YAML file and overrides with environment variables.

    Environment variables must be prefixed with DOCRAG_ and nested with double underscores.
    For example: DOCRAG_EMBEDDING__MODEL="text-embedding-3-small"
    """
    if not config_path:
        config_path = os.environ.get("DOCRAG_CONFIG")

    yaml_data: dict[str, Any] = {}
    if config_path:
        path = Path(config_path)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    yaml_data = loaded

    # Pydantic BaseSettings automatically prioritizes env vars over values passed in init.
    return AppConfig(**yaml_data)
