from pathlib import Path
import pytest
import yaml
from document_rag_mcp.config import AppConfig, load_config
@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.delenv("DOCRAG_STORAGE__DATA_DIR", raising=False)



def test_default_config():
    # When no config file and no env, it should load defaults
    config = AppConfig()
    assert len(config.collections) == 0
    assert config.embedding.base_url == "http://localhost:8080/v1"
    assert config.embedding.model == "embed-gemma-300m-FLM"
    assert config.embedding.dimensions == 768
    assert config.vision.enabled is False
    assert config.chunking.local_model == "all-MiniLM-L6-v2"
    assert config.storage.data_dir == Path("./data")
    assert config.server.port == 8000


def test_load_from_yaml(tmp_path):
    yaml_content = {
        "collections": [
            {
                "name": "test-docs",
                "paths": [str(tmp_path / "docs")],
                "file_patterns": ["*.md"]
            }
        ],
        "embedding": {
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-test",
            "model": "text-embedding-3-small",
            "dimensions": 1536
        },
        "storage": {
            "data_dir": str(tmp_path / "custom_data")
        }
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f)
        
    config = load_config(config_file)
    assert len(config.collections) == 1
    assert config.collections[0].name == "test-docs"
    assert config.collections[0].paths == [tmp_path / "docs"]
    assert config.collections[0].file_patterns == ["*.md"]
    assert config.embedding.base_url == "https://api.openai.com/v1"
    assert config.embedding.model == "text-embedding-3-small"
    assert config.embedding.dimensions == 1536
    assert config.storage.data_dir == tmp_path / "custom_data"
    assert config.vision.enabled is False  # default kept


def test_env_override(tmp_path, monkeypatch):
    yaml_content = {
        "collections": [],
        "embedding": {
            "model": "original-model",
            "dimensions": 768
        }
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f)
        
    # Set environment variables to override
    monkeypatch.setenv("DOCRAG_EMBEDDING__MODEL", "overridden-model")
    monkeypatch.setenv("DOCRAG_EMBEDDING__DIMENSIONS", "1024")
    monkeypatch.setenv("DOCRAG_VISION__ENABLED", "true")
    monkeypatch.setenv("DOCRAG_STORAGE__DATA_DIR", str(tmp_path / "env_data"))
    
    config = load_config(config_file)
    assert config.embedding.model == "overridden-model"
    assert config.embedding.dimensions == 1024
    assert config.vision.enabled is True
    assert config.storage.data_dir == tmp_path / "env_data"


def test_load_from_env_var_path(tmp_path, monkeypatch):
    yaml_content = {
        "collections": [],
        "server": {
            "port": 9999
        }
    }
    config_file = tmp_path / "config_env.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f)
        
    monkeypatch.setenv("DOCRAG_CONFIG", str(config_file))
    
    # Passing None to load_config should trigger loading from DOCRAG_CONFIG env var
    config = load_config()
    assert config.server.port == 9999
