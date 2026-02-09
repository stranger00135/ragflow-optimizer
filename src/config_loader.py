"""Configuration loader with environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from dotenv import load_dotenv


def load_env_file(env_path: Optional[str] = None) -> None:
    """Load environment variables from .env file.

    Args:
        env_path: Optional path to .env file. If not provided,
                  looks for .env in project root.
    """
    if env_path:
        load_dotenv(env_path)
    else:
        # Auto-discover .env in project root
        project_root = Path(__file__).parent.parent
        default_env = project_root / ".env"
        if default_env.exists():
            load_dotenv(default_env)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file with env var substitution."""
    with open(config_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Substitute environment variables
    def replace_env_var(match):
        var_name = match.group(1)
        return os.environ.get(var_name, "")

    content = re.sub(r"\$\{(\w+)\}", replace_env_var, content)

    return yaml.safe_load(content)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def resolve_path(config: Dict, key: str) -> Path:
    """Resolve a path from config relative to project root."""
    project_root = get_project_root()
    path_str = config.get("paths", {}).get(key, "")
    return project_root / path_str


class Config:
    """Configuration wrapper for easy access."""

    def __init__(self, config_path: str, env_path: Optional[str] = None):
        # Load environment variables from .env file first
        load_env_file(env_path)

        self.data = load_config(config_path)
        self.project_root = get_project_root()

    # Backend configuration properties
    @property
    def backend_type(self) -> str:
        """Get the configured backend type."""
        backend_section = self.data.get("backend", {})
        if isinstance(backend_section, dict):
            return backend_section.get("type", "ragflow")
        return str(backend_section) if backend_section else "ragflow"

    @property
    def backend_config(self) -> Dict[str, Any]:
        """Get the configuration for the current backend type."""
        backend_type = self.backend_type
        backend_section = self.data.get("backend", {})
        if not isinstance(backend_section, dict):
            backend_section = {}

        # Get backend-specific config from nested section
        config = backend_section.get(backend_type, {})

        # Fall back to legacy ragflow section for backward compatibility
        if backend_type == "ragflow" and not config:
            config = {
                "base_url": self.ragflow_base_url,
                "api_key": self.ragflow_api_key,
            }

        # Add max_retries from retry section
        config["max_retries"] = self.max_retries

        return config

    # Legacy RAGFlow properties (kept for backward compatibility)
    @property
    def ragflow_base_url(self) -> str:
        # First check backend.ragflow, then fall back to legacy ragflow section
        backend_section = self.data.get("backend", {})
        if not isinstance(backend_section, dict):
            backend_section = {}
        backend_ragflow = backend_section.get("ragflow", {})
        if backend_ragflow.get("base_url"):
            return backend_ragflow.get("base_url")
        return self.data.get("ragflow", {}).get("base_url", "http://localhost/")

    @property
    def ragflow_api_key(self) -> str:
        # First check backend.ragflow, then fall back to legacy ragflow section
        backend_section = self.data.get("backend", {})
        if not isinstance(backend_section, dict):
            backend_section = {}
        backend_ragflow = backend_section.get("ragflow", {})
        if backend_ragflow.get("api_key"):
            return backend_ragflow.get("api_key")
        return self.data.get("ragflow", {}).get("api_key", "")

    @property
    def llm_provider(self) -> str:
        return self.data.get("llm", {}).get("provider", "openai")

    @property
    def llm_model(self) -> str:
        return self.data.get("llm", {}).get("model", "gpt-4o-mini")

    @property
    def llm_api_key(self) -> str:
        return self.data.get("llm", {}).get("api_key", "")

    @property
    def source_docs_path(self) -> Path:
        return self.project_root / self.data.get("paths", {}).get("source_docs", "Source_docs/")

    @property
    def distractors_path(self) -> Path:
        return self.project_root / self.data.get("paths", {}).get("distractors", "Distractors/")

    @property
    def output_path(self) -> Path:
        return self.project_root / self.data.get("paths", {}).get("output", "output/")

    @property
    def questions_cache_path(self) -> Path:
        return self.project_root / self.data.get("paths", {}).get("questions_cache", "questions_cache/")

    @property
    def questions_per_folder(self) -> int:
        return self.data.get("evaluation", {}).get("questions_per_folder", 5)

    @property
    def top_k(self) -> int:
        return self.data.get("evaluation", {}).get("top_k", 3)

    @property
    def similarity_threshold(self) -> float:
        return self.data.get("evaluation", {}).get("similarity_threshold", 0.2)

    @property
    def distractor_kb_name(self) -> str:
        return self.data.get("distractor_kb", {}).get("name", "eval_distractor_kb")

    @property
    def distractor_ingest_preset(self) -> str:
        return self.data.get("distractor_kb", {}).get("ingest_preset", "general")

    @property
    def backend_name(self) -> str:
        """Alias for backend_type for orchestrator compatibility."""
        return self.backend_type

    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get preset configuration by name."""
        return self.data.get("presets", {}).get(preset_name, {})

    def get_optimization_space(self, preset_name: str) -> list:
        """Get optimization space for a preset."""
        return self.data.get("optimization_spaces", {}).get(preset_name, [])

    @property
    def max_retries(self) -> int:
        return self.data.get("retry", {}).get("max_retries", 3)
