"""Configuration loader with environment variable substitution."""

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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


def load_credentials(credentials_path: str) -> Dict[str, str]:
    """Load credentials from a file and set as environment variables."""
    credentials = {}

    with open(credentials_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                credentials[key] = value
                os.environ[key] = value

    return credentials


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

    def __init__(self, config_path: str, credentials_path: Optional[str] = None):
        if credentials_path:
            load_credentials(credentials_path)

        self.data = load_config(config_path)
        self.project_root = get_project_root()

    @property
    def ragflow_base_url(self) -> str:
        return self.data.get("ragflow", {}).get("base_url", "http://localhost/")

    @property
    def ragflow_api_key(self) -> str:
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

    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get preset configuration by name."""
        return self.data.get("presets", {}).get(preset_name, {})

    def get_optimization_space(self, preset_name: str) -> list:
        """Get optimization space for a preset."""
        return self.data.get("optimization_spaces", {}).get(preset_name, [])

    @property
    def max_retries(self) -> int:
        return self.data.get("retry", {}).get("max_retries", 3)
