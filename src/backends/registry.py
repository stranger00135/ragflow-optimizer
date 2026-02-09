"""Backend registry and factory."""

from typing import Any, Dict

from .base import BackendError, RAGBackend
from .openwebui_backend import OpenWebUIBackend
from .ragflow_backend import RAGFlowBackend
from ..config_loader import Config


def get_backend(config: Config) -> RAGBackend:
    """Instantiate backend based on configuration."""
    backend_name = config.backend_name

    if backend_name == "ragflow":
        return RAGFlowBackend(
            base_url=config.ragflow_base_url,
            api_key=config.ragflow_api_key,
            max_retries=config.max_retries,
        )

    if backend_name == "openwebui":
        return OpenWebUIBackend(config.backend_config)

    raise BackendError(f"Unknown backend: {backend_name}")
