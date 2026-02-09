"""Backend factory and exports for RAG backends."""

from typing import Any, Dict

from .base import RAGBackend, BackendError
from .ragflow import RAGFlowBackend, RAGFlowAPIError

# Export base classes and errors
__all__ = [
    "RAGBackend",
    "BackendError",
    "RAGFlowBackend",
    "RAGFlowAPIError",
    "OpenWebUIBackend",
    "OpenWebUIAPIError",
    "create_backend",
]


def create_backend(backend_type: str, config: Dict[str, Any]) -> RAGBackend:
    """Factory function to create the appropriate backend.

    Args:
        backend_type: Type of backend ("ragflow" or "openwebui")
        config: Backend-specific configuration containing:
            - base_url: API base URL
            - api_key: API authentication key
            - max_retries: (optional) Maximum retry attempts

    Returns:
        Configured RAGBackend instance

    Raises:
        ValueError: If backend_type is unknown
    """
    if backend_type == "ragflow":
        return RAGFlowBackend(
            base_url=config["base_url"],
            api_key=config["api_key"],
            max_retries=config.get("max_retries", 3),
        )

    elif backend_type == "openwebui":
        from .openwebui import OpenWebUIBackend
        return OpenWebUIBackend(
            base_url=config["base_url"],
            api_key=config["api_key"],
            max_retries=config.get("max_retries", 3),
        )

    else:
        raise ValueError(
            f"Unknown backend type: '{backend_type}'. "
            f"Supported backends: ragflow, openwebui"
        )


# Lazy import for OpenWebUI to avoid import errors if not needed
def __getattr__(name: str):
    if name == "OpenWebUIBackend":
        from .openwebui import OpenWebUIBackend
        return OpenWebUIBackend
    elif name == "OpenWebUIAPIError":
        from .openwebui import OpenWebUIAPIError
        return OpenWebUIAPIError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
