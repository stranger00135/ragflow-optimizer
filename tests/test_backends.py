"""Tests for backend abstraction layer."""

import pytest
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

from src.backends.base import RAGBackend, BackendError
from src.backends.ragflow_backend import RAGFlowBackend
from src.backends.openwebui_backend import OpenWebUIBackend
from src.backends.registry import get_backend


class TestRAGBackendAbstract:
    """Test that RAGBackend is properly abstract."""

    def test_ragbackend_is_abstract(self):
        """Cannot instantiate abstract RAGBackend directly."""
        with pytest.raises(TypeError):
            RAGBackend()

    def test_all_abstract_methods_defined(self):
        """All required abstract methods are defined."""
        abstract_methods = [
            "create_collection",
            "upload_documents",
            "ingest_with_config",
            "retrieve",
            "delete_collection",
            "get_collection_stats",
            "test_connection",
            "get_embedding_model",
        ]
        for method in abstract_methods:
            assert hasattr(RAGBackend, method), f"Missing abstract method: {method}"


class TestRAGBackendValidation:
    """Test default validation logic in RAGBackend."""

    class ConcreteBackend(RAGBackend):
        """Minimal concrete implementation for testing."""

        def create_collection(self, name: str, config: Dict[str, Any]) -> str:
            return "test_id"

        def upload_documents(self, collection_id: str, folder_path: Path) -> Dict[str, Any]:
            return {}

        def ingest_with_config(self, collection_id: str, config: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
            return {}

        def retrieve(self, query: str, collection_ids: List[str], top_k: int = 3, similarity_threshold: float = 0.2) -> List[Dict[str, Any]]:
            return []

        def delete_collection(self, collection_id: str) -> bool:
            return True

        def get_collection_stats(self, collection_id: str) -> Dict[str, Any]:
            return {}

        def test_connection(self) -> bool:
            return True

        def get_embedding_model(self, collection_id: str) -> str:
            return "test_model"

    def test_validate_ingestion_success(self):
        """Validation passes for valid ingestion result."""
        backend = self.ConcreteBackend()
        result = {
            "total_files": 5,
            "ingested_files": 5,
            "chunk_count": 100,
            "failed_files": [],
        }
        is_valid, reason = backend.validate_ingestion(result)
        assert is_valid is True
        assert reason is None

    def test_validate_ingestion_all_files_failed(self):
        """Validation fails when all files fail to ingest."""
        backend = self.ConcreteBackend()
        result = {
            "total_files": 5,
            "ingested_files": 0,
            "chunk_count": 0,
            "failed_files": [{"name": f"file{i}.pdf"} for i in range(5)],
        }
        is_valid, reason = backend.validate_ingestion(result)
        assert is_valid is False
        assert "All files failed" in reason

    def test_validate_ingestion_zero_chunks(self):
        """Validation fails when chunk_count is 0."""
        backend = self.ConcreteBackend()
        result = {
            "total_files": 5,
            "ingested_files": 5,
            "chunk_count": 0,
            "failed_files": [],
        }
        is_valid, reason = backend.validate_ingestion(result)
        assert is_valid is False
        assert "0 chunks" in reason

    def test_validate_ingestion_high_fail_rate(self):
        """Validation fails when >50% of files fail."""
        backend = self.ConcreteBackend()
        result = {
            "total_files": 10,
            "ingested_files": 4,
            "chunk_count": 50,
            "failed_files": [{"name": f"file{i}.pdf"} for i in range(6)],
        }
        is_valid, reason = backend.validate_ingestion(result)
        assert is_valid is False
        assert "files failed" in reason


class TestRAGFlowBackend:
    """Test RAGFlow backend adapter."""

    @pytest.fixture
    def mock_client(self):
        """Create mock RAGFlow client."""
        with patch("src.backends.ragflow_backend.RAGFlowClient") as mock:
            yield mock.return_value

    @pytest.fixture
    def mock_ingestion(self):
        """Create mock ingestion engine."""
        with patch("src.backends.ragflow_backend.IngestionEngine") as mock:
            yield mock.return_value

    @pytest.fixture
    def mock_retrieval(self):
        """Create mock retrieval engine."""
        with patch("src.backends.ragflow_backend.RetrievalEngine") as mock:
            yield mock.return_value

    def test_ragflow_init(self, mock_client, mock_ingestion, mock_retrieval):
        """RAGFlow backend initializes correctly."""
        backend = RAGFlowBackend(
            base_url="http://test.example.com",
            api_key="test_key",
            max_retries=3,
        )
        assert backend is not None

    def test_ragflow_test_connection(self, mock_client, mock_ingestion, mock_retrieval):
        """RAGFlow backend test_connection delegates to client."""
        mock_client.test_connection.return_value = True
        backend = RAGFlowBackend(
            base_url="http://test.example.com",
            api_key="test_key",
        )
        assert backend.test_connection() is True


class TestOpenWebUIBackend:
    """Test OpenWebUI backend adapter."""

    def test_openwebui_init_requires_config(self):
        """OpenWebUI backend requires base_url and api_key in config."""
        with pytest.raises(BackendError) as exc_info:
            OpenWebUIBackend({})
        assert "base_url" in str(exc_info.value) or "api_key" in str(exc_info.value)

    def test_openwebui_init_with_config(self):
        """OpenWebUI backend initializes with valid config."""
        backend = OpenWebUIBackend({
            "base_url": "http://localhost:3000",
            "api_key": "test_key",
        })
        assert backend.base_url == "http://localhost:3000"
        assert backend.api_key == "test_key"

    @patch("requests.Session")
    def test_openwebui_test_connection(self, mock_session_class):
        """OpenWebUI test_connection tries to list knowledge bases."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = b'[]'
        mock_response.json.return_value = []
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        backend = OpenWebUIBackend({
            "base_url": "http://localhost:3000",
            "api_key": "test_key",
        })
        backend.session = mock_session

        result = backend.test_connection()
        assert result is True

    @patch("requests.Session")
    def test_openwebui_create_collection(self, mock_session_class):
        """OpenWebUI creates knowledge base via API."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = b'{"id": "kb_123"}'
        mock_response.json.return_value = {"id": "kb_123"}
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        backend = OpenWebUIBackend({
            "base_url": "http://localhost:3000",
            "api_key": "test_key",
        })
        backend.session = mock_session

        kb_id = backend.create_collection("test_kb", {})
        assert kb_id == "kb_123"

    @patch("requests.Session")
    def test_openwebui_retrieve(self, mock_session_class):
        """OpenWebUI retrieves and ranks results."""
        mock_session = MagicMock()
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.content = b'{"results": [{"content": "test content", "score": 0.95}]}'
        mock_response.json.return_value = {
            "results": [{"content": "test content", "score": 0.95}]
        }
        mock_session.request.return_value = mock_response
        mock_session_class.return_value = mock_session

        backend = OpenWebUIBackend({
            "base_url": "http://localhost:3000",
            "api_key": "test_key",
        })
        backend.session = mock_session

        results = backend.retrieve("test query", ["kb_123"], top_k=3)
        assert len(results) == 1
        assert results[0]["content"] == "test content"
        assert results[0]["similarity"] == 0.95
        assert results[0]["rank"] == 1


class TestBackendRegistry:
    """Test backend factory/registry."""

    @patch("src.backends.registry.RAGFlowBackend")
    def test_get_backend_ragflow(self, mock_ragflow):
        """Registry returns RAGFlow backend for ragflow type."""
        mock_config = MagicMock()
        mock_config.backend_name = "ragflow"
        mock_config.ragflow_base_url = "http://test.example.com"
        mock_config.ragflow_api_key = "test_key"
        mock_config.max_retries = 3

        backend = get_backend(mock_config)
        mock_ragflow.assert_called_once()

    @patch("src.backends.registry.OpenWebUIBackend")
    def test_get_backend_openwebui(self, mock_openwebui):
        """Registry returns OpenWebUI backend for openwebui type."""
        mock_config = MagicMock()
        mock_config.backend_name = "openwebui"
        mock_config.backend_config = {
            "base_url": "http://localhost:3000",
            "api_key": "test_key",
        }

        backend = get_backend(mock_config)
        mock_openwebui.assert_called_once()

    def test_get_backend_unknown_raises(self):
        """Registry raises BackendError for unknown backend type."""
        mock_config = MagicMock()
        mock_config.backend_name = "unknown_backend"

        with pytest.raises(BackendError) as exc_info:
            get_backend(mock_config)
        assert "Unknown backend" in str(exc_info.value)
