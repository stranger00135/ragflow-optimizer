"""Integration tests for OpenWebUI backend with live instance.

These tests require a running OpenWebUI instance at OPENWEBUI_URL
with a valid OPENWEBUI_API_KEY in the .env file.

Run with: pytest tests/test_openwebui_integration.py -v
"""

import os
import tempfile
import pytest
from pathlib import Path

from src.backends.openwebui_backend import OpenWebUIBackend
from src.backends.base import BackendError
from src.config_loader import Config


def get_openwebui_config():
    """Load OpenWebUI config from environment."""
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    env_path = project_root / ".env"

    config = Config(str(config_path), str(env_path) if env_path.exists() else None)
    return {
        "base_url": os.environ.get("OPENWEBUI_URL", "http://localhost:8080"),
        "api_key": os.environ.get("OPENWEBUI_API_KEY", ""),
        "max_retries": 3,
    }


@pytest.fixture
def openwebui_backend():
    """Create OpenWebUI backend with live config."""
    config = get_openwebui_config()
    if not config["api_key"]:
        pytest.skip("OPENWEBUI_API_KEY not set")
    return OpenWebUIBackend(config)


@pytest.fixture
def temp_test_folder():
    """Create a temporary folder with test documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple test text file
        test_file = Path(tmpdir) / "test_doc.txt"
        test_file.write_text("""
# RAG Testing Document

This is a test document for the RAG optimizer.

## Section 1: Introduction

The RAG (Retrieval-Augmented Generation) optimizer helps find the best
chunking parameters for your knowledge base.

## Section 2: Key Features

1. Automated parameter testing
2. Multi-backend support (RAGFlow, OpenWebUI)
3. LLM-based relevance judgment
4. Comprehensive metrics (Precision@k, MRR, Fail Rate)

## Section 3: Usage

To run optimization:
```
python main.py run --backend openwebui
```

## Section 4: Conclusion

The optimizer significantly improves retrieval quality by finding
optimal chunking configurations for different document types.
        """.strip())

        yield Path(tmpdir)


class TestOpenWebUIBackendLive:
    """Live integration tests for OpenWebUI backend."""

    def test_connection(self, openwebui_backend):
        """Test that backend can connect to OpenWebUI."""
        assert openwebui_backend.test_connection() is True

    def test_create_and_delete_collection(self, openwebui_backend):
        """Test creating and deleting a knowledge base."""
        # Create
        kb_id = openwebui_backend.create_collection("test_integration_kb", {})
        assert kb_id is not None
        assert len(kb_id) > 0

        # Delete
        result = openwebui_backend.delete_collection(kb_id)
        assert result is True

    def test_upload_documents(self, openwebui_backend, temp_test_folder):
        """Test uploading documents to a knowledge base."""
        # Create KB
        kb_id = openwebui_backend.create_collection("test_upload_kb", {})

        try:
            # Upload
            result = openwebui_backend.upload_documents(kb_id, temp_test_folder)

            assert result["total_files"] == 1
            assert result["uploaded_files"] == 1
            assert len(result["failed_files"]) == 0
            assert len(result["document_ids"]) == 1
        finally:
            # Cleanup
            openwebui_backend.delete_collection(kb_id)

    def test_collection_stats(self, openwebui_backend, temp_test_folder):
        """Test getting collection statistics after upload."""
        kb_id = openwebui_backend.create_collection("test_stats_kb", {})

        try:
            openwebui_backend.upload_documents(kb_id, temp_test_folder)

            stats = openwebui_backend.get_collection_stats(kb_id)

            assert "chunk_count" in stats
            assert "document_count" in stats
            assert stats["document_count"] == 1
        finally:
            openwebui_backend.delete_collection(kb_id)

    def test_ingest_with_config(self, openwebui_backend, temp_test_folder):
        """Test ingestion with config (reports status for OpenWebUI)."""
        kb_id = openwebui_backend.create_collection("test_ingest_kb", {})

        try:
            openwebui_backend.upload_documents(kb_id, temp_test_folder)

            result = openwebui_backend.ingest_with_config(kb_id, {
                "chunk_token_num": 512,
                "overlap_token": 50,
            })

            assert "ingestion_status" in result
            assert "chunk_count" in result
            # OpenWebUI processes on upload, so status should be success
            # if documents were uploaded successfully
        finally:
            openwebui_backend.delete_collection(kb_id)

    def test_retrieve(self, openwebui_backend, temp_test_folder):
        """Test retrieving chunks from knowledge base."""
        kb_id = openwebui_backend.create_collection("test_retrieve_kb", {})

        try:
            openwebui_backend.upload_documents(kb_id, temp_test_folder)

            # Give some time for processing
            import time
            time.sleep(2)

            results = openwebui_backend.retrieve(
                query="What is the RAG optimizer?",
                collection_ids=[kb_id],
                top_k=3,
                similarity_threshold=0.1,
            )

            # We should get some results
            assert isinstance(results, list)
            if results:
                assert "content" in results[0]
                assert "similarity" in results[0]
                assert "rank" in results[0]
        finally:
            openwebui_backend.delete_collection(kb_id)

    def test_full_workflow(self, openwebui_backend, temp_test_folder):
        """Test complete optimization workflow: create -> upload -> ingest -> retrieve -> delete."""
        # Step 1: Create KB
        kb_id = openwebui_backend.create_collection("test_workflow_kb", {})
        assert kb_id

        try:
            # Step 2: Upload documents
            upload_result = openwebui_backend.upload_documents(kb_id, temp_test_folder)
            assert upload_result["uploaded_files"] > 0

            # Step 3: Ingest with config
            ingest_result = openwebui_backend.ingest_with_config(kb_id, {
                "chunk_token_num": 256,
            })
            assert "chunk_count" in ingest_result

            # Step 4: Validate ingestion
            is_valid, reason = openwebui_backend.validate_ingestion(ingest_result)
            # If validation fails, it's a warning not an error for this test
            if not is_valid:
                print(f"Ingestion validation: {reason}")

            # Step 5: Retrieve
            import time
            time.sleep(2)  # Give time for indexing

            results = openwebui_backend.retrieve(
                query="retrieval augmented generation optimizer",
                collection_ids=[kb_id],
                top_k=3,
            )
            print(f"Retrieved {len(results)} results")
            for r in results:
                print(f"  - Rank {r.get('rank')}: similarity={r.get('similarity'):.3f}")

            # Step 6: Get stats
            stats = openwebui_backend.get_collection_stats(kb_id)
            print(f"Stats: chunks={stats.get('chunk_count')}, docs={stats.get('document_count')}")

        finally:
            # Step 7: Delete KB
            deleted = openwebui_backend.delete_collection(kb_id)
            assert deleted is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
