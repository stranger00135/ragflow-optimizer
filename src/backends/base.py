"""Abstract base class for RAG backends."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class RAGBackend(ABC):
    """Abstract base class for RAG backends.

    This interface abstracts the differences between RAG platforms
    (RAGFlow, OpenWebUI, etc.) to enable multi-platform optimization.

    All backends must implement these core operations:
    - Collection management (create, delete, stats)
    - Document handling (upload, ingest with config)
    - Retrieval (search with top-k, similarity threshold)
    - Connection testing
    """

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".txt", ".md",
        ".xlsx", ".xls", ".csv",
        ".ppt", ".pptx",
        ".jpg", ".jpeg", ".png",
    }

    @abstractmethod
    def create_collection(self, name: str, config: Dict[str, Any]) -> str:
        """Create a collection/knowledge base.

        Args:
            name: Name for the collection
            config: Initial configuration (may include chunking params)

        Returns:
            Collection ID (string)
        """
        pass

    @abstractmethod
    def upload_documents(
        self,
        collection_id: str,
        folder_path: Path,
    ) -> Dict[str, Any]:
        """Upload documents to a collection.

        Files are uploaded but may not be immediately processed.
        Use ingest_with_config to trigger processing with specific settings.

        Args:
            collection_id: Target collection ID
            folder_path: Path to folder containing documents

        Returns:
            Dict with:
                - document_ids: List[str] - IDs of uploaded documents
                - total_files: int - Total files found
                - uploaded_files: int - Successfully uploaded
                - failed_files: List[Dict] - Files that failed with error details
        """
        pass

    @abstractmethod
    def ingest_with_config(
        self,
        collection_id: str,
        config: Dict[str, Any],
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """Ingest/re-ingest documents with specified configuration.

        This is the key method for optimization - it applies chunking
        settings and triggers document processing.

        Args:
            collection_id: Target collection ID
            config: Chunking configuration (chunk_token_num, overlap_token, etc.)
            timeout: Maximum wait time for processing completion

        Returns:
            Dict with:
                - ingestion_status: str - "success", "partial", or "failed"
                - total_files: int
                - ingested_files: int
                - failed_files: List[Dict]
                - chunk_count: int
                - error: Optional[str] - Error message if failed
        """
        pass

    @abstractmethod
    def retrieve(
        self,
        query: str,
        collection_ids: List[str],
        top_k: int = 3,
        similarity_threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Search query text
            collection_ids: List of collection IDs to search
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of chunk dictionaries, each containing:
                - rank: int - Position in results (1-indexed)
                - content: str - Chunk text
                - similarity: float - Similarity score
                - document_id: str (optional)
                - document_name: str (optional)
                - collection_id: str
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection.

        Args:
            collection_id: Collection to delete

        Returns:
            True if deleted successfully, False otherwise
        """
        pass

    @abstractmethod
    def get_collection_stats(self, collection_id: str) -> Dict[str, Any]:
        """Get statistics for a collection.

        Args:
            collection_id: Collection to query

        Returns:
            Dict with:
                - chunk_count: int
                - document_count: int
                - token_num: int (optional, if available)
        """
        pass

    @abstractmethod
    def test_connection(self) -> bool:
        """Test backend API connectivity.

        Returns:
            True if connection successful, False otherwise
        """
        pass

    @abstractmethod
    def get_embedding_model(self, collection_id: str) -> str:
        """Get the embedding model used by a collection.

        Args:
            collection_id: Collection to query

        Returns:
            Model name/identifier string, or "unknown" if not available
        """
        pass

    def validate_ingestion(
        self,
        ingestion_result: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """Validate ingestion result using standard disqualification rules.

        Default implementation applies these rules:
        1. All files fail to ingest -> disqualified
        2. chunk_count = 0 -> disqualified
        3. >50% files fail -> disqualified

        Backends can override this method for custom validation logic.

        Args:
            ingestion_result: Result from ingest_with_config

        Returns:
            Tuple of (is_valid, disqualification_reason)
            is_valid=True means retrieval can proceed
            is_valid=False means experiment should be disqualified
        """
        total_files = ingestion_result.get("total_files", 0)
        ingested_files = ingestion_result.get("ingested_files", 0)
        chunk_count = ingestion_result.get("chunk_count", 0)
        failed_files = ingestion_result.get("failed_files", [])

        # Rule 1: All files fail to ingest
        if ingested_files == 0 and total_files > 0:
            return False, f"All files failed to ingest ({total_files} files)"

        # Rule 2: chunk_count = 0
        if chunk_count == 0:
            return False, f"0 chunks generated from {total_files} files"

        # Rule 3: >50% files fail
        if total_files > 0:
            fail_rate = len(failed_files) / total_files
            if fail_rate > 0.5:
                return False, f">{int(fail_rate*100)}% files failed ({len(failed_files)}/{total_files})"

        # Passed validation
        return True, None

    def get_document_ids(self, collection_id: str) -> List[str]:
        """Get tracked document IDs for a collection.

        Default implementation returns empty list.
        Backends should override to provide actual tracking.

        Args:
            collection_id: Collection to query

        Returns:
            List of document IDs
        """
        return []

    def get_collection_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a collection by name.

        Default implementation returns None if unsupported.

        Args:
            name: Collection name

        Returns:
            Collection dict if found, otherwise None
        """
        return None


class BackendError(Exception):
    """Base exception for backend errors."""
    pass
