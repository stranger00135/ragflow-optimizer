"""OpenWebUI backend implementation."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from .base import RAGBackend, BackendError


class OpenWebUIAPIError(BackendError):
    """Exception for OpenWebUI API errors."""
    pass


class OpenWebUIBackend(RAGBackend):
    """OpenWebUI implementation of RAG backend.

    This backend wraps the OpenWebUI API for document ingestion,
    knowledge base management, and retrieval operations.

    OpenWebUI API Reference:
    - https://docs.openwebui.com/getting-started/api-endpoints/
    - https://docs.openwebui.com/features/rag/
    """

    def __init__(self, base_url: str, api_key: str, max_retries: int = 3):
        """Initialize OpenWebUI backend.

        Args:
            base_url: OpenWebUI API base URL (e.g., http://localhost:3000)
            api_key: API authentication key (Bearer token)
            max_retries: Maximum retry attempts for failed requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })
        # Track files per knowledge base for management
        self._kb_files: Dict[str, List[str]] = {}

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        files: Optional[Dict] = None,
        data: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                if files:
                    # For file uploads, don't set Content-Type header
                    headers = {
                        "Authorization": f"Bearer {self.api_key}",
                        "Accept": "application/json",
                    }
                    response = requests.request(
                        method=method,
                        url=url,
                        params=params,
                        files=files,
                        data=data,
                        headers=headers,
                        timeout=300,
                    )
                else:
                    response = self.session.request(
                        method=method,
                        url=url,
                        params=params,
                        json=json_data,
                        timeout=300,
                    )

                response.raise_for_status()

                # Handle empty responses
                if not response.content:
                    return {}

                return response.json()
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise OpenWebUIAPIError(f"Request failed after {self.max_retries} retries: {e}")
                time.sleep(2 ** attempt)

        raise OpenWebUIAPIError("Unexpected error in request handling")

    def test_connection(self) -> bool:
        """Test OpenWebUI API connectivity."""
        try:
            # Try to list knowledge bases
            self._request("GET", "/api/v1/knowledge/")
            return True
        except Exception:
            return False

    def get_embedding_model(self, collection_id: str) -> str:
        """Get the embedding model used by OpenWebUI.

        Note: OpenWebUI uses a global embedding model setting,
        not per-knowledge-base settings.
        """
        try:
            # OpenWebUI doesn't expose embedding model per KB
            # Try to get from config endpoint if available
            response = self._request("GET", "/api/v1/configs")
            return response.get("rag", {}).get("embedding_model", "unknown")
        except Exception:
            return "unknown"

    def create_collection(self, name: str, config: Dict[str, Any]) -> str:
        """Create a knowledge base in OpenWebUI.

        Args:
            name: Name for the knowledge base
            config: Configuration (not directly used by OpenWebUI per-KB)

        Returns:
            Knowledge base ID
        """
        response = self._request(
            "POST",
            "/api/v1/knowledge/create",
            json_data={
                "name": name,
                "description": f"RAG Optimizer temp KB: {name}",
            }
        )

        kb_id = response.get("id", "")
        self._kb_files[kb_id] = []
        return kb_id

    def _wait_for_file_processing(
        self,
        file_id: str,
        timeout: int = 300,
        poll_interval: int = 2,
    ) -> bool:
        """Wait for OpenWebUI to finish processing a file.

        Args:
            file_id: File ID to monitor
            timeout: Maximum wait time in seconds
            poll_interval: Seconds between status checks

        Returns:
            True if processing completed successfully

        Raises:
            OpenWebUIAPIError: If processing fails or times out
        """
        start = time.time()
        while time.time() - start < timeout:
            try:
                response = self._request("GET", f"/api/v1/files/{file_id}")
                # OpenWebUI file status field - check for completion
                # Status can be: "pending", "processing", "completed", "failed"
                status = response.get("status", response.get("meta", {}).get("status", ""))

                if status in ("completed", "indexed", ""):
                    # Empty status or completed means ready
                    # Some OpenWebUI versions don't return status once done
                    return True
                elif status == "failed":
                    raise OpenWebUIAPIError(f"File processing failed: {file_id}")
                elif status in ("pending", "processing"):
                    time.sleep(poll_interval)
                else:
                    # Unknown status - assume completed after brief wait
                    time.sleep(poll_interval)
                    return True
            except OpenWebUIAPIError:
                raise
            except Exception:
                # If we can't get status, wait briefly and continue
                time.sleep(poll_interval)

        raise OpenWebUIAPIError(f"File processing timeout: {file_id}")

    def upload_documents(
        self,
        collection_id: str,
        folder_path: Path,
    ) -> Dict[str, Any]:
        """Upload documents to OpenWebUI and add to knowledge base.

        Args:
            collection_id: Knowledge base ID
            folder_path: Path to folder containing documents

        Returns:
            Dict with document_ids, upload statistics, and failed files
        """
        import mimetypes

        document_ids = []
        failed_files = []
        total_files = 0

        for item in folder_path.iterdir():
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                if item.name.startswith("."):
                    continue

                total_files += 1

                try:
                    # Step 1: Upload file to OpenWebUI
                    filename = item.name
                    mime_type, _ = mimetypes.guess_type(filename)
                    if mime_type is None:
                        mime_type = "application/octet-stream"

                    with open(item, "rb") as f:
                        response = self._request(
                            "POST",
                            "/api/v1/files/",
                            files={"file": (filename, f, mime_type)},
                        )

                    file_id = response.get("id", "")
                    if not file_id:
                        failed_files.append({
                            "name": item.name,
                            "error": "No file ID returned"
                        })
                        continue

                    # Step 2: Wait for file processing
                    try:
                        self._wait_for_file_processing(file_id, timeout=300)
                    except OpenWebUIAPIError as e:
                        failed_files.append({
                            "name": item.name,
                            "error": str(e)
                        })
                        continue

                    # Step 3: Add file to knowledge base
                    self._request(
                        "POST",
                        f"/api/v1/knowledge/{collection_id}/file/add",
                        json_data={"file_id": file_id}
                    )

                    document_ids.append(file_id)

                except OpenWebUIAPIError as e:
                    failed_files.append({
                        "name": item.name,
                        "error": str(e)
                    })
                except Exception as e:
                    failed_files.append({
                        "name": item.name,
                        "error": f"Unexpected error: {e}"
                    })

        self._kb_files[collection_id] = document_ids

        return {
            "document_ids": document_ids,
            "total_files": total_files,
            "uploaded_files": len(document_ids),
            "failed_files": failed_files,
        }

    def ingest_with_config(
        self,
        collection_id: str,
        config: Dict[str, Any],
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """Re-ingest with new chunking configuration.

        Note: OpenWebUI chunking parameters are typically set at the
        instance level (Admin Settings), not per-knowledge-base.
        This method works within those constraints.

        For full optimization, the OpenWebUI instance settings should be
        configured before running optimization:
        - CHUNK_SIZE
        - CHUNK_OVERLAP
        - CHUNK_MIN_SIZE_TARGET

        Args:
            collection_id: Knowledge base ID
            config: Chunking configuration (limited effectiveness with OpenWebUI)
            timeout: Not used (OpenWebUI processes on upload)

        Returns:
            Ingestion result with status and chunk count
        """
        # OpenWebUI doesn't support per-KB re-ingestion like RAGFlow
        # Documents are processed on upload with global settings
        # We can only report current KB state

        document_ids = self._kb_files.get(collection_id, [])

        try:
            stats = self.get_collection_stats(collection_id)
            chunk_count = stats.get("chunk_count", 0)
            document_count = stats.get("document_count", len(document_ids))

            # Determine status based on what we have
            if document_count == 0:
                return {
                    "ingestion_status": "failed",
                    "total_files": 0,
                    "ingested_files": 0,
                    "failed_files": [],
                    "chunk_count": 0,
                    "error": "No documents in knowledge base"
                }

            if chunk_count == 0:
                return {
                    "ingestion_status": "failed",
                    "total_files": document_count,
                    "ingested_files": document_count,
                    "failed_files": [],
                    "chunk_count": 0,
                    "error": "No chunks generated"
                }

            return {
                "ingestion_status": "success",
                "total_files": document_count,
                "ingested_files": document_count,
                "failed_files": [],
                "chunk_count": chunk_count,
            }

        except OpenWebUIAPIError as e:
            return {
                "ingestion_status": "failed",
                "total_files": len(document_ids),
                "ingested_files": 0,
                "failed_files": [],
                "chunk_count": 0,
                "error": str(e)
            }

    def retrieve(
        self,
        query: str,
        collection_ids: List[str],
        top_k: int = 3,
        similarity_threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Query knowledge bases for relevant chunks.

        Args:
            query: Search query
            collection_ids: List of knowledge base IDs to search
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score

        Returns:
            List of chunk results with content and similarity scores
        """
        all_results = []

        for kb_id in collection_ids:
            try:
                # OpenWebUI knowledge query endpoint
                response = self._request(
                    "POST",
                    f"/api/v1/knowledge/{kb_id}/query",
                    json_data={
                        "query": query,
                        "k": top_k,
                    }
                )

                chunks = response.get("results", response.get("documents", []))

                for chunk in chunks:
                    # Handle different response formats
                    if isinstance(chunk, dict):
                        content = chunk.get("content", chunk.get("text", ""))
                        score = chunk.get("score", chunk.get("similarity", 0.0))
                    else:
                        content = str(chunk)
                        score = 0.5  # Default score if not provided

                    if score >= similarity_threshold:
                        all_results.append({
                            "content": content,
                            "similarity": score,
                            "collection_id": kb_id,
                            "document_id": chunk.get("document_id", "") if isinstance(chunk, dict) else "",
                            "document_name": chunk.get("document_name", chunk.get("filename", "")) if isinstance(chunk, dict) else "",
                        })

            except OpenWebUIAPIError:
                # Skip this KB if query fails
                continue

        # Sort by similarity and assign ranks
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        final_results = []
        for i, result in enumerate(all_results[:top_k]):
            result["rank"] = i + 1
            final_results.append(result)

        return final_results

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a knowledge base.

        Args:
            collection_id: Knowledge base ID to delete

        Returns:
            True if deletion succeeded
        """
        try:
            self._request(
                "DELETE",
                f"/api/v1/knowledge/{collection_id}/delete"
            )
            if collection_id in self._kb_files:
                del self._kb_files[collection_id]
            return True
        except OpenWebUIAPIError:
            return False

    def get_collection_stats(self, collection_id: str) -> Dict[str, Any]:
        """Get knowledge base statistics.

        Args:
            collection_id: Knowledge base ID

        Returns:
            Dict with chunk_count, document_count
        """
        try:
            response = self._request("GET", f"/api/v1/knowledge/{collection_id}")

            # OpenWebUI may return stats in different formats
            files = response.get("files", [])
            chunk_count = response.get("chunk_count", 0)

            # If chunk_count not provided, estimate from files
            if chunk_count == 0 and files:
                # Sum up chunks from individual files if available
                for f in files:
                    chunk_count += f.get("chunk_count", f.get("chunks", 1))

            return {
                "chunk_count": chunk_count,
                "document_count": len(files),
            }

        except OpenWebUIAPIError:
            return {"chunk_count": 0, "document_count": 0}

    def get_document_ids(self, collection_id: str) -> List[str]:
        """Get tracked file IDs for a knowledge base."""
        return self._kb_files.get(collection_id, [])
