"""RAGFlow API Client wrapper with retry logic."""

import time
from typing import Any, Dict, List, Optional

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class RAGFlowClient:
    """Client for interacting with RAGFlow API."""

    def __init__(self, base_url: str, api_key: str, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        json_data: Optional[Dict] = None,
        files: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}{endpoint}"

        for attempt in range(self.max_retries):
            try:
                if files:
                    # For file uploads, use raw requests without Content-Type header
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    response = requests.request(
                        method=method,
                        url=url,
                        params=params,
                        files=files,
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
                result = response.json()

                if result.get("code") != 0:
                    raise RAGFlowAPIError(
                        f"API error: {result.get('message', 'Unknown error')}"
                    )
                return result
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RAGFlowAPIError(f"Request failed after {self.max_retries} retries: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff

        raise RAGFlowAPIError("Unexpected error in request handling")

    def test_connection(self) -> bool:
        """Test API connectivity."""
        try:
            self._request("GET", "/api/v1/datasets")
            return True
        except Exception:
            return False

    # Dataset (Knowledge Base) Operations
    def list_datasets(self) -> List[Dict]:
        """List all datasets accessible by current user."""
        result = self._request("GET", "/api/v1/datasets")
        data = result.get("data", [])
        # Handle case where API returns False or non-list
        if not isinstance(data, list):
            return []
        return data

    def get_dataset_by_name(self, name: str) -> Optional[Dict]:
        """Get dataset by exact name match."""
        datasets = self.list_datasets()
        for ds in datasets:
            if ds.get("name") == name:
                return ds
        return None

    def create_dataset(
        self,
        name: str,
        chunk_method: str = "naive",
        parser_config: Optional[Dict] = None,
    ) -> Dict:
        """Create a new dataset (knowledge base)."""
        data = {
            "name": name,
            "chunk_method": chunk_method,
        }
        if parser_config:
            data["parser_config"] = parser_config

        result = self._request("POST", "/api/v1/datasets", json_data=data)
        return result.get("data", {})

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete a dataset by ID."""
        try:
            self._request("DELETE", f"/api/v1/datasets", json_data={"ids": [dataset_id]})
            return True
        except RAGFlowAPIError:
            return False

    # Document Operations
    def upload_document(
        self,
        dataset_id: str,
        file_path: str,
    ) -> Dict:
        """Upload a document to a dataset."""
        import mimetypes

        filename = file_path.split("/")[-1]
        mime_type, _ = mimetypes.guess_type(filename)
        if mime_type is None:
            mime_type = "application/octet-stream"

        with open(file_path, "rb") as f:
            files = {"file": (filename, f, mime_type)}
            result = self._request(
                "POST",
                f"/api/v1/datasets/{dataset_id}/documents",
                files=files,
            )
        return result.get("data", [{}])[0] if result.get("data") else {}

    def list_documents(
        self,
        dataset_id: str,
        document_id: Optional[str] = None,
    ) -> List[Dict]:
        """List documents in a dataset."""
        params = {}
        if document_id:
            params["id"] = document_id
        result = self._request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/documents",
            params=params,
        )
        return result.get("data", {}).get("docs", [])

    def update_document(
        self,
        dataset_id: str,
        document_id: str,
        chunk_method: Optional[str] = None,
        parser_config: Optional[Dict] = None,
        enabled: Optional[int] = None,
    ) -> Dict:
        """Update document configuration."""
        data = {}
        if chunk_method:
            data["chunk_method"] = chunk_method
        if parser_config:
            data["parser_config"] = parser_config
        if enabled is not None:
            data["enabled"] = enabled

        result = self._request(
            "PUT",
            f"/api/v1/datasets/{dataset_id}/documents/{document_id}",
            json_data=data,
        )
        return result.get("data", {})

    def delete_documents(self, dataset_id: str, document_ids: List[str]) -> bool:
        """Delete documents from a dataset."""
        try:
            self._request(
                "DELETE",
                f"/api/v1/datasets/{dataset_id}/documents",
                json_data={"ids": document_ids},
            )
            return True
        except RAGFlowAPIError:
            return False

    # Parsing Operations
    def parse_documents(self, dataset_id: str, document_ids: List[str]) -> bool:
        """Start parsing documents."""
        try:
            self._request(
                "POST",
                f"/api/v1/datasets/{dataset_id}/chunks",
                json_data={"document_ids": document_ids},
            )
            return True
        except RAGFlowAPIError:
            return False

    def get_document_status(self, dataset_id: str, document_id: str) -> Dict:
        """Get document parsing status."""
        docs = self.list_documents(dataset_id, document_id)
        return docs[0] if docs else {}

    def wait_for_parsing(
        self,
        dataset_id: str,
        document_ids: List[str],
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> Dict[str, str]:
        """Wait for documents to finish parsing.

        Returns:
            Dict mapping document IDs to final status (DONE, FAIL, etc.)
        """
        start_time = time.time()
        final_statuses = {}

        while time.time() - start_time < timeout:
            all_done = True
            for doc_id in document_ids:
                if doc_id in final_statuses:
                    continue

                status = self.get_document_status(dataset_id, doc_id)
                run_status = status.get("run", "UNSTART")

                if run_status in ("DONE", "3"):
                    final_statuses[doc_id] = "DONE"
                elif run_status == "FAIL":
                    final_statuses[doc_id] = "FAIL"
                else:
                    all_done = False

            if all_done or len(final_statuses) == len(document_ids):
                return final_statuses

            time.sleep(poll_interval)

        # Timeout - return what we have
        for doc_id in document_ids:
            if doc_id not in final_statuses:
                final_statuses[doc_id] = "TIMEOUT"
        return final_statuses

    # Retrieval Operations
    def retrieve_chunks(
        self,
        question: str,
        dataset_ids: List[str],
        top_k: int = 3,
        similarity_threshold: float = 0.2,
        highlight: bool = False,
    ) -> List[Dict]:
        """Retrieve relevant chunks for a question."""
        data = {
            "question": question,
            "dataset_ids": dataset_ids,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "highlight": highlight,
            "page_size": top_k,
        }

        result = self._request("POST", "/api/v1/retrieval", json_data=data)
        return result.get("data", {}).get("chunks", [])

    # Chunk Operations
    def list_chunks(
        self,
        dataset_id: str,
        document_id: str,
        page: int = 1,
        page_size: int = 100,
    ) -> List[Dict]:
        """List chunks for a document."""
        params = {
            "page": page,
            "page_size": page_size,
        }
        result = self._request(
            "GET",
            f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
            params=params,
        )
        return result.get("data", {}).get("chunks", [])


class RAGFlowAPIError(Exception):
    """Custom exception for RAGFlow API errors."""
    pass
