"""RAGFlow backend implementation."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import RAGBackend, BackendError


class RAGFlowAPIError(BackendError):
    """Exception for RAGFlow API errors."""
    pass


class RAGFlowBackend(RAGBackend):
    """RAGFlow implementation of RAG backend.

    This backend wraps the RAGFlow API for document ingestion,
    knowledge base management, and retrieval operations.
    """

    def __init__(self, base_url: str, api_key: str, max_retries: int = 3):
        """Initialize RAGFlow backend.

        Args:
            base_url: RAGFlow API base URL
            api_key: API authentication key
            max_retries: Maximum retry attempts for failed requests
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        # Track document IDs per collection for re-ingestion
        self._collection_docs: Dict[str, List[str]] = {}

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
                time.sleep(2 ** attempt)

        raise RAGFlowAPIError("Unexpected error in request handling")

    def test_connection(self) -> bool:
        """Test RAGFlow API connectivity."""
        try:
            self._request("GET", "/api/v1/datasets")
            return True
        except Exception:
            return False

    def _list_datasets(self) -> List[Dict]:
        """List all datasets."""
        result = self._request("GET", "/api/v1/datasets")
        data = result.get("data", [])
        if not isinstance(data, list):
            return []
        return data

    def _get_dataset_by_name(self, name: str) -> Optional[Dict]:
        """Get dataset by exact name match."""
        datasets = self._list_datasets()
        for ds in datasets:
            if ds.get("name") == name:
                return ds
        return None

    def _get_dataset_by_id(self, dataset_id: str) -> Optional[Dict]:
        """Get dataset by ID."""
        datasets = self._list_datasets()
        for ds in datasets:
            if ds.get("id") == dataset_id:
                return ds
        return None

    def get_embedding_model(self, collection_id: str) -> str:
        """Get the embedding model used by a collection."""
        ds = self._get_dataset_by_id(collection_id)
        if ds:
            return ds.get("embedding_model", "unknown")
        return "unknown"

    def _build_parser_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build RAGFlow parser_config from generic config."""
        parser_config = {}

        if "chunk_token_num" in config:
            parser_config["chunk_token_num"] = config["chunk_token_num"]

        if "overlap_token" in config:
            parser_config["overlap_token"] = config["overlap_token"]
            parser_config["delimiter"] = "\n"

        if "auto_questions" in config:
            parser_config["auto_questions"] = config["auto_questions"]

        if "auto_keywords" in config:
            parser_config["auto_keywords"] = config["auto_keywords"]

        if "layout_recognize" in config:
            parser_config["layout_recognize"] = config["layout_recognize"]

        if config.get("toc_enhance"):
            parser_config["html4excel"] = False

        return parser_config

    def create_collection(self, name: str, config: Dict[str, Any]) -> str:
        """Create a dataset (knowledge base) in RAGFlow."""
        # Delete if exists
        existing = self._get_dataset_by_name(name)
        if existing:
            self._request("DELETE", "/api/v1/datasets", json_data={"ids": [existing["id"]]})
            time.sleep(1)

        chunk_method = config.get("chunk_method", "naive")
        parser_config = self._build_parser_config(config)

        data = {
            "name": name,
            "chunk_method": chunk_method,
        }
        if parser_config:
            data["parser_config"] = parser_config

        result = self._request("POST", "/api/v1/datasets", json_data=data)
        dataset_id = result.get("data", {}).get("id", "")
        self._collection_docs[dataset_id] = []
        return dataset_id

    def upload_documents(
        self,
        collection_id: str,
        folder_path: Path,
    ) -> Dict[str, Any]:
        """Upload documents to a dataset."""
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
                    filename = item.name
                    mime_type, _ = mimetypes.guess_type(filename)
                    if mime_type is None:
                        mime_type = "application/octet-stream"

                    with open(item, "rb") as f:
                        files = {"file": (filename, f, mime_type)}
                        result = self._request(
                            "POST",
                            f"/api/v1/datasets/{collection_id}/documents",
                            files=files,
                        )

                    doc_data = result.get("data", [{}])
                    doc_id = doc_data[0].get("id") if doc_data else None

                    if doc_id:
                        document_ids.append(doc_id)
                    else:
                        failed_files.append({
                            "name": item.name,
                            "error": "No document ID returned"
                        })
                except RAGFlowAPIError as e:
                    failed_files.append({
                        "name": item.name,
                        "error": str(e)
                    })

        self._collection_docs[collection_id] = document_ids

        return {
            "document_ids": document_ids,
            "total_files": total_files,
            "uploaded_files": len(document_ids),
            "failed_files": failed_files,
        }

    def _list_documents(self, collection_id: str, document_id: Optional[str] = None) -> List[Dict]:
        """List documents in a dataset."""
        params = {}
        if document_id:
            params["id"] = document_id
        result = self._request(
            "GET",
            f"/api/v1/datasets/{collection_id}/documents",
            params=params,
        )
        return result.get("data", {}).get("docs", [])

    def _update_document(
        self,
        collection_id: str,
        document_id: str,
        chunk_method: Optional[str] = None,
        parser_config: Optional[Dict] = None,
    ) -> Dict:
        """Update document configuration."""
        data = {}
        if chunk_method:
            data["chunk_method"] = chunk_method
        if parser_config:
            data["parser_config"] = parser_config

        result = self._request(
            "PUT",
            f"/api/v1/datasets/{collection_id}/documents/{document_id}",
            json_data=data,
        )
        return result.get("data", {})

    def _parse_documents(self, collection_id: str, document_ids: List[str]) -> bool:
        """Start parsing documents."""
        try:
            self._request(
                "POST",
                f"/api/v1/datasets/{collection_id}/chunks",
                json_data={"document_ids": document_ids},
            )
            return True
        except RAGFlowAPIError:
            return False

    def _get_document_status(self, collection_id: str, document_id: str) -> Dict:
        """Get document parsing status."""
        docs = self._list_documents(collection_id, document_id)
        return docs[0] if docs else {}

    def _wait_for_parsing(
        self,
        collection_id: str,
        document_ids: List[str],
        timeout: int = 600,
        poll_interval: int = 5,
    ) -> Dict[str, str]:
        """Wait for documents to finish parsing."""
        start_time = time.time()
        final_statuses = {}

        while time.time() - start_time < timeout:
            all_done = True
            for doc_id in document_ids:
                if doc_id in final_statuses:
                    continue

                status = self._get_document_status(collection_id, doc_id)
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

    def ingest_with_config(
        self,
        collection_id: str,
        config: Dict[str, Any],
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """Re-ingest documents with new parser configuration."""
        document_ids = self._collection_docs.get(collection_id, [])
        if not document_ids:
            # Get document IDs from API if not tracked
            docs = self._list_documents(collection_id)
            document_ids = [d.get("id") for d in docs if d.get("id")]
            self._collection_docs[collection_id] = document_ids

        if not document_ids:
            return {
                "ingestion_status": "failed",
                "total_files": 0,
                "ingested_files": 0,
                "failed_files": [],
                "chunk_count": 0,
                "error": "No documents found in collection"
            }

        chunk_method = config.get("chunk_method", "naive")
        parser_config = self._build_parser_config(config)

        # Update parser config for all documents
        failed_updates = []
        successful_ids = []

        for doc_id in document_ids:
            try:
                self._update_document(
                    collection_id,
                    doc_id,
                    chunk_method=chunk_method,
                    parser_config=parser_config,
                )
                successful_ids.append(doc_id)
            except RAGFlowAPIError as e:
                doc_status = self._get_document_status(collection_id, doc_id)
                doc_name = doc_status.get("name", doc_id)
                failed_updates.append({
                    "name": doc_name,
                    "error": str(e)
                })

        if not successful_ids:
            return {
                "ingestion_status": "failed",
                "total_files": len(document_ids),
                "ingested_files": 0,
                "failed_files": failed_updates,
                "chunk_count": 0,
                "error": "All document config updates failed"
            }

        # Trigger re-parsing
        self._parse_documents(collection_id, successful_ids)

        # Wait for parsing completion
        statuses = self._wait_for_parsing(
            collection_id,
            successful_ids,
            timeout=timeout,
        )

        # Check parsing results
        failed_parsing = []
        successful_parsing = 0

        for doc_id, status in statuses.items():
            doc_status = self._get_document_status(collection_id, doc_id)
            doc_name = doc_status.get("name", doc_id)

            if status == "DONE":
                successful_parsing += 1
            else:
                progress_msg = doc_status.get("progress_msg", "")
                error_msg = f"Status: {status}"
                if progress_msg:
                    error_msg += f" - {progress_msg}"
                failed_parsing.append({
                    "name": doc_name,
                    "error": error_msg
                })

        # Get final chunk count
        stats = self.get_collection_stats(collection_id)
        chunk_count = stats.get("chunk_count", 0)

        # Combine all failures
        all_failed = failed_updates + failed_parsing
        total_files = len(document_ids)
        ingested_files = successful_parsing

        # Determine ingestion status
        if ingested_files == 0 or chunk_count == 0:
            ingestion_status = "failed"
        elif len(all_failed) > total_files / 2:
            ingestion_status = "partial"
        elif all_failed:
            ingestion_status = "partial"
        else:
            ingestion_status = "success"

        return {
            "ingestion_status": ingestion_status,
            "total_files": total_files,
            "ingested_files": ingested_files,
            "failed_files": all_failed,
            "chunk_count": chunk_count,
        }

    def retrieve(
        self,
        query: str,
        collection_ids: List[str],
        top_k: int = 3,
        similarity_threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a query."""
        data = {
            "question": query,
            "dataset_ids": collection_ids,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "highlight": False,
            "page_size": top_k,
        }

        result = self._request("POST", "/api/v1/retrieval", json_data=data)
        chunks = result.get("data", {}).get("chunks", [])

        results = []
        for i, chunk in enumerate(chunks[:top_k]):
            results.append({
                "rank": i + 1,
                "chunk_id": chunk.get("id", ""),
                "content": chunk.get("content", ""),
                "similarity": chunk.get("similarity", 0.0),
                "document_id": chunk.get("document_id", ""),
                "document_name": chunk.get("docnm_kwd", ""),
                "collection_id": chunk.get("dataset_id", ""),
                "keywords": chunk.get("important_keywords", []),
            })

        return results

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a dataset."""
        try:
            self._request("DELETE", "/api/v1/datasets", json_data={"ids": [collection_id]})
            if collection_id in self._collection_docs:
                del self._collection_docs[collection_id]
            return True
        except RAGFlowAPIError:
            return False

    def get_collection_stats(self, collection_id: str) -> Dict[str, Any]:
        """Get statistics for a dataset."""
        datasets = self._list_datasets()
        for ds in datasets:
            if ds.get("id") == collection_id:
                return {
                    "chunk_count": ds.get("chunk_count", 0),
                    "document_count": ds.get("document_count", 0),
                    "token_num": ds.get("token_num", 0),
                }
        return {"chunk_count": 0, "document_count": 0, "token_num": 0}

    def get_document_ids(self, collection_id: str) -> List[str]:
        """Get tracked document IDs for a collection."""
        return self._collection_docs.get(collection_id, [])
