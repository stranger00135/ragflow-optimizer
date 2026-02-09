"""RAGFlow backend adapter."""

from typing import Any, Dict, List, Optional

from .base import BackendError, RAGBackend
from ..ingestion_engine import IngestionEngine
from ..ragflow_client import RAGFlowAPIError, RAGFlowClient
from ..retrieval_engine import RetrievalEngine


class RAGFlowBackend(RAGBackend):
    """Backend adapter for RAGFlow."""

    def __init__(self, base_url: str, api_key: str, max_retries: int = 3):
        self.client = RAGFlowClient(base_url=base_url, api_key=api_key, max_retries=max_retries)
        self.ingestion = IngestionEngine(self.client)
        self.retrieval = RetrievalEngine(self.client)

    def create_collection(self, name: str, config: Dict[str, Any]) -> str:
        try:
            if config:
                return self.ingestion.create_kb_for_experiment(name, config)
            return self.ingestion.create_reusable_kb(name)
        except RAGFlowAPIError as exc:
            raise BackendError(str(exc)) from exc

    def upload_documents(self, collection_id: str, folder_path) -> Dict[str, Any]:
        try:
            return self.ingestion.upload_folder_once(collection_id, folder_path)
        except RAGFlowAPIError as exc:
            raise BackendError(str(exc)) from exc

    def ingest_with_config(
        self,
        collection_id: str,
        config: Dict[str, Any],
        timeout: int = 600,
    ) -> Dict[str, Any]:
        try:
            return self.ingestion.reingest_with_config(collection_id, config, timeout=timeout)
        except RAGFlowAPIError as exc:
            raise BackendError(str(exc)) from exc

    def retrieve(
        self,
        query: str,
        collection_ids: List[str],
        top_k: int = 3,
        similarity_threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        try:
            chunks = self.retrieval.retrieve(
                question=query,
                dataset_ids=collection_ids,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )
        except RAGFlowAPIError as exc:
            raise BackendError(str(exc)) from exc

        for chunk in chunks:
            if "collection_id" not in chunk:
                chunk["collection_id"] = chunk.get("dataset_id", "")
        return chunks

    def delete_collection(self, collection_id: str) -> bool:
        return self.ingestion.cleanup_kb(collection_id)

    def get_collection_stats(self, collection_id: str) -> Dict[str, Any]:
        return self.ingestion.get_kb_stats(collection_id)

    def test_connection(self) -> bool:
        return self.client.test_connection()

    def get_embedding_model(self, collection_id: str) -> str:
        return self.client.get_embedding_model(collection_id)

    def get_collection_by_name(self, name: str) -> Optional[Dict[str, Any]]:
        return self.client.get_dataset_by_name(name)

    def get_document_ids(self, collection_id: str) -> List[str]:
        return self.ingestion.get_document_ids(collection_id)
