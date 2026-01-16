"""Retrieval engine for querying RAGFlow knowledge bases."""

from typing import Any, Dict, List

from .ragflow_client import RAGFlowClient


class RetrievalEngine:
    """Manages retrieval operations against RAGFlow knowledge bases."""

    def __init__(self, client: RAGFlowClient):
        self.client = client

    def retrieve(
        self,
        question: str,
        dataset_ids: List[str],
        top_k: int = 3,
        similarity_threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant chunks for a question.

        Args:
            question: The question to search for
            dataset_ids: List of dataset IDs to search
            top_k: Number of top chunks to retrieve
            similarity_threshold: Minimum similarity score

        Returns:
            List of chunk dictionaries with content and metadata
        """
        chunks = self.client.retrieve_chunks(
            question=question,
            dataset_ids=dataset_ids,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

        results = []
        for i, chunk in enumerate(chunks[:top_k]):
            results.append({
                "rank": i + 1,
                "chunk_id": chunk.get("id", ""),
                "content": chunk.get("content", ""),
                "similarity": chunk.get("similarity", 0.0),
                "document_id": chunk.get("document_id", ""),
                "document_name": chunk.get("docnm_kwd", ""),
                "dataset_id": chunk.get("dataset_id", ""),
                "keywords": chunk.get("important_keywords", []),
            })

        return results

    def retrieve_for_evaluation(
        self,
        questions: List[Dict[str, str]],
        target_kb_id: str,
        distractor_kb_id: str,
        top_k: int = 3,
        similarity_threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks for a list of evaluation questions.

        Queries both target and distractor KBs simultaneously.

        Args:
            questions: List of question dicts with question_id and question
            target_kb_id: ID of the target knowledge base
            distractor_kb_id: ID of the distractor knowledge base
            top_k: Number of chunks to retrieve per question
            similarity_threshold: Minimum similarity score

        Returns:
            List of retrieval results, one per question
        """
        dataset_ids = [target_kb_id, distractor_kb_id]
        results = []

        for q in questions:
            question_text = q.get("question", "")
            question_id = q.get("question_id", "")

            chunks = self.retrieve(
                question=question_text,
                dataset_ids=dataset_ids,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
            )

            for chunk in chunks:
                chunk["from_distractor"] = chunk.get("dataset_id") == distractor_kb_id

            results.append({
                "question_id": question_id,
                "question": question_text,
                "chunks_retrieved": chunks,
            })

        return results
