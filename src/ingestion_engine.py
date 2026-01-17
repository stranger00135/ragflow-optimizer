"""Ingestion engine for managing document ingestion into RAGFlow."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .ragflow_client import RAGFlowClient, RAGFlowAPIError


class IngestionEngine:
    """Manages document ingestion into RAGFlow knowledge bases."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".txt", ".md",
        ".xlsx", ".xls", ".csv",
        ".ppt", ".pptx",
        ".jpg", ".jpeg", ".png",
    }

    def __init__(self, client: RAGFlowClient):
        self.client = client
        # Track uploaded document IDs per KB for re-ingestion
        self._kb_documents: Dict[str, List[str]] = {}

    def _build_parser_config(self, preset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build parser_config from preset configuration."""
        parser_config = {}

        if "chunk_token_num" in preset_config:
            parser_config["chunk_token_num"] = preset_config["chunk_token_num"]

        if "overlap_token" in preset_config:
            parser_config["delimiter"] = "\n"

        if "auto_questions" in preset_config:
            parser_config["auto_questions"] = preset_config["auto_questions"]

        if "auto_keywords" in preset_config:
            parser_config["auto_keywords"] = preset_config["auto_keywords"]

        if "layout_recognize" in preset_config:
            parser_config["layout_recognize"] = preset_config["layout_recognize"]

        if preset_config.get("toc_enhance"):
            parser_config["html4excel"] = False

        return parser_config

    def create_kb_for_experiment(
        self,
        name: str,
        preset_config: Dict[str, Any],
    ) -> str:
        """Create a knowledge base for an experiment.

        Args:
            name: Name for the knowledge base
            preset_config: Configuration from preset

        Returns:
            Dataset ID
        """
        chunk_method = preset_config.get("chunk_method", "naive")
        parser_config = self._build_parser_config(preset_config)

        existing = self.client.get_dataset_by_name(name)
        if existing:
            self.client.delete_dataset(existing["id"])
            time.sleep(1)

        result = self.client.create_dataset(
            name=name,
            chunk_method=chunk_method,
            parser_config=parser_config,
        )

        return result.get("id", "")

    def ingest_folder(
        self,
        dataset_id: str,
        folder_path: Path,
        preset_config: Dict[str, Any],
        wait_for_completion: bool = True,
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """Ingest all supported files from a folder into a dataset.

        Args:
            dataset_id: Target dataset ID
            folder_path: Path to folder containing files
            preset_config: Parser configuration
            wait_for_completion: Whether to wait for parsing to complete
            timeout: Maximum time to wait for parsing

        Returns:
            Dict with document_ids, warnings, and chunk_count
        """
        document_ids = []
        warnings = []
        chunk_method = preset_config.get("chunk_method", "naive")
        parser_config = self._build_parser_config(preset_config)

        for item in folder_path.iterdir():
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                if item.name.startswith("."):
                    continue

                try:
                    doc_result = self.client.upload_document(dataset_id, str(item))
                    doc_id = doc_result.get("id")

                    if doc_id:
                        self.client.update_document(
                            dataset_id,
                            doc_id,
                            chunk_method=chunk_method,
                            parser_config=parser_config,
                        )
                        document_ids.append(doc_id)

                except RAGFlowAPIError as e:
                    warning = f"Failed to upload {item.name}: {e}"
                    warnings.append(warning)
                    print(f"Warning: {warning}")
                    continue

        if document_ids:
            self.client.parse_documents(dataset_id, document_ids)

            if wait_for_completion:
                statuses = self.client.wait_for_parsing(
                    dataset_id,
                    document_ids,
                    timeout=timeout,
                )
                # Collect warnings and progress messages
                for doc_id, status in statuses.items():
                    if status != "DONE":
                        warning = f"Document {doc_id} parsing status: {status}"
                        warnings.append(warning)
                        print(f"      Warning: {warning}")

                    # Get progress message for additional context
                    doc_status = self.client.get_document_status(dataset_id, doc_id)
                    progress_msg = doc_status.get("progress_msg", "")
                    if "No chunk built" in progress_msg:
                        warning = f"No chunks generated for document (parser may not support this format)"
                        if warning not in warnings:
                            warnings.append(warning)

        # Get final chunk count
        stats = self.get_kb_stats(dataset_id)
        chunk_count = stats.get("chunk_count", 0)

        if chunk_count == 0 and document_ids:
            warning = f"0 chunks produced - {chunk_method} parser may not be suitable for these documents"
            warnings.append(warning)

        return {
            "document_ids": document_ids,
            "warnings": warnings,
            "chunk_count": chunk_count,
        }

    def setup_distractor_kb(
        self,
        name: str,
        distractors_path: Path,
        preset_config: Dict[str, Any],
    ) -> str:
        """Set up the distractor knowledge base.

        Creates the KB if it doesn't exist, otherwise returns existing ID.
        Distractor KB is persistent and reused across runs.

        Args:
            name: Name for the distractor KB
            distractors_path: Path to distractor files
            preset_config: Parser configuration for distractors

        Returns:
            Dataset ID of the distractor KB
        """
        # Check if distractor KB already exists with chunks
        existing = self.client.get_dataset_by_name(name)
        if existing and existing.get("chunk_count", 0) > 0:
            print(f"  Reusing existing distractor KB: {name} ({existing.get('chunk_count')} chunks)")
            return existing["id"]

        # Delete if exists but has no chunks
        if existing:
            print(f"  Deleting empty distractor KB and recreating...")
            self.client.delete_dataset(existing["id"])
            time.sleep(1)

        # Create new distractor KB
        print(f"  Creating new distractor KB: {name}")
        dataset_id = self.create_kb_for_experiment(name, preset_config)

        result = self.ingest_folder(
            dataset_id,
            distractors_path,
            preset_config,
            wait_for_completion=True,
            timeout=900,
        )

        if result.get("warnings"):
            for warning in result["warnings"]:
                print(f"      Warning: {warning}")

        return dataset_id

    def cleanup_kb(self, dataset_id: str) -> bool:
        """Delete a knowledge base."""
        # Clean up tracked documents
        if dataset_id in self._kb_documents:
            del self._kb_documents[dataset_id]
        return self.client.delete_dataset(dataset_id)

    def get_kb_stats(self, dataset_id: str) -> Dict[str, Any]:
        """Get statistics for a knowledge base."""
        datasets = self.client.list_datasets()
        for ds in datasets:
            if ds.get("id") == dataset_id:
                return {
                    "chunk_count": ds.get("chunk_count", 0),
                    "document_count": ds.get("document_count", 0),
                    "token_num": ds.get("token_num", 0),
                }
        return {}

    def create_reusable_kb(self, name: str) -> str:
        """Create a reusable knowledge base for a folder.

        This KB will be reused across all experiments for a folder.
        Files are uploaded once, then re-ingested with different configs.

        Args:
            name: Name for the knowledge base

        Returns:
            Dataset ID
        """
        # Delete if exists
        existing = self.client.get_dataset_by_name(name)
        if existing:
            self.client.delete_dataset(existing["id"])
            time.sleep(1)

        # Create with default naive method - will be reconfigured per experiment
        result = self.client.create_dataset(
            name=name,
            chunk_method="naive",
            parser_config={},
        )

        dataset_id = result.get("id", "")
        self._kb_documents[dataset_id] = []
        return dataset_id

    def upload_folder_once(
        self,
        dataset_id: str,
        folder_path: Path,
    ) -> Dict[str, Any]:
        """Upload all files from a folder to a KB (once per folder).

        Files are uploaded but NOT parsed yet. Parsing happens in reingest_with_config.

        Args:
            dataset_id: Target dataset ID
            folder_path: Path to folder containing files

        Returns:
            Dict with document_ids, file_count, failed_files
        """
        document_ids = []
        failed_files = []
        total_files = 0

        for item in folder_path.iterdir():
            if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                if item.name.startswith("."):
                    continue

                total_files += 1

                try:
                    doc_result = self.client.upload_document(dataset_id, str(item))
                    doc_id = doc_result.get("id")

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

        # Track documents for this KB
        self._kb_documents[dataset_id] = document_ids

        return {
            "document_ids": document_ids,
            "total_files": total_files,
            "uploaded_files": len(document_ids),
            "failed_files": failed_files,
        }

    def reingest_with_config(
        self,
        dataset_id: str,
        preset_config: Dict[str, Any],
        timeout: int = 600,
    ) -> Dict[str, Any]:
        """Re-ingest documents with a new parser configuration.

        Updates parser config for all documents in the KB and re-triggers parsing.
        This is the v4 efficiency improvement - upload once, re-ingest many times.

        Args:
            dataset_id: Target dataset ID
            preset_config: New parser configuration
            timeout: Maximum time to wait for parsing

        Returns:
            Dict with ingestion_status, chunk_count, ingested_files, failed_files
        """
        document_ids = self._kb_documents.get(dataset_id, [])
        if not document_ids:
            # Get document IDs from API if not tracked
            docs = self.client.list_documents(dataset_id)
            document_ids = [d.get("id") for d in docs if d.get("id")]
            self._kb_documents[dataset_id] = document_ids

        if not document_ids:
            return {
                "ingestion_status": "failed",
                "total_files": 0,
                "ingested_files": 0,
                "failed_files": [],
                "chunk_count": 0,
                "error": "No documents found in KB"
            }

        chunk_method = preset_config.get("chunk_method", "naive")
        parser_config = self._build_parser_config(preset_config)

        # Update parser config for all documents
        failed_updates = []
        successful_ids = []

        for doc_id in document_ids:
            try:
                self.client.update_document(
                    dataset_id,
                    doc_id,
                    chunk_method=chunk_method,
                    parser_config=parser_config,
                )
                successful_ids.append(doc_id)
            except RAGFlowAPIError as e:
                # Get document name for error reporting
                doc_status = self.client.get_document_status(dataset_id, doc_id)
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
        self.client.parse_documents(dataset_id, successful_ids)

        # Wait for parsing completion
        statuses = self.client.wait_for_parsing(
            dataset_id,
            successful_ids,
            timeout=timeout,
        )

        # Check parsing results
        failed_parsing = []
        successful_parsing = 0

        for doc_id, status in statuses.items():
            doc_status = self.client.get_document_status(dataset_id, doc_id)
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
        stats = self.get_kb_stats(dataset_id)
        chunk_count = stats.get("chunk_count", 0)

        # Combine all failures
        all_failed = failed_updates + failed_parsing
        total_files = len(document_ids)
        ingested_files = successful_parsing

        # Determine ingestion status per v4 spec
        if ingested_files == 0 or chunk_count == 0:
            ingestion_status = "failed"
        elif len(all_failed) > total_files / 2:
            ingestion_status = "partial"  # >50% failed
        elif all_failed:
            ingestion_status = "partial"  # some failed
        else:
            ingestion_status = "success"

        return {
            "ingestion_status": ingestion_status,
            "total_files": total_files,
            "ingested_files": ingested_files,
            "failed_files": all_failed,
            "chunk_count": chunk_count,
        }

    def validate_ingestion(self, ingestion_result: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate ingestion result per v4 auto-disqualification rules.

        Args:
            ingestion_result: Result from reingest_with_config

        Returns:
            Tuple of (is_valid, disqualification_reason)
            is_valid=True means retrieval can proceed
            is_valid=False means experiment should be disqualified
        """
        status = ingestion_result.get("ingestion_status", "failed")
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

        # Passed validation - can proceed with retrieval
        return True, None

    def get_document_ids(self, dataset_id: str) -> List[str]:
        """Get tracked document IDs for a KB."""
        return self._kb_documents.get(dataset_id, [])
