"""
Orchestrator - Main controller for RAG ingestion parameter optimization.

This module coordinates the entire optimization process:
1. Generates evaluation questions for each document folder
2. Runs Phase 1 tournament across presets (general, manual, qa)
3. Runs Phase 2 fine-tuning on the winning preset
4. Produces summary reports with recommended configurations

Key concepts:
- Each folder in source_docs represents a document type
- KB reuse: Upload files once, re-ingest with different configs
- Tiebreaker: Same score -> faster ingestion time wins
"""

import atexit
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .backends.base import BackendError
from .backends.registry import get_backend
from .config_loader import Config
from .metrics import calculate_experiment_metrics, enrich_results_with_metrics
from .question_generator import QuestionGenerator
from .relevance_judge import RelevanceJudge


# Global orchestrator reference for signal handler
_active_orchestrator: Optional["Orchestrator"] = None


def _signal_handler(signum, frame):
    """Handle Ctrl+C and other termination signals."""
    global _active_orchestrator
    print("\n\nReceived termination signal. Cleaning up...")
    if _active_orchestrator:
        try:
            _active_orchestrator.emergency_cleanup()
        except Exception as e:
            print(f"Cleanup error: {e}")
    sys.exit(1)


class Orchestrator:
    """Main orchestrator for the RAG ingestion parameter optimization process."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".txt", ".md",
        ".xlsx", ".xls", ".csv",
        ".ppt", ".pptx",
    }

    # Cleanup registry file path
    CLEANUP_REGISTRY_FILE = ".claude/cleanup_registry.json"

    def __init__(self, config: Config):
        global _active_orchestrator
        _active_orchestrator = self

        self.config = config

        self.backend = get_backend(config)
        self.backend_name = config.backend_name

        self.question_generator = QuestionGenerator(
            api_key=config.llm_api_key,
            model=config.llm_model,
            provider=config.llm_provider,
            cache_path=config.questions_cache_path,
        )

        self.relevance_judge = RelevanceJudge(
            api_key=config.llm_api_key,
            model=config.llm_model,
            provider=config.llm_provider,
        )

        self.experiment_counter = 0
        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_output_path = config.output_path / self.run_id
        self.created_kbs: List[str] = []
        self.distractor_kb_id: Optional[str] = None
        # Use fixed name for distractor KB so it persists across runs
        self.distractor_kb_name = config.distractor_kb_name

        # v4: Track current folder's temp KB for reuse
        self._current_folder_kb_id: Optional[str] = None
        self._current_folder_path: Optional[str] = None
        self._current_embedding_model: Optional[str] = None

        # Setup signal handlers for graceful cleanup
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
        atexit.register(self._atexit_cleanup)

    def _atexit_cleanup(self):
        """Cleanup handler for normal exit."""
        # Don't do anything on normal exit - cleanup is explicit
        pass

    def _save_cleanup_registry(self) -> None:
        """Save created KB IDs to registry file for recovery."""
        registry_path = Path(self.CLEANUP_REGISTRY_FILE)
        registry_path.parent.mkdir(parents=True, exist_ok=True)

        registry = {
            "run_id": self.run_id,
            "created_at": datetime.now().isoformat(),
            "kb_ids": self.created_kbs,
            "distractor_kb_id": self.distractor_kb_id,
        }

        with open(registry_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2)

    def _clear_cleanup_registry(self) -> None:
        """Clear the cleanup registry after successful cleanup."""
        registry_path = Path(self.CLEANUP_REGISTRY_FILE)
        if registry_path.exists():
            registry_path.unlink()

    def emergency_cleanup(self) -> None:
        """Emergency cleanup on termination - delete all temp KBs."""
        print("Performing emergency cleanup...")
        deleted = 0

        # Clean up current folder KB
        if self._current_folder_kb_id:
            try:
                self.backend.delete_collection(self._current_folder_kb_id)
                deleted += 1
                print(f"  Deleted current folder KB")
            except Exception:
                pass

        # Clean up all created KBs
        for kb_id in self.created_kbs:
            try:
                self.backend.delete_collection(kb_id)
                deleted += 1
            except Exception:
                pass

        # Clear registry
        self._clear_cleanup_registry()
        print(f"  Deleted {deleted} knowledge base(s)")

    def _next_experiment_id(self) -> str:
        """Generate next experiment ID."""
        self.experiment_counter += 1
        return f"exp_{self.experiment_counter:03d}"

    def _get_folder_output_path(self, folder_path: str) -> Path:
        """Get output path for a folder's experiments."""
        safe_name = folder_path.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return self.run_output_path / safe_name

    def _has_excel_files(self, folder_path: Path) -> bool:
        """Check if folder contains Excel/CSV files."""
        for item in folder_path.iterdir():
            if item.is_file() and item.suffix.lower() in {".xlsx", ".xls", ".csv"}:
                return True
        return False

    def _get_applicable_presets(self, folder_path: Path) -> List[str]:
        """Get list of applicable presets for a folder."""
        presets = ["general", "manual", "qa"]
        if self._has_excel_files(folder_path):
            presets.append("table")
        return presets

    def _find_bottom_folders(self, source_path: Path) -> List[Tuple[Path, str]]:
        """Find all bottom-level folders containing files."""
        folders = []

        def scan(path: Path, relative: str = ""):
            has_files = False

            for item in path.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    has_files = True
                elif item.is_dir():
                    sub_relative = f"{relative}/{item.name}" if relative else item.name
                    scan(item, sub_relative)

            if has_files:
                folders.append((path, relative if relative else path.name))

        scan(source_path)
        return folders

    def test_api_connectivity(self) -> bool:
        """Test that all required APIs are working."""
        print(f"Testing {self.backend_name} backend connectivity...")

        try:
            connected = self.backend.test_connection()
        except BackendError as exc:
            raise BackendError(f"Cannot connect to backend: {exc}") from exc

        if not connected:
            raise BackendError("Cannot connect to backend. Check configuration and credentials.")

        print("  Backend API: OK")

        try:
            self.question_generator.client.chat.completions.create(
                model=self.question_generator.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            print("  LLM API: OK")
        except Exception as e:
            raise BackendError(f"Cannot connect to LLM API: {e}")

        return True

    def setup_distractor_kb(self) -> Optional[str]:
        """Set up the distractor collection."""
        print("Setting up distractor collection...")

        if not self.config.distractors_path.exists():
            print("  Distractors path not found; skipping distractor setup.")
            self.distractor_kb_id = None
            return None

        preset_name = self.config.distractor_ingest_preset
        preset_config = self.config.get_preset(preset_name)

        existing = self.backend.get_collection_by_name(self.distractor_kb_name)
        if existing:
            existing_id = existing.get("id") or existing.get("collection_id")
            if existing_id:
                stats = self.backend.get_collection_stats(existing_id)
                if stats.get("chunk_count", 0) > 0:
                    print(f"  Reusing existing distractor collection: {self.distractor_kb_name}")
                    self.distractor_kb_id = existing_id
                    return self.distractor_kb_id
                self.backend.delete_collection(existing_id)

        self.distractor_kb_id = self.backend.create_collection(self.distractor_kb_name, preset_config)

        upload_result = self.backend.upload_documents(self.distractor_kb_id, self.config.distractors_path)
        if upload_result.get("uploaded_files", 0) == 0:
            print("  Warning: No distractor files uploaded.")
        else:
            print(
                f"  Uploaded {upload_result.get('uploaded_files', 0)}/"
                f"{upload_result.get('total_files', 0)} distractor files"
            )

        self.backend.ingest_with_config(
            collection_id=self.distractor_kb_id,
            config=preset_config,
            timeout=900,
        )

        stats = self.backend.get_collection_stats(self.distractor_kb_id)
        print(f"  Distractor collection ready: {stats.get('chunk_count', 0)} chunks")

        return self.distractor_kb_id

    def _retrieve_for_evaluation(
        self,
        questions: List[Dict[str, Any]],
        target_kb_id: str,
        distractor_kb_id: Optional[str],
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks for evaluation questions using the configured backend."""
        dataset_ids = [target_kb_id]
        if distractor_kb_id:
            dataset_ids.append(distractor_kb_id)

        results = []
        for q in questions:
            question_text = q.get("question", "")
            question_id = q.get("question_id", "")

            chunks = self.backend.retrieve(
                query=question_text,
                collection_ids=dataset_ids,
                top_k=self.config.top_k,
                similarity_threshold=self.config.similarity_threshold,
            )

            for chunk in chunks:
                dataset_id = chunk.get("dataset_id") or chunk.get("collection_id") or ""
                chunk["dataset_id"] = dataset_id
                chunk["collection_id"] = chunk.get("collection_id") or dataset_id
                chunk["from_distractor"] = bool(distractor_kb_id and dataset_id == distractor_kb_id)

            results.append({
                "question_id": question_id,
                "question": question_text,
                "chunks_retrieved": chunks,
            })

        return results

    def run_single_experiment_v4(
        self,
        dataset_id: str,
        relative_path: str,
        preset_name: str,
        preset_config: Dict[str, Any],
        questions: List[Dict],
        phase: int,
        is_baseline: bool = False,
    ) -> Dict[str, Any]:
        """Run a single experiment using v4 KB reuse strategy.

        v4 improvement: Uses existing KB with uploaded files, just re-ingests
        with new parser config instead of creating new KB each time.

        Args:
            dataset_id: Existing KB with uploaded files
            relative_path: Relative path for naming
            preset_name: Name of the preset being used
            preset_config: Configuration for this experiment
            questions: List of evaluation questions
            phase: 1 or 2
            is_baseline: Whether this is the baseline for Phase 2

        Returns:
            Experiment result dictionary with v4 fields (status, ingestion_details)
        """
        experiment_id = self._next_experiment_id()
        start_time = time.time()

        print(f"    Running {experiment_id}: {preset_name}")

        try:
            # v4: Re-ingest with new config instead of creating new KB
            ingestion_result = self.backend.ingest_with_config(
                collection_id=dataset_id,
                config=preset_config,
                timeout=600,
            )

            ingestion_time = time.time() - start_time

            # v4: Build ingestion_details per spec
            ingestion_details = {
                "total_files": ingestion_result.get("total_files", 0),
                "ingested_files": ingestion_result.get("ingested_files", 0),
                "failed_files": ingestion_result.get("failed_files", []),
                "chunk_count": ingestion_result.get("chunk_count", 0),
                "ingestion_time_seconds": round(ingestion_time, 2),
            }

            # v4: Validate ingestion and check for disqualification
            is_valid, disqualification_reason = self.backend.validate_ingestion(ingestion_result)

            if not is_valid:
                # Experiment disqualified - skip retrieval
                print(f"      DISQUALIFIED: {disqualification_reason}")

                experiment_result = {
                    "experiment_id": experiment_id,
                    "folder_path": relative_path,
                    "phase": phase,
                    "preset": preset_name,
                    "status": "disqualified",
                    "disqualification_reason": disqualification_reason,
                    "is_baseline": is_baseline,
                    "config": preset_config,
                    "ingestion_details": ingestion_details,
                    "embedding_model": self._current_embedding_model,
                    "metrics": None,
                    "questions": None,
                }

                # Save experiment file
                folder_output = self._get_folder_output_path(relative_path)
                folder_output.mkdir(parents=True, exist_ok=True)
                with open(folder_output / f"{experiment_id}.json", "w", encoding="utf-8") as f:
                    json.dump(experiment_result, f, ensure_ascii=False, indent=2)

                return experiment_result

            # Ingestion valid - proceed with retrieval
            chunk_count = ingestion_details["chunk_count"]

            retrieval_results = self._retrieve_for_evaluation(
                questions=questions,
                target_kb_id=dataset_id,
                distractor_kb_id=self.distractor_kb_id,
            )

            judged_results = self.relevance_judge.judge_retrieval_results(
                retrieval_results
            )

            enriched_results = enrich_results_with_metrics(
                judged_results,
                top_k=self.config.top_k,
            )

            metrics = calculate_experiment_metrics(
                judged_results,
                top_k=self.config.top_k,
            )

            experiment_result = {
                "experiment_id": experiment_id,
                "folder_path": relative_path,
                "phase": phase,
                "preset": preset_name,
                "status": "completed",
                "is_baseline": is_baseline,
                "config": preset_config,
                "ingestion_details": ingestion_details,
                "embedding_model": self._current_embedding_model,
                "metrics": metrics,
                "questions": enriched_results,
            }

            folder_output = self._get_folder_output_path(relative_path)
            folder_output.mkdir(parents=True, exist_ok=True)

            with open(folder_output / f"{experiment_id}.json", "w", encoding="utf-8") as f:
                json.dump(experiment_result, f, ensure_ascii=False, indent=2)

            print(f"      Score: {metrics['combined_score']:.4f} (chunks: {chunk_count})")

            return experiment_result

        except BackendError as e:
            print(f"      ERROR: {e}")
            return {
                "experiment_id": experiment_id,
                "folder_path": relative_path,
                "phase": phase,
                "preset": preset_name,
                "status": "error",
                "config": preset_config,
                "error": str(e),
                "embedding_model": self._current_embedding_model,
                "ingestion_details": None,
                "metrics": None,
            }

    def run_phase1_tournament_v4(
        self,
        dataset_id: str,
        folder_path: Path,
        relative_path: str,
        questions: List[Dict],
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]], List[Dict]]:
        """Run Phase 1 preset tournament for a folder using v4 KB reuse.

        Returns:
            Tuple of (winning_preset_name, winning_config, all_experiment_results)
            Returns (None, None, results) if all presets disqualified
        """
        print(f"  Phase 1: Preset Selection for {relative_path}")

        applicable_presets = self._get_applicable_presets(folder_path)
        results = []
        best_score = -1.0
        best_preset = None
        best_config = None
        best_ingestion_time = float('inf')

        for preset_name in applicable_presets:
            preset_config = self.config.get_preset(preset_name)

            result = self.run_single_experiment_v4(
                dataset_id=dataset_id,
                relative_path=relative_path,
                preset_name=preset_name,
                preset_config=preset_config,
                questions=questions,
                phase=1,
            )
            results.append(result)

            # Only consider completed experiments for winner selection
            if result.get("status") != "completed":
                continue

            metrics = result.get("metrics") or {}
            score = metrics.get("combined_score", 0.0)
            ingestion_time = (result.get("ingestion_details") or {}).get("ingestion_time_seconds", float('inf'))

            # Select winner: highest score, tie-breaker is lower ingestion time
            if score > best_score or (score == best_score and ingestion_time < best_ingestion_time):
                best_score = score
                best_preset = preset_name
                best_config = preset_config
                best_ingestion_time = ingestion_time

        if best_preset:
            print(f"  Phase 1 Winner: {best_preset} (score: {best_score:.4f})")
        else:
            print(f"  Phase 1: ALL PRESETS DISQUALIFIED - skipping Phase 2")

        return best_preset, best_config, results

    def run_phase2_finetuning_v4(
        self,
        dataset_id: str,
        relative_path: str,
        questions: List[Dict],
        winning_preset: str,
        baseline_config: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """Run Phase 2 parameter fine-tuning using v4 KB reuse.

        Returns:
            Tuple of (best_config, all_experiment_results)
        """
        print(f"  Phase 2: Fine-Tuning {winning_preset}")

        optimization_space = self.config.get_optimization_space(winning_preset)

        if not optimization_space:
            print("    No optimization space defined, using baseline")
            return baseline_config, []

        results = []
        current_config = dict(baseline_config)
        best_config = dict(baseline_config)
        best_score = 0.0
        best_ingestion_time = float('inf')
        best_exp_id = None

        # Run baseline experiment
        baseline_result = self.run_single_experiment_v4(
            dataset_id=dataset_id,
            relative_path=relative_path,
            preset_name=winning_preset,
            preset_config=current_config,
            questions=questions,
            phase=2,
            is_baseline=True,
        )
        results.append(baseline_result)

        if baseline_result.get("status") == "completed":
            metrics = baseline_result.get("metrics") or {}
            best_score = metrics.get("combined_score", 0.0)
            best_ingestion_time = (baseline_result.get("ingestion_details") or {}).get("ingestion_time_seconds", float('inf'))
            best_exp_id = baseline_result.get("experiment_id")

        for param_spec in optimization_space:
            param_name = param_spec["name"]
            param_values = param_spec["values"]

            print(f"    Optimizing {param_name}...")

            best_param_value = current_config.get(param_name)
            best_param_score = best_score
            best_param_time = best_ingestion_time
            best_param_exp_id = best_exp_id

            for value in param_values:
                if value == current_config.get(param_name):
                    continue

                test_config = dict(current_config)
                test_config[param_name] = value

                result = self.run_single_experiment_v4(
                    dataset_id=dataset_id,
                    relative_path=relative_path,
                    preset_name=winning_preset,
                    preset_config=test_config,
                    questions=questions,
                    phase=2,
                )
                results.append(result)

                # Only consider completed experiments
                if result.get("status") != "completed":
                    continue

                metrics = result.get("metrics") or {}
                score = metrics.get("combined_score", 0.0)
                ingestion_time = (result.get("ingestion_details") or {}).get("ingestion_time_seconds", float('inf'))

                # Tie-breaker: lower ingestion time (faster is better)
                if score > best_param_score or (score == best_param_score and ingestion_time < best_param_time):
                    best_param_score = score
                    best_param_value = value
                    best_param_time = ingestion_time
                    best_param_exp_id = result.get("experiment_id")

            current_config[param_name] = best_param_value
            if best_param_score > best_score or (best_param_score == best_score and best_param_time < best_ingestion_time):
                best_score = best_param_score
                best_config = dict(current_config)
                best_ingestion_time = best_param_time
                best_exp_id = best_param_exp_id

        print(f"  Phase 2 Best Score: {best_score:.4f} ({best_exp_id})")
        return best_config, results, best_exp_id

    def run_folder_optimization_v4(
        self,
        folder_path: Path,
        relative_path: str,
    ) -> Dict[str, Any]:
        """Run full optimization for a single folder using v4 KB reuse.

        v4 improvements:
        - Create single temp KB per folder
        - Upload files once
        - Re-ingest with different configs for each experiment
        - Clean up KB after folder completes

        Returns:
            Dict with folder results including winning config
        """
        print(f"\nOptimizing: {relative_path}")

        # Generate/load questions
        questions_data = self.question_generator.generate_questions(
            folder_path=folder_path,
            relative_path=relative_path,
            num_questions=self.config.questions_per_folder,
            use_cache=True,
        )

        questions = questions_data.get("questions", [])
        if not questions:
            print("  No questions generated, skipping folder")
            return {
                "folder_path": relative_path,
                "status": "skipped",
                "reason": "No questions generated",
            }

        print(f"  Using {len(questions)} questions")

        # v4: Create single reusable KB for this folder
        import hashlib
        folder_hash = hashlib.md5(relative_path.encode()).hexdigest()[:8]
        kb_name = f"eval_temp_{folder_hash}"

        dataset_id = self.backend.create_collection(kb_name, {})
        self._current_folder_kb_id = dataset_id
        self._current_folder_path = relative_path
        self._current_embedding_model = self.backend.get_embedding_model(dataset_id)
        self.created_kbs.append(dataset_id)
        self._save_cleanup_registry()

        # v4: Upload files once
        print(f"  Uploading files to temp KB...")
        upload_result = self.backend.upload_documents(
            collection_id=dataset_id,
            folder_path=folder_path,
        )

        if upload_result["uploaded_files"] == 0:
            print(f"  No files uploaded, skipping folder")
            # Clean up KB
            self.backend.delete_collection(dataset_id)
            self.created_kbs.remove(dataset_id)
            self._current_folder_kb_id = None
            return {
                "folder_path": relative_path,
                "status": "skipped",
                "reason": f"No files uploaded ({upload_result['total_files']} files found, all failed)",
            }

        print(f"  Uploaded {upload_result['uploaded_files']}/{upload_result['total_files']} files")

        try:
            # Run Phase 1 tournament
            winning_preset, winning_config, phase1_results = self.run_phase1_tournament_v4(
                dataset_id=dataset_id,
                folder_path=folder_path,
                relative_path=relative_path,
                questions=questions,
            )

            # Check if all presets disqualified
            if winning_preset is None:
                print(f"  All presets disqualified, skipping Phase 2")
                result = {
                    "folder_path": relative_path,
                    "status": "all_disqualified",
                    "reason": "All Phase 1 presets failed ingestion validation",
                    "phase1_experiments": len(phase1_results),
                    "phase2_experiments": 0,
                    "questions_used": len(questions),
                }
            else:
                # Run Phase 2 fine-tuning
                best_config, phase2_results, best_exp_id = self.run_phase2_finetuning_v4(
                    dataset_id=dataset_id,
                    relative_path=relative_path,
                    questions=questions,
                    winning_preset=winning_preset,
                    baseline_config=winning_config,
                )

                result = {
                    "folder_path": relative_path,
                    "status": "completed",
                    "winning_preset": winning_preset,
                    "recommended_config": best_config,
                    "best_experiment_id": best_exp_id,
                    "phase1_experiments": len(phase1_results),
                    "phase2_experiments": len(phase2_results),
                    "questions_used": len(questions),
                }

        finally:
            # v4: Clean up folder's temp KB after all experiments
            print(f"  Cleaning up folder temp KB...")
            self.backend.delete_collection(dataset_id)
            if dataset_id in self.created_kbs:
                self.created_kbs.remove(dataset_id)
            self._current_folder_kb_id = None
            self._save_cleanup_registry()

        return result

    def run_full_optimization(
        self,
        target_folder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run optimization for all folders or a specific folder.

        Uses v4 KB reuse strategy for efficiency.

        Args:
            target_folder: Optional specific folder path to process

        Returns:
            Summary of optimization results
        """
        self.run_output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"RAG Ingestion Parameter Optimizer (v4)")
        print(f"Run ID: {self.run_id}")
        print(f"{'='*60}\n")

        self.test_api_connectivity()

        self.setup_distractor_kb()

        if target_folder:
            source_resolved = self.config.source_docs_path.resolve()
            folder_path = (self.config.source_docs_path / target_folder).resolve()
            try:
                folder_path.relative_to(source_resolved)
            except ValueError:
                raise ValueError(
                    "Folder path must be inside source_docs (path traversal not allowed)."
                )
            if not folder_path.exists() or not folder_path.is_dir():
                raise ValueError(f"Folder not found: {target_folder}")
            folders = [(folder_path, target_folder)]
        else:
            folders = self._find_bottom_folders(self.config.source_docs_path)

        print(f"\nFound {len(folders)} folder(s) to process")

        results = []
        for folder_path, relative_path in folders:
            try:
                # Use v4 method with KB reuse
                result = self.run_folder_optimization_v4(folder_path, relative_path)
                results.append(result)
            except Exception as e:
                print(f"  ERROR processing {relative_path}: {e}")
                results.append({
                    "folder_path": relative_path,
                    "status": "error",
                    "error": str(e),
                })
                # Ensure cleanup on error
                if self._current_folder_kb_id:
                    try:
                        self.backend.delete_collection(self._current_folder_kb_id)
                        if self._current_folder_kb_id in self.created_kbs:
                            self.created_kbs.remove(self._current_folder_kb_id)
                    except Exception:
                        pass
                    self._current_folder_kb_id = None

        summary = {
            "run_id": self.run_id,
            "total_folders": len(folders),
            "completed": sum(1 for r in results if r.get("status") == "completed"),
            "errors": sum(1 for r in results if r.get("status") == "error"),
            "skipped": sum(1 for r in results if r.get("status") == "skipped"),
            "all_disqualified": sum(1 for r in results if r.get("status") == "all_disqualified"),
            "folder_results": results,
        }

        with open(self.run_output_path / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._generate_markdown_report_v4(results)

        # Clear cleanup registry on successful completion
        self._clear_cleanup_registry()

        return summary

    def _generate_markdown_report(self, results: List[Dict]) -> None:
        """Generate markdown summary report with experiment tables."""
        report_lines = [
            f"# Optimization Report",
            f"Run ID: {self.run_id}",
            f"",
        ]

        for result in results:
            folder = result.get("folder_path", "Unknown")
            status = result.get("status", "unknown")

            report_lines.append(f"## {folder}")
            report_lines.append("")

            if status == "completed":
                # Load experiment files for this folder
                folder_output = self._get_folder_output_path(folder)
                experiments = self._load_folder_experiments(folder_output)

                phase1_exps = [e for e in experiments if e.get("phase") == 1]
                phase2_exps = [e for e in experiments if e.get("phase") == 2]

                # Phase 1 table
                if phase1_exps:
                    report_lines.append("### Phase 1: Preset Selection")
                    report_lines.append("")
                    report_lines.append("| Exp ID | Preset | chunk_method | chunk_token | overlap | auto_q | auto_kw | Fail Rate | P@3 | MRR | Score |")
                    report_lines.append("|--------|--------|--------------|-------------|---------|--------|---------|-----------|-----|-----|-------|")

                    winner_exp = None
                    winner_score = -1
                    for exp in sorted(phase1_exps, key=lambda x: x.get("experiment_id", "")):
                        cfg = exp.get("config", {})
                        m = exp.get("metrics", {})
                        exp_id = exp.get("experiment_id", "")
                        preset = exp.get("preset", "")
                        score = m.get("combined_score", 0)

                        if score > winner_score:
                            winner_score = score
                            winner_exp = exp_id

                        report_lines.append(
                            f"| {exp_id} | {preset} | {cfg.get('chunk_method', '-')} | "
                            f"{cfg.get('chunk_token_num', '-')} | {cfg.get('overlap_token', '-')} | "
                            f"{cfg.get('auto_questions', '-')} | {cfg.get('auto_keywords', '-')} | "
                            f"{m.get('fail_rate', 0):.2f} | {m.get('precision_at_k', 0):.2f} | "
                            f"{m.get('mrr', 0):.2f} | {score:.4f} |"
                        )

                    report_lines.append("")
                    report_lines.append(f"**Phase 1 Winner: {result.get('winning_preset', 'N/A')} ({winner_exp})**")
                    report_lines.append("")

                # Phase 2 table
                if phase2_exps:
                    report_lines.append(f"### Phase 2: Fine-Tuning ({result.get('winning_preset', 'N/A')})")
                    report_lines.append("")
                    report_lines.append("| Exp ID | Baseline | chunk_token | overlap | auto_q | auto_kw | Fail Rate | P@3 | MRR | Score |")
                    report_lines.append("|--------|----------|-------------|---------|--------|---------|-----------|-----|-----|-------|")

                    best_exp = None
                    best_score = -1
                    for exp in sorted(phase2_exps, key=lambda x: x.get("experiment_id", "")):
                        cfg = exp.get("config", {})
                        m = exp.get("metrics", {})
                        exp_id = exp.get("experiment_id", "")
                        is_baseline = "✓" if exp.get("is_baseline") else ""
                        score = m.get("combined_score", 0)

                        if score > best_score:
                            best_score = score
                            best_exp = exp_id

                        report_lines.append(
                            f"| {exp_id} | {is_baseline} | {cfg.get('chunk_token_num', '-')} | "
                            f"{cfg.get('overlap_token', '-')} | {cfg.get('auto_questions', '-')} | "
                            f"{cfg.get('auto_keywords', '-')} | {m.get('fail_rate', 0):.2f} | "
                            f"{m.get('precision_at_k', 0):.2f} | {m.get('mrr', 0):.2f} | {score:.4f} |"
                        )

                    report_lines.append("")
                    report_lines.append(f"**Best Configuration: {best_exp} (Score: {best_score:.4f})**")
                    report_lines.append("")

                # Recommended config
                report_lines.append("### Recommended Configuration")
                report_lines.append("")
                report_lines.append("```json")
                report_lines.append(json.dumps(result.get("recommended_config", {}), indent=2))
                report_lines.append("```")

            elif status == "error":
                report_lines.append(f"**Status**: Error")
                report_lines.append(f"**Error**: {result.get('error', 'Unknown')}")
            else:
                report_lines.append(f"**Status**: {status}")
                report_lines.append(f"**Reason**: {result.get('reason', 'Unknown')}")

            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        with open(self.run_output_path / "summary_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    def _load_folder_experiments(self, folder_path: Path) -> List[Dict]:
        """Load all experiment JSON files from a folder."""
        experiments = []
        if not folder_path.exists():
            return experiments

        for json_file in sorted(folder_path.glob("exp_*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    experiments.append(json.load(f))
            except Exception:
                continue
        return experiments

    def _generate_markdown_report_v4(self, results: List[Dict]) -> None:
        """Generate v4 markdown summary report with ingestion status columns."""
        report_lines = [
            f"# Optimization Report",
            f"Run ID: {self.run_id}",
            f"",
        ]

        for result in results:
            folder = result.get("folder_path", "Unknown")
            status = result.get("status", "unknown")

            report_lines.append(f"## {folder}")
            report_lines.append("")

            if status in ("completed", "all_disqualified"):
                # Load experiment files for this folder
                folder_output = self._get_folder_output_path(folder)
                experiments = self._load_folder_experiments(folder_output)

                # Show embedding model if available
                if experiments and experiments[0].get("embedding_model"):
                    report_lines.append(f"**Embedding Model**: {experiments[0].get('embedding_model')}")
                    report_lines.append("")

                phase1_exps = [e for e in experiments if e.get("phase") == 1]
                phase2_exps = [e for e in experiments if e.get("phase") == 2]

                # Phase 1 table with v4 columns (Status, Files, Chunks, Time)
                if phase1_exps:
                    report_lines.append("### Phase 1: Preset Selection")
                    report_lines.append("")
                    report_lines.append("| Exp ID | Preset | Status | Files | Chunks | Time(s) | Fail Rate | P@3 | MRR | Score |")
                    report_lines.append("|--------|--------|--------|-------|--------|---------|-----------|-----|-----|-------|")

                    winner_exp = None
                    winner_score = -1
                    for exp in sorted(phase1_exps, key=lambda x: x.get("experiment_id", "")):
                        exp_id = exp.get("experiment_id", "")
                        preset = exp.get("preset", "")
                        exp_status = exp.get("status", "unknown")

                        # Get ingestion details
                        ing = exp.get("ingestion_details") or {}
                        total_files = ing.get("total_files", "-")
                        ingested_files = ing.get("ingested_files", "-")
                        chunk_count = ing.get("chunk_count", "-")
                        ingestion_time = ing.get("ingestion_time_seconds", "-")
                        files_str = f"{ingested_files}/{total_files}" if isinstance(ingested_files, int) else "-"
                        time_str = f"{ingestion_time:.1f}" if isinstance(ingestion_time, (int, float)) else "-"

                        if exp_status == "completed":
                            m = exp.get("metrics") or {}
                            score = m.get("combined_score", 0)

                            if score > winner_score:
                                winner_score = score
                                winner_exp = exp_id

                            report_lines.append(
                                f"| {exp_id} | {preset} | completed | {files_str} | {chunk_count} | {time_str} | "
                                f"{m.get('fail_rate', 0):.2f} | {m.get('precision_at_k', 0):.2f} | "
                                f"{m.get('mrr', 0):.2f} | {score:.4f} |"
                            )
                        else:
                            # Disqualified or error
                            report_lines.append(
                                f"| {exp_id} | {preset} | {exp_status} | {files_str} | {chunk_count} | {time_str} | "
                                f"- | - | - | - |"
                            )

                    report_lines.append("")

                    if result.get("winning_preset"):
                        report_lines.append(f"**Phase 1 Winner: {result.get('winning_preset')} ({winner_exp})**")
                    else:
                        report_lines.append("**Phase 1: All presets disqualified**")

                    # Add notes for disqualified experiments
                    disqualified = [e for e in phase1_exps if e.get("status") == "disqualified"]
                    if disqualified:
                        report_lines.append("")
                        for exp in disqualified:
                            reason = exp.get("disqualification_reason", "Unknown reason")
                            report_lines.append(f"**Note: {exp.get('preset')} preset disqualified - {reason}**")

                    report_lines.append("")

                # Phase 2 table with v4 columns (including Time, overlap)
                if phase2_exps:
                    report_lines.append(f"### Phase 2: Fine-Tuning ({result.get('winning_preset', 'N/A')})")
                    report_lines.append("")
                    report_lines.append("| Exp ID | Baseline | Status | Files | Chunks | Time(s) | chunk_token | overlap | auto_q | auto_kw | Fail Rate | P@3 | MRR | Score |")
                    report_lines.append("|--------|----------|--------|-------|--------|---------|-------------|---------|--------|---------|-----------|-----|-----|-------|")

                    best_exp = None
                    best_score = -1
                    for exp in sorted(phase2_exps, key=lambda x: x.get("experiment_id", "")):
                        cfg = exp.get("config", {})
                        exp_id = exp.get("experiment_id", "")
                        is_baseline = "✓" if exp.get("is_baseline") else ""
                        exp_status = exp.get("status", "unknown")

                        # Get ingestion details
                        ing = exp.get("ingestion_details") or {}
                        total_files = ing.get("total_files", "-")
                        ingested_files = ing.get("ingested_files", "-")
                        chunk_count = ing.get("chunk_count", "-")
                        ingestion_time = ing.get("ingestion_time_seconds", "-")
                        files_str = f"{ingested_files}/{total_files}" if isinstance(ingested_files, int) else "-"
                        time_str = f"{ingestion_time:.1f}" if isinstance(ingestion_time, (int, float)) else "-"

                        if exp_status == "completed":
                            m = exp.get("metrics") or {}
                            score = m.get("combined_score", 0)

                            if score > best_score:
                                best_score = score
                                best_exp = exp_id

                            report_lines.append(
                                f"| {exp_id} | {is_baseline} | completed | {files_str} | {chunk_count} | {time_str} | "
                                f"{cfg.get('chunk_token_num', '-')} | {cfg.get('overlap_token', '-')} | "
                                f"{cfg.get('auto_questions', '-')} | {cfg.get('auto_keywords', '-')} | "
                                f"{m.get('fail_rate', 0):.2f} | {m.get('precision_at_k', 0):.2f} | "
                                f"{m.get('mrr', 0):.2f} | {score:.4f} |"
                            )
                        else:
                            report_lines.append(
                                f"| {exp_id} | {is_baseline} | {exp_status} | {files_str} | {chunk_count} | {time_str} | "
                                f"{cfg.get('chunk_token_num', '-')} | {cfg.get('overlap_token', '-')} | "
                                f"{cfg.get('auto_questions', '-')} | {cfg.get('auto_keywords', '-')} | "
                                f"- | - | - | - |"
                            )

                    report_lines.append("")
                    # Use best_experiment_id from result if available (uses time tiebreaker)
                    actual_best_exp = result.get("best_experiment_id", best_exp)
                    report_lines.append(f"**Best Configuration: {actual_best_exp} (Score: {best_score:.4f})**")
                    report_lines.append("")

                # Recommended config (now guaranteed to match best_experiment_id)
                if result.get("recommended_config"):
                    report_lines.append("### Recommended Configuration")
                    report_lines.append("")
                    report_lines.append("```json")
                    report_lines.append(json.dumps(result.get("recommended_config", {}), indent=2))
                    report_lines.append("```")

            elif status == "error":
                report_lines.append(f"**Status**: Error")
                report_lines.append(f"**Error**: {result.get('error', 'Unknown')}")
            else:
                report_lines.append(f"**Status**: {status}")
                report_lines.append(f"**Reason**: {result.get('reason', 'Unknown')}")

            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")

        with open(self.run_output_path / "summary_report.md", "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))

    def cleanup(self, include_distractor: bool = False) -> None:
        """Clean up created knowledge bases."""
        print("\nCleaning up temporary knowledge bases...")

        for kb_id in self.created_kbs:
            try:
                self.backend.delete_collection(kb_id)
            except Exception:
                pass

        if include_distractor and self.distractor_kb_id:
            try:
                self.backend.delete_collection(self.distractor_kb_id)
            except Exception:
                pass

        self.created_kbs = []
        self._clear_cleanup_registry()
        print("Cleanup complete")
