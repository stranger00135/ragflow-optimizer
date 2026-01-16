"""Orchestrator for RAG ingestion parameter optimization."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config_loader import Config
from .ingestion_engine import IngestionEngine
from .metrics import calculate_experiment_metrics, enrich_results_with_metrics
from .question_generator import QuestionGenerator
from .ragflow_client import RAGFlowClient, RAGFlowAPIError
from .relevance_judge import RelevanceJudge
from .retrieval_engine import RetrievalEngine


class Orchestrator:
    """Main orchestrator for the RAG ingestion parameter optimization process."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".txt", ".md",
        ".xlsx", ".xls", ".csv",
        ".ppt", ".pptx",
    }

    def __init__(self, config: Config):
        self.config = config

        self.client = RAGFlowClient(
            base_url=config.ragflow_base_url,
            api_key=config.ragflow_api_key,
            max_retries=config.max_retries,
        )

        self.ingestion_engine = IngestionEngine(self.client)
        self.retrieval_engine = RetrievalEngine(self.client)

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
        print("Testing RAGFlow API connectivity...")

        if not self.client.test_connection():
            raise RAGFlowAPIError("Cannot connect to RAGFlow API. Check base_url and api_key.")

        print("  RAGFlow API: OK")

        try:
            self.question_generator.client.chat.completions.create(
                model=self.question_generator.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            print("  LLM API: OK")
        except Exception as e:
            raise RAGFlowAPIError(f"Cannot connect to LLM API: {e}")

        return True

    def setup_distractor_kb(self) -> str:
        """Set up the distractor knowledge base."""
        print("Setting up distractor knowledge base...")

        preset_config = self.config.get_preset("general")

        self.distractor_kb_id = self.ingestion_engine.setup_distractor_kb(
            name=self.distractor_kb_name,
            distractors_path=self.config.distractors_path,
            preset_config=preset_config,
        )

        stats = self.ingestion_engine.get_kb_stats(self.distractor_kb_id)
        print(f"  Distractor KB ready: {stats.get('chunk_count', 0)} chunks")

        return self.distractor_kb_id

    def run_single_experiment(
        self,
        folder_path: Path,
        relative_path: str,
        preset_name: str,
        preset_config: Dict[str, Any],
        questions: List[Dict],
        phase: int,
        is_baseline: bool = False,
    ) -> Dict[str, Any]:
        """Run a single experiment with given configuration.

        Args:
            folder_path: Absolute path to folder
            relative_path: Relative path for naming
            preset_name: Name of the preset being used
            preset_config: Configuration for this experiment
            questions: List of evaluation questions
            phase: 1 or 2
            is_baseline: Whether this is the baseline for Phase 2

        Returns:
            Experiment result dictionary
        """
        experiment_id = self._next_experiment_id()
        kb_name = f"eval_{experiment_id}_{relative_path.replace('/', '_')[:30]}"

        print(f"    Running {experiment_id}: {preset_name}")

        try:
            dataset_id = self.ingestion_engine.create_kb_for_experiment(
                name=kb_name,
                preset_config=preset_config,
            )
            self.created_kbs.append(dataset_id)

            ingestion_result = self.ingestion_engine.ingest_folder(
                dataset_id=dataset_id,
                folder_path=folder_path,
                preset_config=preset_config,
                wait_for_completion=True,
            )

            # Capture ingestion warnings and chunk count
            ingestion_warnings = ingestion_result.get("warnings", [])
            chunk_count = ingestion_result.get("chunk_count", 0)

            retrieval_results = self.retrieval_engine.retrieve_for_evaluation(
                questions=questions,
                target_kb_id=dataset_id,
                distractor_kb_id=self.distractor_kb_id,
                top_k=self.config.top_k,
                similarity_threshold=self.config.similarity_threshold,
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
                "is_baseline": is_baseline,
                "config": preset_config,
                "chunk_count": chunk_count,
                "metrics": metrics,
                "questions": enriched_results,
            }

            # Add warnings if any
            if ingestion_warnings:
                experiment_result["warnings"] = ingestion_warnings

            folder_output = self._get_folder_output_path(relative_path)
            folder_output.mkdir(parents=True, exist_ok=True)

            with open(folder_output / f"{experiment_id}.json", "w", encoding="utf-8") as f:
                json.dump(experiment_result, f, ensure_ascii=False, indent=2)

            print(f"      Score: {metrics['combined_score']:.4f} (chunks: {chunk_count})")

            return experiment_result

        except RAGFlowAPIError as e:
            print(f"      ERROR: {e}")
            return {
                "experiment_id": experiment_id,
                "folder_path": relative_path,
                "phase": phase,
                "preset": preset_name,
                "config": preset_config,
                "error": str(e),
                "metrics": {"combined_score": 0.0},
            }

    def run_phase1_tournament(
        self,
        folder_path: Path,
        relative_path: str,
        questions: List[Dict],
    ) -> Tuple[str, Dict[str, Any], List[Dict]]:
        """Run Phase 1 preset tournament for a folder.

        Returns:
            Tuple of (winning_preset_name, winning_config, all_experiment_results)
        """
        print(f"  Phase 1: Preset Selection for {relative_path}")

        applicable_presets = self._get_applicable_presets(folder_path)
        results = []
        best_score = -1.0
        best_preset = "general"
        best_config = self.config.get_preset("general")

        for preset_name in applicable_presets:
            preset_config = self.config.get_preset(preset_name)

            result = self.run_single_experiment(
                folder_path=folder_path,
                relative_path=relative_path,
                preset_name=preset_name,
                preset_config=preset_config,
                questions=questions,
                phase=1,
            )
            results.append(result)

            score = result.get("metrics", {}).get("combined_score", 0.0)
            if score > best_score:
                best_score = score
                best_preset = preset_name
                best_config = preset_config

        print(f"  Phase 1 Winner: {best_preset} (score: {best_score:.4f})")
        return best_preset, best_config, results

    def run_phase2_finetuning(
        self,
        folder_path: Path,
        relative_path: str,
        questions: List[Dict],
        winning_preset: str,
        baseline_config: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict]]:
        """Run Phase 2 parameter fine-tuning.

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

        baseline_result = self.run_single_experiment(
            folder_path=folder_path,
            relative_path=relative_path,
            preset_name=winning_preset,
            preset_config=current_config,
            questions=questions,
            phase=2,
            is_baseline=True,
        )
        results.append(baseline_result)
        best_score = baseline_result.get("metrics", {}).get("combined_score", 0.0)

        for param_spec in optimization_space:
            param_name = param_spec["name"]
            param_values = param_spec["values"]

            print(f"    Optimizing {param_name}...")

            best_param_value = current_config.get(param_name)
            best_param_score = best_score

            for value in param_values:
                if value == current_config.get(param_name):
                    continue

                test_config = dict(current_config)
                test_config[param_name] = value

                result = self.run_single_experiment(
                    folder_path=folder_path,
                    relative_path=relative_path,
                    preset_name=winning_preset,
                    preset_config=test_config,
                    questions=questions,
                    phase=2,
                )
                results.append(result)

                score = result.get("metrics", {}).get("combined_score", 0.0)
                if score > best_param_score:
                    best_param_score = score
                    best_param_value = value

            current_config[param_name] = best_param_value
            if best_param_score > best_score:
                best_score = best_param_score
                best_config = dict(current_config)

        print(f"  Phase 2 Best Score: {best_score:.4f}")
        return best_config, results

    def run_folder_optimization(
        self,
        folder_path: Path,
        relative_path: str,
    ) -> Dict[str, Any]:
        """Run full optimization for a single folder.

        Returns:
            Dict with folder results including winning config
        """
        print(f"\nOptimizing: {relative_path}")

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

        winning_preset, winning_config, phase1_results = self.run_phase1_tournament(
            folder_path=folder_path,
            relative_path=relative_path,
            questions=questions,
        )

        best_config, phase2_results = self.run_phase2_finetuning(
            folder_path=folder_path,
            relative_path=relative_path,
            questions=questions,
            winning_preset=winning_preset,
            baseline_config=winning_config,
        )

        return {
            "folder_path": relative_path,
            "status": "completed",
            "winning_preset": winning_preset,
            "recommended_config": best_config,
            "phase1_experiments": len(phase1_results),
            "phase2_experiments": len(phase2_results),
            "questions_used": len(questions),
        }

    def run_full_optimization(
        self,
        target_folder: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run optimization for all folders or a specific folder.

        Args:
            target_folder: Optional specific folder path to process

        Returns:
            Summary of optimization results
        """
        self.run_output_path.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"RAG Ingestion Parameter Optimizer")
        print(f"Run ID: {self.run_id}")
        print(f"{'='*60}\n")

        self.test_api_connectivity()

        self.setup_distractor_kb()

        if target_folder:
            folder_path = self.config.source_docs_path / target_folder
            if not folder_path.exists():
                raise ValueError(f"Folder not found: {target_folder}")
            folders = [(folder_path, target_folder)]
        else:
            folders = self._find_bottom_folders(self.config.source_docs_path)

        print(f"\nFound {len(folders)} folder(s) to process")

        results = []
        for folder_path, relative_path in folders:
            try:
                result = self.run_folder_optimization(folder_path, relative_path)
                results.append(result)
            except Exception as e:
                print(f"  ERROR processing {relative_path}: {e}")
                results.append({
                    "folder_path": relative_path,
                    "status": "error",
                    "error": str(e),
                })

        summary = {
            "run_id": self.run_id,
            "total_folders": len(folders),
            "completed": sum(1 for r in results if r.get("status") == "completed"),
            "errors": sum(1 for r in results if r.get("status") == "error"),
            "skipped": sum(1 for r in results if r.get("status") == "skipped"),
            "folder_results": results,
        }

        with open(self.run_output_path / "summary_report.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        self._generate_markdown_report(results)

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

    def cleanup(self, include_distractor: bool = False) -> None:
        """Clean up created knowledge bases."""
        print("\nCleaning up temporary knowledge bases...")

        for kb_id in self.created_kbs:
            try:
                self.ingestion_engine.cleanup_kb(kb_id)
            except Exception:
                pass

        if include_distractor and self.distractor_kb_id:
            try:
                self.ingestion_engine.cleanup_kb(self.distractor_kb_id)
            except Exception:
                pass

        self.created_kbs = []
        print("Cleanup complete")
