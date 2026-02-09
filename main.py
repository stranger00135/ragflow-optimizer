"""CLI entry point for RAG Ingestion Parameter Optimizer."""

import argparse
import sys
from pathlib import Path

from src.config_loader import Config
from src.orchestrator import Orchestrator
from src.backends.base import BackendError
from src.backends.registry import get_backend
from src.cleanup import cleanup_all_temp_kbs, force_cleanup_all_temp_kbs
from src.ragflow_client import RAGFlowClient
from src.question_generator import QuestionGenerator


def get_config(backend_override: str = None) -> Config:
    """Load configuration from default paths.

    Args:
        backend_override: Optional backend type to override config setting
    """
    project_root = Path(__file__).parent
    config_path = project_root / "config" / "config.yaml"
    env_path = project_root / ".env"
    # Pass env_path if exists, otherwise Config will auto-discover
    config = Config(str(config_path), str(env_path) if env_path.exists() else None)

    # Override backend type if specified
    if backend_override:
        config.data.setdefault("backend", {})["type"] = backend_override

    return config


def cmd_run(args):
    """Run full optimization."""
    backend = getattr(args, 'backend', None)
    config = get_config(backend_override=backend)
    orchestrator = Orchestrator(config)

    try:
        summary = orchestrator.run_full_optimization(
            target_folder=args.folder if hasattr(args, 'folder') else None
        )

        print(f"\n{'='*60}")
        print("OPTIMIZATION COMPLETE")
        print(f"{'='*60}")
        print(f"Run ID: {summary['run_id']}")
        print(f"Folders processed: {summary['total_folders']}")
        print(f"Completed: {summary['completed']}")
        print(f"Errors: {summary['errors']}")
        print(f"Skipped: {summary['skipped']}")
        print(f"\nResults saved to: {orchestrator.run_output_path}")

        if not args.keep_kbs:
            orchestrator.cleanup(include_distractor=False)

    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def cmd_generate_questions(args):
    """Generate questions only without running experiments."""
    config = get_config()

    generator = QuestionGenerator(
        api_key=config.llm_api_key,
        model=config.llm_model,
        provider=config.llm_provider,
        cache_path=config.questions_cache_path,
    )

    source_path = config.source_docs_path

    def find_bottom_folders(path: Path):
        """Find all bottom-level folders containing files."""
        supported = {".pdf", ".doc", ".docx", ".txt", ".md", ".xlsx", ".xls", ".csv", ".ppt", ".pptx"}
        folders = []

        def scan(p: Path, rel: str = ""):
            has_files = False
            for item in p.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_file() and item.suffix.lower() in supported:
                    has_files = True
                elif item.is_dir():
                    sub_rel = f"{rel}/{item.name}" if rel else item.name
                    scan(item, sub_rel)
            if has_files:
                folders.append((p, rel if rel else p.name))

        scan(path)
        return folders

    if args.folder:
        folder_path = (source_path / args.folder).resolve()
        source_resolved = source_path.resolve()
        try:
            folder_path.relative_to(source_resolved)
        except ValueError:
            print("ERROR: Folder path must be inside source_docs (path traversal not allowed).")
            sys.exit(1)
        if not folder_path.exists() or not folder_path.is_dir():
            print(f"ERROR: Folder not found: {args.folder}")
            sys.exit(1)
        folders = [(folder_path, args.folder)]
    else:
        folders = find_bottom_folders(source_path)

    print(f"Generating questions for {len(folders)} folder(s)...")

    for folder_path, relative_path in folders:
        print(f"\nProcessing: {relative_path}")
        result = generator.generate_questions(
            folder_path=folder_path,
            relative_path=relative_path,
            num_questions=config.questions_per_folder,
            use_cache=not args.regenerate,
        )
        questions = result.get("questions", [])
        print(f"  Generated {len(questions)} questions")

        if args.verbose:
            for i, q in enumerate(questions, 1):
                print(f"    {i}. {q.get('question', 'N/A')}")

    print("\nQuestion generation complete.")


def cmd_cleanup(args):
    """Clean up temporary knowledge bases."""
    backend = getattr(args, 'backend', None)
    config = get_config(backend_override=backend)
    backend_name = config.backend_name

    print(f"Backend: {backend_name}")

    if backend_name == "ragflow":
        client = RAGFlowClient(
            base_url=config.ragflow_base_url,
            api_key=config.ragflow_api_key,
        )

        if args.force:
            print("Force cleanup mode: removing ALL temporary KBs (recovery mode)...")
            deleted = force_cleanup_all_temp_kbs(client)
        else:
            print("Cleaning up temporary evaluation knowledge bases...")
            deleted = cleanup_all_temp_kbs(client)

        print(f"\nDeleted {len(deleted)} knowledge base(s)")
    else:
        # Use backend abstraction for other backends
        try:
            backend_instance = get_backend(config)
            print(f"Cleanup for {backend_name} backend is limited to API testing.")
            print("Use the backend's native UI or API for full cleanup.")
        except BackendError as e:
            print(f"Cannot connect to backend: {e}")


def cmd_test_api(args):
    """Test API connectivity."""
    backend_arg = getattr(args, 'backend', None)
    config = get_config(backend_override=backend_arg)
    backend_name = config.backend_name

    print(f"Testing {backend_name} backend connectivity...")

    if backend_name == "ragflow":
        project_root = Path(__file__).parent
        config_path = project_root / "config" / "config.yaml"
        env_path = project_root / ".env"

        from src.api_test import test_api

        success = test_api(
            str(config_path),
            str(env_path) if env_path.exists() else None,
        )
        sys.exit(0 if success else 1)

    try:
        backend = get_backend(config)
        success = backend.test_connection()
        if success:
            print(f"  {backend_name.upper()} API: OK")
        else:
            print(f"  {backend_name.upper()} API: FAILED")
        sys.exit(0 if success else 1)
    except BackendError as exc:
        print(f"  {backend_name.upper()} API: FAILED")
        print(f"  Error: {exc}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="RAG Ingestion Parameter Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py run                         Run full optimization on all folders
  python main.py run --folder "docs"         Optimize specific folder
  python main.py run --backend openwebui     Use OpenWebUI backend
  python main.py generate-questions          Generate questions only
  python main.py cleanup                     Remove temporary KBs
  python main.py test-api                    Test API connectivity
  python main.py test-api --backend openwebui Test OpenWebUI connectivity
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run command
    run_parser = subparsers.add_parser("run", help="Run optimization")
    run_parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Specific folder to optimize (relative path)",
    )
    run_parser.add_argument(
        "--keep-kbs",
        action="store_true",
        help="Keep temporary knowledge bases after run",
    )
    run_parser.add_argument(
        "--backend",
        type=str,
        choices=["ragflow", "openwebui"],
        default=None,
        help="Backend to use (overrides config.yaml setting)",
    )
    run_parser.set_defaults(func=cmd_run)

    # generate-questions command
    gen_parser = subparsers.add_parser(
        "generate-questions",
        help="Generate evaluation questions only",
    )
    gen_parser.add_argument(
        "--folder",
        type=str,
        default=None,
        help="Specific folder to generate questions for",
    )
    gen_parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Regenerate questions even if cached",
    )
    gen_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print generated questions",
    )
    gen_parser.set_defaults(func=cmd_generate_questions)

    # cleanup command
    cleanup_parser = subparsers.add_parser(
        "cleanup",
        help="Remove temporary evaluation KBs",
    )
    cleanup_parser.add_argument(
        "--force",
        action="store_true",
        help="Force cleanup ALL temp KBs including from registry (recovery mode)",
    )
    cleanup_parser.add_argument(
        "--backend",
        type=str,
        choices=["ragflow", "openwebui"],
        default=None,
        help="Backend to use (overrides config.yaml setting)",
    )
    cleanup_parser.set_defaults(func=cmd_cleanup)

    # test-api command
    test_parser = subparsers.add_parser(
        "test-api",
        help="Test API connectivity",
    )
    test_parser.add_argument(
        "--backend",
        type=str,
        choices=["ragflow", "openwebui"],
        default=None,
        help="Backend to test (overrides config.yaml setting)",
    )
    test_parser.set_defaults(func=cmd_test_api)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
