"""Question generator using LLM to create evaluation questions from folder content."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


class QuestionGenerator:
    """Generates evaluation questions from document content using LLM."""

    SUPPORTED_EXTENSIONS = {
        ".pdf", ".doc", ".docx", ".txt", ".md",
        ".xlsx", ".xls", ".csv",
        ".ppt", ".pptx",
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
        cache_path: Optional[Path] = None,
    ):
        self.model = model
        self.provider = provider
        self.cache_path = cache_path

        if provider == "openai":
            self.client = OpenAI(api_key=api_key)
        elif provider == "deepseek":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
        elif provider == "dashscope":
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
        else:
            self.client = OpenAI(api_key=api_key)

    def _get_cache_file(self, folder_path: str) -> Path:
        """Get cache file path for a folder."""
        if not self.cache_path:
            return None

        safe_name = folder_path.replace("/", "_").replace("\\", "_").replace(" ", "_")
        return self.cache_path / f"{safe_name}.json"

    def _load_from_cache(self, folder_path: str) -> Optional[Dict]:
        """Load cached questions for a folder."""
        cache_file = self._get_cache_file(folder_path)
        if cache_file and cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def _save_to_cache(self, folder_path: str, data: Dict) -> None:
        """Save questions to cache."""
        cache_file = self._get_cache_file(folder_path)
        if cache_file:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _list_files_in_folder(self, folder_path: Path) -> List[str]:
        """List supported files in a folder (non-recursive)."""
        files = []
        if folder_path.is_dir():
            for item in folder_path.iterdir():
                if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    files.append(item.name)
        return files

    def _read_file_sample(self, file_path: Path, max_chars: int = 2000) -> str:
        """Read a sample of file content for text files."""
        try:
            if file_path.suffix.lower() in {".txt", ".md"}:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()[:max_chars]
        except Exception:
            pass
        return ""

    def _build_prompt(
        self,
        folder_path: str,
        files: List[str],
        file_samples: Dict[str, str],
        num_questions: int,
    ) -> str:
        """Build the prompt for question generation."""
        file_info = "\n".join([f"- {f}" for f in files])

        sample_content = ""
        for fname, content in file_samples.items():
            if content:
                sample_content += f"\n\n[Sample from {fname}]:\n{content[:1000]}"

        prompt = f"""You are analyzing documents in a folder to generate evaluation questions for a RAG (Retrieval-Augmented Generation) system.

Folder path: {folder_path}

Files in this folder:
{file_info}

{sample_content if sample_content else ""}

Based on the folder path, file names, and any available content samples:

1. First, infer the purpose and nature of these documents (e.g., SOPs, manuals, reports, records, etc.)
2. Then generate exactly {num_questions} evaluation questions that:
   - Are realistic questions a user might ask about this content
   - Vary in complexity (some simple lookups, some requiring synthesis)
   - Are specific enough that the answer would be in the documents
   - Are written in the same language as the documents (Chinese if file names are Chinese)

Return your response as JSON with this exact format:
{{
  "inferred_purpose": "Brief description of what these documents are for",
  "questions": [
    {{"question_id": "q_001", "question": "Question 1 text"}},
    {{"question_id": "q_002", "question": "Question 2 text"}},
    ...
  ]
}}

Only return valid JSON, no markdown formatting or additional text."""

        return prompt

    def generate_questions(
        self,
        folder_path: Path,
        relative_path: str,
        num_questions: int = 5,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """Generate questions for a folder.

        Args:
            folder_path: Absolute path to the folder
            relative_path: Relative path from source_docs for caching
            num_questions: Number of questions to generate
            use_cache: Whether to use cached questions if available

        Returns:
            Dict with folder info, inferred purpose, and questions
        """
        if use_cache:
            cached = self._load_from_cache(relative_path)
            if cached:
                return cached

        files = self._list_files_in_folder(folder_path)

        if not files:
            return {
                "folder_path": relative_path,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "files_analyzed": [],
                "inferred_purpose": "Empty folder - no supported files found",
                "questions": [],
            }

        file_samples = {}
        for fname in files[:5]:
            file_path = folder_path / fname
            sample = self._read_file_sample(file_path)
            if sample:
                file_samples[fname] = sample

        prompt = self._build_prompt(relative_path, files, file_samples, num_questions)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )

        response_text = response.choices[0].message.content.strip()

        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]

        try:
            parsed = json.loads(response_text)
        except json.JSONDecodeError:
            parsed = {
                "inferred_purpose": "Could not parse LLM response",
                "questions": [],
            }

        result = {
            "folder_path": relative_path,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "files_analyzed": files,
            "inferred_purpose": parsed.get("inferred_purpose", "Unknown"),
            "questions": parsed.get("questions", []),
        }

        if use_cache:
            self._save_to_cache(relative_path, result)

        return result

    def generate_questions_for_all_folders(
        self,
        source_docs_path: Path,
        num_questions: int = 5,
        use_cache: bool = True,
    ) -> Dict[str, Dict]:
        """Generate questions for all bottom-level folders.

        Returns:
            Dict mapping relative folder paths to their question data
        """
        results = {}

        def find_bottom_folders(path: Path, relative: str = "") -> List[tuple]:
            """Find all bottom-level folders (folders containing files)."""
            folders = []
            has_files = False
            has_subdirs = False

            for item in path.iterdir():
                if item.name.startswith("."):
                    continue
                if item.is_file() and item.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                    has_files = True
                elif item.is_dir():
                    has_subdirs = True
                    sub_relative = f"{relative}/{item.name}" if relative else item.name
                    folders.extend(find_bottom_folders(item, sub_relative))

            if has_files:
                folders.append((path, relative if relative else path.name))

            return folders

        bottom_folders = find_bottom_folders(source_docs_path)

        for folder_path, relative_path in bottom_folders:
            result = self.generate_questions(
                folder_path,
                relative_path,
                num_questions,
                use_cache,
            )
            results[relative_path] = result

        return results
