"""
Relevance Judge - Uses LLM to evaluate if retrieved chunks answer questions.

This module:
1. Takes a question and retrieved chunk
2. Asks LLM if the chunk contains relevant information
3. Returns (is_relevant, rationale) for metrics calculation

The rationale is returned in Chinese for Chinese document evaluation.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI


class RelevanceJudge:
    """Uses LLM to judge whether retrieved chunks are relevant to questions."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        provider: str = "openai",
    ):
        self.model = model
        self.provider = provider

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

    def judge_relevance(
        self,
        question: str,
        chunk_content: str,
    ) -> Tuple[bool, str]:
        """Judge whether a chunk is relevant to a question.

        Args:
            question: The evaluation question
            chunk_content: The retrieved chunk content

        Returns:
            Tuple of (is_relevant, rationale)
        """
        prompt = f"""你正在评估一个检索到的文本片段是否包含有助于回答问题的信息。

问题: {question}

检索到的文本片段:
{chunk_content[:2000]}

该文本片段是否包含有助于回答问题的信息？请考虑：
- 是否包含与问题直接相关的事实、流程或数据？
- 用户能否从中找到有用的信息来回答他们的问题？

请用中文简要解释原因，并以JSON格式返回:
{{"relevant": true/false, "rationale": "简要说明相关或不相关的原因（中文）"}}

只返回有效的JSON格式。"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=200,
            )

            response_text = response.choices[0].message.content.strip()

            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]

            parsed = json.loads(response_text)
            return (
                bool(parsed.get("relevant", False)),
                parsed.get("rationale", "No rationale provided"),
            )

        except (json.JSONDecodeError, Exception) as e:
            return (False, f"Error in relevance judgment: {str(e)}")

    def judge_retrieval_results(
        self,
        retrieval_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Judge relevance for all retrieval results.

        Args:
            retrieval_results: List of retrieval results from RetrievalEngine

        Returns:
            Updated retrieval results with relevance judgments
        """
        judged_results = []

        for result in retrieval_results:
            question = result.get("question", "")
            chunks = result.get("chunks_retrieved", [])

            judged_chunks = []
            for chunk in chunks:
                content = chunk.get("content", "")

                if chunk.get("from_distractor", False):
                    relevant = False
                    rationale = "该文本片段来自干扰知识库，标记为不相关"
                else:
                    relevant, rationale = self.judge_relevance(question, content)

                judged_chunk = dict(chunk)
                judged_chunk["relevant"] = relevant
                judged_chunk["rationale"] = rationale
                judged_chunks.append(judged_chunk)

            judged_results.append({
                "question_id": result.get("question_id", ""),
                "question": question,
                "chunks_retrieved": judged_chunks,
            })

        return judged_results

    def batch_judge_relevance(
        self,
        questions_and_chunks: List[Tuple[str, str]],
    ) -> List[Tuple[bool, str]]:
        """Batch judge relevance for multiple question-chunk pairs.

        More efficient than individual calls for large batches.

        Args:
            questions_and_chunks: List of (question, chunk_content) tuples

        Returns:
            List of (is_relevant, rationale) tuples
        """
        results = []
        for question, chunk in questions_and_chunks:
            result = self.judge_relevance(question, chunk)
            results.append(result)
        return results
