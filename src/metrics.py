"""Metrics calculation for RAG evaluation."""

from typing import Any, Dict, List


def calculate_question_metrics(
    judged_chunks: List[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Calculate metrics for a single question's retrieval results.

    Args:
        judged_chunks: List of chunks with relevance judgments
        top_k: Number of top chunks to consider

    Returns:
        Dict with precision_at_k, first_relevant_rank, mrr_contribution, is_fail
    """
    chunks = judged_chunks[:top_k]

    if not chunks:
        return {
            "precision_at_k": 0.0,
            "first_relevant_rank": None,
            "mrr_contribution": 0.0,
            "is_fail": True,
            "relevant_count": 0,
        }

    relevant_count = sum(1 for c in chunks if c.get("relevant", False))

    precision_at_k = relevant_count / len(chunks) if chunks else 0.0

    first_relevant_rank = None
    for i, chunk in enumerate(chunks):
        if chunk.get("relevant", False):
            first_relevant_rank = i + 1
            break

    mrr_contribution = 1.0 / first_relevant_rank if first_relevant_rank else 0.0

    is_fail = relevant_count == 0

    return {
        "precision_at_k": precision_at_k,
        "first_relevant_rank": first_relevant_rank,
        "mrr_contribution": mrr_contribution,
        "is_fail": is_fail,
        "relevant_count": relevant_count,
    }


def calculate_experiment_metrics(
    judged_results: List[Dict[str, Any]],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Calculate aggregate metrics for an experiment.

    Args:
        judged_results: List of judged retrieval results for all questions
        top_k: Number of top chunks considered

    Returns:
        Dict with fail_rate, precision_at_k, mrr, combined_score
    """
    if not judged_results:
        return {
            "fail_rate": 1.0,
            "precision_at_k": 0.0,
            "mrr": 0.0,
            "combined_score": 0.0,
            "total_questions": 0,
        }

    question_metrics = []

    for result in judged_results:
        chunks = result.get("chunks_retrieved", [])
        metrics = calculate_question_metrics(chunks, top_k)
        question_metrics.append(metrics)

    total_questions = len(question_metrics)

    fail_count = sum(1 for m in question_metrics if m["is_fail"])
    fail_rate = fail_count / total_questions

    precision_sum = sum(m["precision_at_k"] for m in question_metrics)
    avg_precision = precision_sum / total_questions

    mrr_sum = sum(m["mrr_contribution"] for m in question_metrics)
    mrr = mrr_sum / total_questions

    combined_score = (
        (1 - fail_rate) * 0.4 +
        avg_precision * 0.3 +
        mrr * 0.3
    )

    return {
        "fail_rate": round(fail_rate, 4),
        "precision_at_k": round(avg_precision, 4),
        "mrr": round(mrr, 4),
        "combined_score": round(combined_score, 4),
        "total_questions": total_questions,
        "fail_count": fail_count,
    }


def enrich_results_with_metrics(
    judged_results: List[Dict[str, Any]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """Enrich judged results with per-question metrics.

    Args:
        judged_results: List of judged retrieval results
        top_k: Number of top chunks considered

    Returns:
        List of results with added metric fields
    """
    enriched = []

    for result in judged_results:
        chunks = result.get("chunks_retrieved", [])
        metrics = calculate_question_metrics(chunks, top_k)

        enriched_result = dict(result)
        enriched_result["precision_at_k"] = metrics["precision_at_k"]
        enriched_result["first_relevant_rank"] = metrics["first_relevant_rank"]
        enriched_result["mrr_contribution"] = metrics["mrr_contribution"]
        enriched.append(enriched_result)

    return enriched
