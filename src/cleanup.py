"""Cleanup utilities for RAGFlow knowledge bases."""

from typing import List

from .ragflow_client import RAGFlowClient


def cleanup_eval_kbs(client: RAGFlowClient, prefix: str = "eval_") -> List[str]:
    """Delete all knowledge bases with a given prefix.

    Args:
        client: RAGFlow client
        prefix: Prefix to match for deletion

    Returns:
        List of deleted dataset IDs
    """
    deleted = []
    datasets = client.list_datasets()

    for ds in datasets:
        name = ds.get("name", "")
        if name.startswith(prefix):
            ds_id = ds.get("id")
            if ds_id and client.delete_dataset(ds_id):
                deleted.append(ds_id)
                print(f"Deleted: {name}")

    return deleted


def cleanup_all_temp_kbs(client: RAGFlowClient) -> List[str]:
    """Delete all temporary evaluation knowledge bases.

    Args:
        client: RAGFlow client

    Returns:
        List of deleted dataset IDs
    """
    return cleanup_eval_kbs(client, prefix="eval_exp_")
