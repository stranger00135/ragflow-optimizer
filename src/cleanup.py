"""Cleanup utilities for RAGFlow knowledge bases."""

import json
from pathlib import Path
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

    This includes:
    - eval_exp_* (legacy naming from v3)
    - eval_temp_* (v4 temp KBs)

    Args:
        client: RAGFlow client

    Returns:
        List of deleted dataset IDs
    """
    deleted = []
    # Clean up both v3 and v4 temp KB prefixes
    deleted.extend(cleanup_eval_kbs(client, prefix="eval_exp_"))
    deleted.extend(cleanup_eval_kbs(client, prefix="eval_temp_"))
    return deleted


def cleanup_from_registry(client: RAGFlowClient) -> List[str]:
    """Clean up KBs listed in the cleanup registry file.

    Used for recovery after crashes or interruptions.

    Args:
        client: RAGFlow client

    Returns:
        List of deleted dataset IDs
    """
    registry_path = Path(".claude/cleanup_registry.json")
    deleted = []

    if not registry_path.exists():
        print("No cleanup registry found.")
        return deleted

    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading registry: {e}")
        return deleted

    kb_ids = registry.get("kb_ids", [])
    run_id = registry.get("run_id", "unknown")

    print(f"Found cleanup registry from run: {run_id}")
    print(f"KBs to clean up: {len(kb_ids)}")

    for kb_id in kb_ids:
        try:
            if client.delete_dataset(kb_id):
                deleted.append(kb_id)
                print(f"  Deleted KB: {kb_id}")
        except Exception as e:
            print(f"  Failed to delete {kb_id}: {e}")

    # Clear the registry after cleanup
    registry_path.unlink()
    print("Registry cleared.")

    return deleted


def force_cleanup_all_temp_kbs(client: RAGFlowClient) -> List[str]:
    """Force cleanup ALL temporary evaluation KBs (recovery mode).

    This is more aggressive than cleanup_all_temp_kbs:
    - Cleans up from registry first
    - Then scans for any orphaned temp KBs by prefix
    - Useful after crashes or when orphaned KBs accumulate

    Args:
        client: RAGFlow client

    Returns:
        List of deleted dataset IDs
    """
    deleted = []

    # First, clean up from registry if it exists
    deleted.extend(cleanup_from_registry(client))

    # Then scan for any orphaned temp KBs
    print("\nScanning for orphaned temporary KBs...")

    datasets = client.list_datasets()
    temp_prefixes = ["eval_exp_", "eval_temp_"]

    for ds in datasets:
        name = ds.get("name", "")
        ds_id = ds.get("id")

        # Check if it's a temp KB
        is_temp = any(name.startswith(prefix) for prefix in temp_prefixes)

        if is_temp and ds_id and ds_id not in deleted:
            try:
                if client.delete_dataset(ds_id):
                    deleted.append(ds_id)
                    print(f"  Deleted orphaned KB: {name}")
            except Exception as e:
                print(f"  Failed to delete {name}: {e}")

    return deleted
