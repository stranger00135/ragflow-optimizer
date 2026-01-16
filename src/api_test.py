"""Test RAGFlow API connectivity and basic operations."""

import sys
from pathlib import Path
from typing import Optional

from .config_loader import Config
from .ragflow_client import RAGFlowClient, RAGFlowAPIError


def test_api(config_path: str, env_path: Optional[str] = None) -> bool:
    """Test all RAGFlow API operations.

    Args:
        config_path: Path to config.yaml
        env_path: Optional path to .env file

    Returns:
        True if all tests pass
    """
    print("=" * 60)
    print("RAGFlow API Connectivity Test")
    print("=" * 60)

    config = Config(config_path, env_path)

    client = RAGFlowClient(
        base_url=config.ragflow_base_url,
        api_key=config.ragflow_api_key,
    )

    print("\n1. Testing basic connectivity...")
    if client.test_connection():
        print("   PASS: Can connect to RAGFlow API")
    else:
        print("   FAIL: Cannot connect to RAGFlow API")
        return False

    print("\n2. Testing list datasets...")
    try:
        datasets = client.list_datasets()
        print(f"   PASS: Found {len(datasets)} datasets")
    except RAGFlowAPIError as e:
        print(f"   FAIL: {e}")
        return False

    print("\n3. Testing create dataset...")
    try:
        test_ds = client.create_dataset(
            name="__api_test_temp__",
            chunk_method="naive",
        )
        test_ds_id = test_ds.get("id")
        if test_ds_id:
            print(f"   PASS: Created test dataset: {test_ds_id}")
        else:
            print("   FAIL: No dataset ID returned")
            return False
    except RAGFlowAPIError as e:
        print(f"   FAIL: {e}")
        return False

    print("\n4. Testing delete dataset...")
    try:
        if client.delete_dataset(test_ds_id):
            print("   PASS: Deleted test dataset")
        else:
            print("   WARN: Delete may have failed")
    except RAGFlowAPIError as e:
        print(f"   FAIL: {e}")
        return False

    print("\n5. Testing retrieval endpoint...")
    try:
        datasets = client.list_datasets()
        if datasets:
            test_ids = [datasets[0].get("id")]
            chunks = client.retrieve_chunks(
                question="test query",
                dataset_ids=test_ids,
                top_k=1,
            )
            print(f"   PASS: Retrieval returned {len(chunks)} chunks")
        else:
            print("   SKIP: No datasets available for retrieval test")
    except RAGFlowAPIError as e:
        print(f"   WARN: Retrieval test: {e}")

    print("\n" + "=" * 60)
    print("All critical API tests PASSED")
    print("=" * 60)

    return True


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    config_path = project_root / "config" / "config.yaml"
    env_path = project_root / ".env"

    success = test_api(
        str(config_path),
        str(env_path) if env_path.exists() else None,
    )
    sys.exit(0 if success else 1)
