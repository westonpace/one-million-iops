#!/usr/bin/env python3
"""
Lance Vector Search Benchmark

Benchmarks Lance vector search performance with:
- 3 datasets with 1M rows each
- 768-dimensional float32 vectors
- IVF_PQ index with 256 partitions and 48 subvectors
- 2,000 queries
- ThreadPoolExecutor with 8 workers for concurrent execution
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Dict

import lance
import numpy as np
import pyarrow as pa
from tqdm import tqdm

# ==================== CONFIGURATION ====================

# Edit these paths as needed
DATASET_PATHS = [
    "./lance_dataset_0",
    "./lance_dataset_1",
    "./lance_dataset_2",
]

# Dataset parameters
NUM_DATASETS = 3
ROWS_PER_DATASET = 1_000_000
VECTOR_DIM = 768
BATCH_SIZE = 100_000  # For memory-efficient dataset generation

# Index parameters
NUM_PARTITIONS = 256
NUM_SUB_VECTORS = 48  # 768 / 48 = 16 dimensions per subvector

# Query parameters
NUM_QUERIES = 2000
NUM_WORKERS = 8

# Query search parameters
QUERY_K = 50  # Top K results
QUERY_NPROBES = 20
QUERY_REFINE_FACTOR = 10


# ==================== DATASET GENERATION ====================


def dataset_exists(uri: str, expected_rows: int = ROWS_PER_DATASET) -> bool:
    """
    Check if a dataset exists and has the expected number of rows.

    Args:
        uri: Path to the dataset
        expected_rows: Expected number of rows

    Returns:
        True if dataset exists with correct number of rows, False otherwise
    """
    try:
        ds = lance.dataset(uri)
        row_count = ds.count_rows()
        return row_count == expected_rows
    except Exception:
        return False


def has_vector_index(dataset: lance.LanceDataset, column: str = "vector") -> bool:
    """
    Check if a dataset has a vector index on the specified column.

    Args:
        dataset: LanceDataset to check
        column: Column name to check for index

    Returns:
        True if index exists, False otherwise
    """
    try:
        indices = dataset.list_indices()
        return any(idx["columns"] == [column] for idx in indices)
    except Exception:
        return False


def generate_dataset(
    uri: str, num_rows: int = ROWS_PER_DATASET, dim: int = VECTOR_DIM
) -> lance.LanceDataset:
    """
    Generate a Lance dataset with random vectors in batches.

    Args:
        uri: Path to create the dataset
        num_rows: Number of rows to generate
        dim: Vector dimensionality

    Returns:
        LanceDataset object
    """
    num_batches = num_rows // BATCH_SIZE

    print(f"\nGenerating dataset: {uri}")
    for i in tqdm(range(num_batches), desc="  Writing batches"):
        # Generate random vectors for this batch
        vectors = np.random.randn(BATCH_SIZE, dim).astype(np.float32)

        # Create PyArrow FixedSizeListArray
        arr = pa.FixedSizeListArray.from_arrays(
            pa.array(vectors.ravel(), type=pa.float32()), list_size=dim
        )

        # Create table with single vector column
        table = pa.Table.from_arrays([arr], names=["vector"])

        # Write batch (create on first, append on subsequent)
        mode = "create" if i == 0 else "append"
        lance.write_dataset(table, uri, mode=mode)

    return lance.dataset(uri)


def create_index(dataset: lance.LanceDataset) -> None:
    """
    Create IVF_PQ vector index on the dataset.

    Args:
        dataset: LanceDataset to index
    """
    dataset.create_index(
        column="vector",
        index_type="IVF_PQ",
        num_partitions=NUM_PARTITIONS,
        num_sub_vectors=NUM_SUB_VECTORS,
        metric="L2",
    )


# ==================== QUERY GENERATION ====================


def generate_queries(
    num_queries: int = NUM_QUERIES, dim: int = VECTOR_DIM
) -> np.ndarray:
    """
    Generate random query vectors.

    Args:
        num_queries: Number of query vectors to generate
        dim: Vector dimensionality

    Returns:
        numpy array of shape (num_queries, dim)
    """
    print(f"\nGenerating {num_queries:,} query vectors...")
    start = time.perf_counter()
    queries = np.random.randn(num_queries, dim).astype(np.float32)
    elapsed = time.perf_counter() - start
    print(f"  Done in {elapsed:.2f}s")
    return queries


# ==================== QUERY EXECUTION ====================


def execute_query(
    dataset_idx: int, query_vector: np.ndarray, datasets: List[lance.LanceDataset]
) -> float:
    """
    Execute a single vector search query and return latency.

    Args:
        dataset_idx: Index of dataset to query
        query_vector: Query vector
        datasets: List of datasets

    Returns:
        Query latency in seconds
    """
    start = time.perf_counter()

    datasets[dataset_idx].to_table(
        nearest={
            "column": "vector",
            "q": query_vector,
            "k": QUERY_K,
            "nprobes": QUERY_NPROBES,
            "refine_factor": QUERY_REFINE_FACTOR,
        }
    )

    return time.perf_counter() - start


def run_queries(
    datasets: List[lance.LanceDataset],
    queries: np.ndarray,
    assignments: List[Tuple[int, int]],
    num_workers: int = NUM_WORKERS,
    warmup: bool = False,
) -> List[float]:
    """
    Execute queries concurrently using ThreadPoolExecutor.

    Args:
        datasets: List of Lance datasets
        queries: Array of query vectors
        assignments: List of (dataset_idx, query_idx) tuples
        num_workers: Number of worker threads
        warmup: If True, discard timing results (warmup phase)

    Returns:
        List of query latencies (empty if warmup=True)
    """
    latencies = []
    desc = "  Warmup queries" if warmup else "  Timed queries"

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all queries
        futures = []
        for dataset_idx, query_idx in assignments:
            future = executor.submit(
                execute_query, dataset_idx, queries[query_idx], datasets
            )
            futures.append(future)

        # Collect results with progress bar
        with tqdm(total=len(futures), desc=desc, unit="queries") as pbar:
            for future in as_completed(futures):
                if not warmup:
                    latencies.append(future.result())
                pbar.update(1)

    return latencies


# ==================== STATISTICS ====================


def compute_statistics(latencies: List[float]) -> Dict[str, float]:
    """
    Compute latency statistics.

    Args:
        latencies: List of query latencies in seconds

    Returns:
        Dictionary with mean, std, min, max, and percentiles
    """
    arr = np.array(latencies)
    return {
        "mean": np.mean(arr),
        "std": np.std(arr),
        "min": np.min(arr),
        "max": np.max(arr),
        "p50": np.percentile(arr, 50),
        "p95": np.percentile(arr, 95),
        "p99": np.percentile(arr, 99),
    }


def compute_throughput(latencies: List[float]) -> float:
    """
    Compute throughput as queries per second.

    Args:
        latencies: List of query latencies in seconds

    Returns:
        Queries per second
    """
    return len(latencies) / sum(latencies)


# ==================== MAIN ====================


def main():
    """Main benchmark orchestration."""
    print("=" * 60)
    print("Lance Vector Search Benchmark")
    print("=" * 60)
    print("\nConfiguration:")
    print(f"  Datasets: {NUM_DATASETS}")
    print(f"  Rows per dataset: {ROWS_PER_DATASET:,}")
    print(f"  Vector dimensions: {VECTOR_DIM}")
    print(
        f"  Index: IVF_PQ (partitions={NUM_PARTITIONS}, subvectors={NUM_SUB_VECTORS})"
    )
    print(f"  Num queries: {NUM_QUERIES:,} ")
    print(
        f"  Query parameters: k={QUERY_K}, nprobes={QUERY_NPROBES}, refine_factor={QUERY_REFINE_FACTOR}"
    )
    print(f"  Worker threads: {NUM_WORKERS}")

    # Step 1: Create datasets
    print("\n" + "=" * 60)
    print("Step 1: Loading/Creating Datasets")
    print("=" * 60)

    datasets = []
    for i, path in enumerate(DATASET_PATHS, 1):
        print(f"\nDataset {i}/{NUM_DATASETS}: {path}")
        if dataset_exists(path, ROWS_PER_DATASET):
            print(f"  Dataset exists with {ROWS_PER_DATASET:,} rows - loading")
            ds = lance.dataset(path)
        else:
            print(f"  Dataset not found or has wrong row count - creating")
            ds = generate_dataset(path)
        datasets.append(ds)

    # Step 2: Create indices
    print("\n" + "=" * 60)
    print("Step 2: Loading/Creating Indices")
    print("=" * 60)

    for i, ds in enumerate(datasets, 1):
        print(f"\nIndex {i}/{NUM_DATASETS}...")
        if has_vector_index(ds, "vector"):
            print(f"  Vector index already exists - skipping")
        else:
            print(f"  Creating vector index...")
            start = time.perf_counter()
            create_index(ds)
            elapsed = time.perf_counter() - start
            print(f"  Done in {elapsed:.1f}s")

    # Step 3: Generate queries
    print("\n" + "=" * 60)
    print("Step 3: Generating Queries")
    print("=" * 60)
    queries = generate_queries()

    # Step 4: Create query assignments (round-robin distribution)
    print("\nCreating query assignments...")
    assignments = [(i % NUM_DATASETS, i) for i in range(NUM_QUERIES)]

    # Step 5: Warmup phase
    print("\n" + "=" * 60)
    print("Step 4: Warmup Phase")
    print("=" * 60)
    print(f"\nExecuting {NUM_QUERIES:,} queries...")
    run_queries(datasets, queries, assignments, NUM_WORKERS, warmup=True)

    # Step 6: Timed phase
    print("\n" + "=" * 60)
    print("Step 5: Timed Phase")
    print("=" * 60)
    print(f"\nExecuting {NUM_QUERIES:,} queries...")
    latencies = run_queries(datasets, queries, assignments, NUM_WORKERS, warmup=False)

    # Step 7: Compute and display results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    stats = compute_statistics(latencies)
    throughput = compute_throughput(latencies)

    print("\nLatency Statistics (seconds):")
    print(f"  Mean:   {stats['mean']:.6f}")
    print(f"  Std:    {stats['std']:.6f}")
    print(f"  Min:    {stats['min']:.6f}")
    print(f"  Max:    {stats['max']:.6f}")
    print(f"  p50:    {stats['p50']:.6f}")
    print(f"  p95:    {stats['p95']:.6f}")
    print(f"  p99:    {stats['p99']:.6f}")

    print(f"\nThroughput: {throughput:.2f} queries/sec")

    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
