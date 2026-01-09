# Lance Vector Search Benchmark

A comprehensive benchmark for testing Lance vector search performance with concurrent query execution.

## Overview

This benchmark tests Lance's vector search capabilities with:

- **3 datasets** with 1 million rows each
- **768-dimensional float32 vectors** (common for text embeddings)
- **IVF_PQ index** with 256 partitions and 48 subvectors
- **2 million total queries** (1M warmup + 1M timed)
- **ThreadPoolExecutor with 8 workers** for concurrent execution
- **Query parameters**: k=50, nprobes=20, refine_factor=10

## Requirements

- Python 3.9 or higher
- ~6GB RAM for query vectors
- ~12GB disk space for datasets and indices
- [uv](https://github.com/astral-sh/uv) package manager

## Installation

1. Install dependencies using uv:

```bash
cd vector_search
uv sync
```

## Usage

Run the benchmark:

```bash
uv run benchmark.py
```

### Customizing Dataset Paths

Edit the `DATASET_PATHS` constant at the top of `benchmark.py`:

```python
DATASET_PATHS = [
    "./lance_dataset_0",
    "./lance_dataset_1",
    "./lance_dataset_2",
]
```

### Customizing Parameters

Other configurable parameters in `benchmark.py`:

```python
# Dataset parameters
ROWS_PER_DATASET = 1_000_000
VECTOR_DIM = 768

# Index parameters
NUM_PARTITIONS = 256
NUM_SUB_VECTORS = 48

# Query parameters
WARMUP_QUERIES = 1_000_000
TIMED_QUERIES = 1_000_000
NUM_WORKERS = 8
QUERY_K = 50
QUERY_NPROBES = 20
QUERY_REFINE_FACTOR = 10
```

## Expected Runtime

Total benchmark time: **~20-25 minutes**

Breakdown:
- Dataset creation: ~2-3 minutes (3 datasets)
- Index creation: ~7-9 minutes (2-3 minutes per dataset)
- Query generation: ~10 seconds
- Warmup phase: ~5 minutes (1M queries)
- Timed phase: ~5 minutes (1M queries)

## Output

The benchmark provides:

### Latency Statistics (in seconds)
- Mean, Standard Deviation
- Min, Max
- Percentiles: p50 (median), p95, p99

### Throughput
- Queries per second

Example output:

```
============================================================
BENCHMARK RESULTS
============================================================

Latency Statistics (seconds):
  Mean:   0.012345
  Std:    0.002456
  Min:    0.008901
  Max:    0.034567
  p50:    0.012123
  p95:    0.016789
  p99:    0.020123

Throughput: 81.03 queries/sec

============================================================
Benchmark Complete!
============================================================
```

## How It Works

1. **Dataset Generation**: Creates 3 Lance datasets with random 768-dim vectors in batches of 100,000 rows to manage memory efficiently

2. **Index Creation**: Builds IVF_PQ indices with 256 partitions and 48 subvectors on each dataset

3. **Query Generation**: Pre-generates 2M random query vectors before timing starts

4. **Warmup Phase**: Executes 1M queries (distributed evenly across datasets) to warm up caches and stabilize the system

5. **Timed Phase**: Executes 1M queries with ThreadPoolExecutor (8 workers) and collects latency measurements

6. **Statistics**: Computes comprehensive latency statistics and throughput

## Memory Usage

- **Datasets on disk**: ~9GB total (~3GB per dataset)
- **Indices on disk**: ~3GB total (~1GB per dataset)
- **Query vectors in RAM**: ~6GB (2M × 768 × 4 bytes)
- **Peak RAM**: ~6GB

## Dependencies

- **pylance** (>=0.19.0): Lance columnar data format with vector search
- **numpy** (>=1.22.0): Numerical operations and statistics
- **pyarrow** (>=14.0.0): Arrow data format for Lance
- **tqdm** (>=4.65.0): Progress bars

## Troubleshooting

### Out of Memory

If you encounter OOM errors:
1. Reduce `BATCH_SIZE` in `benchmark.py` (default: 100,000)
2. Reduce `ROWS_PER_DATASET` for smaller datasets
3. Close other memory-intensive applications

### Disk Space

Each dataset requires ~4GB (3GB data + 1GB index). Ensure you have at least 15GB free disk space.

### Slow Performance

If queries are very slow:
- Check if datasets are on SSD (recommended) vs HDD
- Reduce `QUERY_REFINE_FACTOR` for faster but less accurate results
- Reduce `QUERY_NPROBES` to search fewer partitions

## License

This benchmark is provided as-is for testing Lance vector search performance.
