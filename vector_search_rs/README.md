# Lance Vector Search Benchmark (Rust)

A comprehensive benchmark for testing Lance vector search performance with concurrent query execution, implemented in Rust.

## Overview

This benchmark tests Lance's vector search capabilities with:

- **3 datasets** with 1 million rows each
- **768-dimensional float32 vectors** (common for text embeddings)
- **IVF_PQ index** with 256 partitions and 48 subvectors
- **10,000 queries** (warmup + timed)
- **Tokio async runtime** with 32 concurrent workers
- **Query parameters**: k=50, nprobes=1, refine_factor=10

## Requirements

- Rust 1.70 or higher
- ~6GB RAM for query vectors
- ~12GB disk space for datasets and indices
- Linux (for page cache management)

## Building

Build the benchmark in release mode:

```bash
cd vector_search_rs
cargo build --release
```

## Usage

Run the benchmark:

```bash
cargo run --release
```

### Customizing Dataset Paths

**Option 1: Environment Variable (Recommended)**

Set the `DATASET_PATHS` environment variable with comma-separated paths:

```bash
export DATASET_PATHS="./lance_dataset_0,./lance_dataset_1,./lance_dataset_2"
cargo run --release
```

Or inline:

```bash
DATASET_PATHS="path1,path2,path3" cargo run --release
```

**Option 2: Edit the Source**

Edit the `get_dataset_paths()` function in `src/main.rs`:

```rust
fn get_dataset_paths() -> Vec<String> {
    vec![
        "./lance_dataset_0".to_string(),
        "./lance_dataset_1".to_string(),
        "./lance_dataset_2".to_string(),
    ]
}
```

### Customizing Parameters

Other configurable constants in `src/main.rs`:

```rust
// Dataset parameters
const ROWS_PER_DATASET: usize = 1_000_000;
const VECTOR_DIM: usize = 768;

// Index parameters
const NUM_PARTITIONS: u32 = 256;
const NUM_SUB_VECTORS: usize = 48;

// Query parameters
const NUM_QUERIES: usize = 10_000;
const NUM_WORKERS: usize = 32;
const QUERY_K: usize = 50;
const QUERY_NPROBES: usize = 1;
const QUERY_REFINE_FACTOR: u32 = 10;
```

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

3. **Query Generation**: Pre-generates 10,000 random query vectors before timing starts

4. **Warmup Phase**: Executes 10,000 queries (distributed evenly across datasets) to warm up caches and stabilize the system

5. **Cache Drop**: Drops all dataset files from the Linux page cache using posix_fadvise

6. **Timed Phase**: Executes 10,000 queries with Tokio async runtime (32 concurrent workers) and collects latency measurements

7. **Statistics**: Computes comprehensive latency statistics and throughput

## Memory Usage

- **Datasets on disk**: ~9GB total (~3GB per dataset)
- **Indices on disk**: ~3GB total (~1GB per dataset)
- **Query vectors in RAM**: ~30MB (10,000 × 768 × 4 bytes)
- **Peak RAM**: ~1GB (during dataset generation and query execution)

## Dependencies

- **lance**: Lance columnar data format with vector search
- **tokio**: Async runtime for concurrent query execution
- **arrow**: Arrow data format for Lance
- **rand/rand_distr**: Random number generation for vectors
- **indicatif**: Progress bars
- **anyhow**: Error handling
- **walkdir**: Directory traversal for cache management

## Performance Tips

1. **Release Mode**: Always use `--release` flag for accurate benchmarks
2. **SSD Storage**: Store datasets on SSD for best performance
3. **Worker Tuning**: Adjust `NUM_WORKERS` based on your CPU cores
4. **Query Parameters**:
   - Increase `QUERY_NPROBES` for higher recall (slower)
   - Decrease `QUERY_REFINE_FACTOR` for faster queries (lower accuracy)

## Comparison with Python Version

This Rust implementation offers:

- **Native Performance**: Direct access to Lance Rust APIs without Python overhead
- **Lower Memory Usage**: More efficient memory management
- **Better Concurrency**: Tokio's async runtime for efficient task scheduling
- **Compiled Binary**: No Python interpreter needed for deployment

## Troubleshooting

### Build Errors

If you encounter build errors related to Lance dependencies:

```bash
# Ensure Lance is properly checked out
cd /home/pace/dev/lance
git pull
cd /home/pace/dev/one-million-iops/vector_search_rs
cargo clean
cargo build --release
```

### Out of Memory

If you encounter OOM errors during dataset generation:
1. Reduce `BATCH_SIZE` in `src/main.rs` (default: 100,000)
2. Reduce `ROWS_PER_DATASET` for smaller datasets

### Slow Performance

If queries are very slow:
- Check if datasets are on SSD (recommended) vs HDD
- Reduce `QUERY_REFINE_FACTOR` for faster but less accurate results
- Reduce `QUERY_NPROBES` to search fewer partitions
- Verify datasets have indices created (check Step 2 output)

## License

This benchmark is provided as-is for testing Lance vector search performance.
