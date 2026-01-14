//! Lance Vector Search Benchmark
//!
//! Benchmarks Lance vector search performance with:
//! - 3 datasets with 1M rows each
//! - 768-dimensional float32 vectors
//! - IVF_PQ index with 256 partitions and 48 subvectors
//! - 10,000 queries
//! - Multiple thread-locked current-thread runtimes with MPMC queue

use anyhow::Result;
use arrow::array::{FixedSizeListArray, Float32Array, RecordBatchIterator};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use crossbeam_channel::{bounded, Receiver, Sender};
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use lance::dataset::{Dataset, WriteMode, WriteParams};
use lance::index::vector::VectorIndexParams;
use lance_index::vector::ivf::IvfBuildParams;
use lance_index::vector::pq::PQBuildParams;
use lance_index::{DatasetIndexExt, IndexType};
use lance_linalg::distance::MetricType;
use rand_distr::{Distribution, StandardNormal};
use std::env;
use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;

extern crate jemallocator;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// ==================== CONFIGURATION ====================

const NUM_DATASETS: usize = 3;
const ROWS_PER_DATASET: usize = 1_000_000;
const VECTOR_DIM: usize = 768;
const BATCH_SIZE: usize = 100_000;

// Index parameters
const NUM_PARTITIONS: usize = 256;
const NUM_SUB_VECTORS: usize = 48;

// Query parameters
const NUM_QUERIES: usize = 2_000;
const NUM_RUNTIMES: usize = 16;
const CONCURRENT_QUERIES: usize = 4;
const QUERY_K: usize = 50;
const QUERY_NPROBES: usize = 1;
const QUERY_REFINE_FACTOR: u32 = 10;

// Default dataset paths
fn get_dataset_paths() -> Vec<String> {
    if let Ok(paths) = env::var("DATASET_PATHS") {
        paths.split(',').map(|s| s.trim().to_string()).collect()
    } else {
        vec![
            "file+uring:///var/data/one/dataset.lance".to_string(),
            "file+uring:///var/data/two/dataset.lance".to_string(),
            "file+uring:///var/data/three/dataset.lance".to_string(),
        ]
    }
}

// ==================== DATASET GENERATION ====================

async fn dataset_exists(uri: &str, expected_rows: usize) -> bool {
    if let Ok(dataset) = Dataset::open(uri).await {
        if let Ok(count) = dataset.count_rows(None).await {
            return count == expected_rows;
        }
    }
    false
}

async fn has_vector_index(dataset: &Dataset) -> Result<bool> {
    let indices = dataset.load_indices().await?;
    Ok(!indices.is_empty())
}

async fn generate_dataset(uri: &str, num_rows: usize, dim: usize) -> Result<Dataset> {
    println!("\nGenerating dataset: {}", uri);

    let num_batches = num_rows / BATCH_SIZE;
    assert!(num_batches > 0, "Number of batches must be greater than 0");
    let pb = ProgressBar::new(num_batches as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  Writing batches [{bar:40}] {pos}/{len}")
            .unwrap(),
    );

    let schema = Arc::new(Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
        ),
        true,
    )]));

    let schema_clone = schema.clone();
    let pb_clone = pb.clone();

    let batches = (0..num_batches).map(move |_| {
        let mut rng = rand::thread_rng();
        let mut values: Vec<f32> = Vec::with_capacity(BATCH_SIZE * dim);
        for _ in 0..BATCH_SIZE * dim {
            values.push(StandardNormal.sample(&mut rng));
        }
        let values_array = Float32Array::from(values);
        let list_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(values_array),
            None,
        );
        pb.inc(1);

        RecordBatch::try_new(schema.clone(), vec![Arc::new(list_array)])
    });

    let params = WriteParams {
        mode: WriteMode::Create,
        max_rows_per_file: ROWS_PER_DATASET,
        ..Default::default()
    };

    let reader = RecordBatchIterator::new(batches, schema_clone);
    let dataset = Dataset::write(reader, uri, Some(params)).await?;
    pb_clone.finish();

    Ok(dataset)
}

// ==================== INDEX CREATION ====================

async fn create_index(dataset: &mut Dataset) -> Result<()> {
    let ivf_params = IvfBuildParams {
        num_partitions: Some(NUM_PARTITIONS),
        ..Default::default()
    };

    let pq_params = PQBuildParams {
        num_sub_vectors: NUM_SUB_VECTORS,
        ..Default::default()
    };

    let index_params = VectorIndexParams::with_ivf_pq_params(MetricType::L2, ivf_params, pq_params);

    dataset
        .create_index(
            &["vector"],
            IndexType::Vector,
            Some("vector_idx".to_string()),
            &index_params,
            true,
        )
        .await?;

    Ok(())
}

// ==================== CACHE MANAGEMENT ====================

fn drop_file_cache(file_path: &Path) -> Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::io::AsRawFd;

        const POSIX_FADV_DONTNEED: libc::c_int = 4;

        let file = fs::File::open(file_path)?;
        let fd = file.as_raw_fd();
        let metadata = file.metadata()?;
        let file_size = metadata.len() as i64;

        unsafe {
            libc::posix_fadvise(fd, 0, file_size, POSIX_FADV_DONTNEED);
        }
    }

    Ok(())
}

fn drop_dataset_cache(dataset_uri: &str) -> Result<()> {
    // Strip the URI scheme to get the actual file path
    let path = if dataset_uri.contains("://") {
        dataset_uri.split("://").nth(1).unwrap()
    } else {
        dataset_uri
    };

    if !Path::new(path).exists() {
        println!("    Warning: Dataset path does not exist: {}", path);
        return Ok(());
    }

    let mut file_count = 0;
    let mut total_size = 0u64;

    for entry in walkdir::WalkDir::new(path) {
        let entry = entry?;
        if entry.file_type().is_file() {
            if let Ok(metadata) = entry.metadata() {
                total_size += metadata.len();
                let _ = drop_file_cache(entry.path());
                file_count += 1;
            }
        }
    }

    println!(
        "    Dropped {} files ({:.2} GB) from cache",
        file_count,
        total_size as f64 / 1024.0 / 1024.0 / 1024.0
    );

    Ok(())
}

// ==================== QUERY GENERATION ====================

fn generate_queries(num_queries: usize, dim: usize) -> Vec<Vec<f32>> {
    println!("\nGenerating {} query vectors...", num_queries);
    let start = Instant::now();

    let mut rng = rand::thread_rng();
    let mut queries = Vec::with_capacity(num_queries);

    for _ in 0..num_queries {
        let mut query = Vec::with_capacity(dim);
        for _ in 0..dim {
            query.push(StandardNormal.sample(&mut rng));
        }
        queries.push(query);
    }

    let elapsed = start.elapsed();
    println!("  Done in {:.2}s", elapsed.as_secs_f64());

    queries
}

static ROW_COUNTER: AtomicUsize = AtomicUsize::new(0);

// ==================== QUERY EXECUTION ====================

async fn execute_query(dataset: Arc<Dataset>, query_vector: Vec<f32>) -> Result<f64> {
    let start = Instant::now();

    // Convert vector to Arrow array
    let query_array = Float32Array::from(query_vector);

    let batch = dataset
        .scan()
        .nearest("vector", &query_array, QUERY_K)?
        .nprobes(QUERY_NPROBES)
        .refine(QUERY_REFINE_FACTOR)
        .try_into_batch()
        .await?;

    ROW_COUNTER.fetch_add(batch.num_rows(), std::sync::atomic::Ordering::Relaxed);

    Ok(start.elapsed().as_secs_f64())
}

// Query task: (dataset_idx, query_vector)
type QueryTask = (usize, Vec<f32>);

fn run_queries(
    datasets: Vec<Arc<Dataset>>,
    queries: Vec<Vec<f32>>,
    warmup: bool,
) -> Result<Vec<f64>> {
    let desc = if warmup {
        "Warmup queries"
    } else {
        "Timed queries"
    };
    let pb = ProgressBar::new(queries.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template(&format!("  {} [{{bar:40}}] {{pos}}/{{len}}", desc))
            .unwrap(),
    );

    let num_datasets = datasets.len();

    // Create MPMC channel for query tasks
    let (tx, rx): (Sender<QueryTask>, Receiver<QueryTask>) = bounded(queries.len());

    // Send all queries to the channel
    for (i, query) in queries.into_iter().enumerate() {
        let dataset_idx = i % num_datasets;
        tx.send((dataset_idx, query))?;
    }
    drop(tx); // Close the sender so threads know when to stop

    // Spawn worker threads
    let mut handles = Vec::new();
    let latencies = Arc::new(std::sync::Mutex::new(Vec::new()));

    for thread_idx in 0..NUM_RUNTIMES {
        let rx = rx.clone();
        let datasets = datasets.clone();
        let pb = pb.clone();
        let latencies = latencies.clone();

        let handle = std::thread::spawn(move || {
            // Create a current-thread runtime for this thread
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();

            rt.block_on(async move {
                // Process queries from the queue with concurrency control
                let query_stream = stream::iter(std::iter::from_fn(|| rx.recv().ok()))
                    .map(|(dataset_idx, query)| {
                        let dataset = datasets[dataset_idx].clone();
                        let pb = pb.clone();
                        let latencies = latencies.clone();

                        tokio::task::spawn(async move {
                            let result = execute_query(dataset, query).await;
                            pb.inc(1);

                            let latency = result.unwrap_or_else(|e| {
                                eprintln!("Query failed in thread {}: {:?}", thread_idx, e);
                                0.0f64
                            });

                            if !warmup {
                                latencies.lock().unwrap().push(latency);
                            }
                        })
                    })
                    .buffer_unordered(CONCURRENT_QUERIES);

                // Collect all results
                query_stream
                    .for_each(|result| async {
                        if let Err(e) = result {
                            eprintln!("Query failed in thread {}: {:?}", thread_idx, e);
                        }
                    })
                    .await;
            });
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for handle in handles {
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("Thread panicked"))?;
    }

    pb.finish();

    let latencies = Arc::try_unwrap(latencies).unwrap().into_inner().unwrap();

    Ok(latencies)
}

// ==================== STATISTICS ====================

struct Statistics {
    mean: f64,
    std: f64,
    min: f64,
    max: f64,
    p50: f64,
    p95: f64,
    p99: f64,
}

fn compute_statistics(latencies: &[f64]) -> Statistics {
    let n = latencies.len() as f64;
    let mean = latencies.iter().sum::<f64>() / n;

    let variance = latencies.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = variance.sqrt();

    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    let p50 = sorted[(n * 0.50) as usize];
    let p95 = sorted[(n * 0.95) as usize];
    let p99 = sorted[(n * 0.99) as usize];

    Statistics {
        mean,
        std,
        min,
        max,
        p50,
        p95,
        p99,
    }
}

// ==================== MAIN ====================

fn main() -> Result<()> {
    env_logger::init();

    // Create a single-threaded runtime for setup phase
    let setup_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    println!("{}", "=".repeat(60));
    println!("Lance Vector Search Benchmark");
    println!("{}", "=".repeat(60));
    println!("\nConfiguration:");
    println!("  Datasets: {}", NUM_DATASETS);
    println!("  Rows per dataset: {}", ROWS_PER_DATASET);
    println!("  Vector dimensions: {}", VECTOR_DIM);
    println!(
        "  Index: IVF_PQ (partitions={}, subvectors={})",
        NUM_PARTITIONS, NUM_SUB_VECTORS
    );
    println!("  Num queries: {}", NUM_QUERIES);
    println!(
        "  Query parameters: k={}, nprobes={}, refine_factor={}",
        QUERY_K, QUERY_NPROBES, QUERY_REFINE_FACTOR
    );
    println!("  Number of runtimes: {}", NUM_RUNTIMES);
    println!("  Concurrent queries per runtime: {}", CONCURRENT_QUERIES);

    let dataset_paths = get_dataset_paths();

    // Step 1 & 2: Create datasets and indices (using setup runtime)
    let datasets = setup_rt.block_on(async {
        // Step 1: Create datasets
        println!("\n{}", "=".repeat(60));
        println!("Step 1: Loading/Creating Datasets");
        println!("{}", "=".repeat(60));

        let mut datasets_mut = Vec::new();
        for (i, path) in dataset_paths.iter().enumerate() {
            println!("\nDataset {}/{}: {}", i + 1, NUM_DATASETS, path);

            let dataset = if dataset_exists(path, ROWS_PER_DATASET).await {
                println!("  Dataset exists with {} rows - loading", ROWS_PER_DATASET);
                Dataset::open(path).await?
            } else {
                println!("  Dataset not found or has wrong row count - creating");
                generate_dataset(path, ROWS_PER_DATASET, VECTOR_DIM).await?
            };

            datasets_mut.push(dataset);
        }

        // Step 2: Create indices
        println!("\n{}", "=".repeat(60));
        println!("Step 2: Loading/Creating Indices");
        println!("{}", "=".repeat(60));

        for (i, dataset) in datasets_mut.iter_mut().enumerate() {
            println!("\nIndex {}/{}...", i + 1, NUM_DATASETS);

            if has_vector_index(dataset).await? {
                println!("  Vector index already exists - skipping");
            } else {
                println!("  Creating vector index...");
                let start = Instant::now();
                create_index(dataset).await?;
                let elapsed = start.elapsed();
                println!("  Done in {:.1}s", elapsed.as_secs_f64());
            }
        }

        // Convert to Arc for concurrent access
        let datasets: Vec<Arc<Dataset>> = datasets_mut.into_iter().map(Arc::new).collect();
        Ok::<_, anyhow::Error>(datasets)
    })?;

    // Step 3: Generate queries
    println!("\n{}", "=".repeat(60));
    println!("Step 3: Generating Queries");
    println!("{}", "=".repeat(60));
    let queries = generate_queries(NUM_QUERIES, VECTOR_DIM);

    // Step 4: Warmup phase
    println!("\n{}", "=".repeat(60));
    println!("Step 4: Warmup Phase");
    println!("{}", "=".repeat(60));
    println!("\nExecuting {} queries...", NUM_QUERIES);
    run_queries(datasets.clone(), queries.clone(), true)?;

    // Step 5: Drop cache
    println!("\n{}", "=".repeat(60));
    println!("Step 5: Dropping Page Cache");
    println!("{}", "=".repeat(60));
    println!("\nDropping dataset files from kernel page cache...");
    for (i, path) in dataset_paths.iter().enumerate() {
        println!("\n  Dataset {}/{}: {}", i + 1, NUM_DATASETS, path);
        drop_dataset_cache(path)?;
    }

    // Step 6: Timed phase
    println!("\n{}", "=".repeat(60));
    println!("Step 6: Timed Phase");
    println!("{}", "=".repeat(60));
    println!("\nExecuting {} queries...", NUM_QUERIES);
    let start = Instant::now();
    let latencies = run_queries(datasets, queries, false)?;
    let elapsed = start.elapsed();

    // Step 7: Compute and display results
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK RESULTS");
    println!("{}", "=".repeat(60));

    let stats = compute_statistics(&latencies);
    let throughput = NUM_QUERIES as f64 / elapsed.as_secs_f64();

    println!("\nLatency Statistics (seconds):");
    println!("  Mean:   {:.6}", stats.mean);
    println!("  Std:    {:.6}", stats.std);
    println!("  Min:    {:.6}", stats.min);
    println!("  Max:    {:.6}", stats.max);
    println!("  p50:    {:.6}", stats.p50);
    println!("  p95:    {:.6}", stats.p95);
    println!("  p99:    {:.6}", stats.p99);

    println!("\nThroughput: {:.2} queries/sec", throughput);

    println!("\n{}", "=".repeat(60));
    println!("Benchmark Complete!");
    println!("{}", "=".repeat(60));

    println!(
        "  Total rows scanned: {}",
        ROW_COUNTER.load(std::sync::atomic::Ordering::Relaxed)
    );

    Ok(())
}
