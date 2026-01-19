//! Lance Vector Search Benchmark

use anyhow::Result;
use arrow::array::{FixedSizeListArray, Float32Array, RecordBatchIterator};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::Parser;
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
use std::fs;
use std::path::Path;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::Instant;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::EnvFilter;

extern crate jemallocator;

#[global_allocator]
static GLOBAL: jemallocator::Jemalloc = jemallocator::Jemalloc;

// ==================== CONFIGURATION ====================

#[derive(Parser, Debug, Clone)]
#[command(name = "vector-search-benchmark")]
#[command(about = "Lance Vector Search Benchmark", long_about = None)]
struct Args {
    /// Dataset paths (comma-separated or multiple -d flags)
    #[arg(short = 'd', long = "dataset", value_delimiter = ',')]
    datasets: Option<Vec<String>>,

    /// Number of datasets to create/use (only used if --dataset not specified)
    #[arg(long, default_value_t = 3)]
    num_datasets: usize,

    /// Rows per dataset
    #[arg(long, default_value_t = 1_000_000)]
    rows_per_dataset: usize,

    /// Vector dimensions
    #[arg(long, default_value_t = 768)]
    vector_dim: usize,

    /// Batch size for writing
    #[arg(long, default_value_t = 100_000)]
    batch_size: usize,

    /// Number of IVF partitions
    #[arg(long, default_value_t = 256)]
    num_partitions: usize,

    /// Number of PQ sub-vectors
    #[arg(long, default_value_t = 48)]
    num_sub_vectors: usize,

    /// Number of queries to run
    #[arg(long, default_value_t = 2_000)]
    num_queries: usize,

    /// Number of runtime threads
    #[arg(long, default_value_t = 16)]
    num_runtimes: usize,

    /// Concurrent queries per runtime
    #[arg(long, default_value_t = 4)]
    concurrent_queries: usize,

    /// Top K results to return
    #[arg(short = 'k', long, default_value_t = 50)]
    query_k: usize,

    /// Number of probes for IVF search
    #[arg(long, default_value_t = 1)]
    nprobes: usize,

    /// Refine factor for PQ search
    #[arg(long, default_value_t = 10)]
    refine_factor: u32,

    /// Path to write tracing output (enables lance_io info tracing)
    #[arg(long)]
    tracing_output_path: Option<String>,
}

impl Args {
    fn get_dataset_paths(&self) -> Vec<String> {
        if let Some(ref paths) = self.datasets {
            paths.clone()
        } else {
            (1..=self.num_datasets)
                .map(|i| format!("file+uring:///var/data/{}/dataset.lance", i))
                .collect()
        }
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

async fn generate_dataset(
    uri: &str,
    num_rows: usize,
    dim: usize,
    batch_size: usize,
) -> Result<Dataset> {
    println!("\nGenerating dataset: {}", uri);

    let num_batches = num_rows / batch_size;
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
        let mut values: Vec<f32> = Vec::with_capacity(batch_size * dim);
        for _ in 0..batch_size * dim {
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
        max_rows_per_file: num_rows,
        ..Default::default()
    };

    let reader = RecordBatchIterator::new(batches, schema_clone);
    let dataset = Dataset::write(reader, uri, Some(params)).await?;
    pb_clone.finish();

    Ok(dataset)
}

// ==================== INDEX CREATION ====================

async fn create_index(
    dataset: &mut Dataset,
    num_partitions: usize,
    num_sub_vectors: usize,
) -> Result<()> {
    let ivf_params = IvfBuildParams {
        num_partitions: Some(num_partitions),
        ..Default::default()
    };

    let pq_params = PQBuildParams {
        num_sub_vectors,
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

async fn execute_query(
    dataset: Arc<Dataset>,
    query_vector: Vec<f32>,
    query_k: usize,
    nprobes: usize,
    refine_factor: u32,
) -> Result<f64> {
    let start = Instant::now();

    // Convert vector to Arrow array
    let query_array = Float32Array::from(query_vector);

    let batch = dataset
        .scan()
        .nearest("vector", &query_array, query_k)?
        .nprobes(nprobes)
        .refine(refine_factor)
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
    num_runtimes: usize,
    concurrent_queries: usize,
    query_k: usize,
    nprobes: usize,
    refine_factor: u32,
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

    for thread_idx in 0..num_runtimes {
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
                            let result =
                                execute_query(dataset, query, query_k, nprobes, refine_factor)
                                    .await;
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
                    .buffer_unordered(concurrent_queries);

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
    let args = Args::parse();

    // Set up tracing if output path is specified
    let _guard = if let Some(ref tracing_path) = args.tracing_output_path {
        let path = Path::new(tracing_path);
        let parent = path.parent().unwrap_or(Path::new("."));
        let filename = path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "trace.log".to_string());

        let file_appender = tracing_appender::rolling::never(parent, filename);
        let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

        let filter = EnvFilter::new("lance_io=info");

        tracing_subscriber::registry()
            .with(filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_ansi(false)
                    .with_writer(non_blocking),
            )
            .init();

        Some(guard)
    } else {
        env_logger::init();
        None
    };
    let dataset_paths = args.get_dataset_paths();
    let num_datasets = dataset_paths.len();

    // Create a single-threaded runtime for setup phase
    let setup_rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;

    println!("{}", "=".repeat(60));
    println!("Lance Vector Search Benchmark");
    println!("{}", "=".repeat(60));
    println!("\nConfiguration:");
    println!("  Datasets: {}", num_datasets);
    println!("  Rows per dataset: {}", args.rows_per_dataset);
    println!("  Vector dimensions: {}", args.vector_dim);
    println!(
        "  Index: IVF_PQ (partitions={}, subvectors={})",
        args.num_partitions, args.num_sub_vectors
    );
    println!("  Num queries: {}", args.num_queries);
    println!(
        "  Query parameters: k={}, nprobes={}, refine_factor={}",
        args.query_k, args.nprobes, args.refine_factor
    );
    println!("  Number of runtimes: {}", args.num_runtimes);
    println!(
        "  Concurrent queries per runtime: {}",
        args.concurrent_queries
    );

    // Step 1 & 2: Create datasets and indices (using setup runtime)
    let datasets = setup_rt.block_on(async {
        // Step 1: Create datasets
        println!("\n{}", "=".repeat(60));
        println!("Step 1: Loading/Creating Datasets");
        println!("{}", "=".repeat(60));

        let mut datasets_mut = Vec::new();
        for (i, path) in dataset_paths.iter().enumerate() {
            println!("\nDataset {}/{}: {}", i + 1, num_datasets, path);

            let dataset = if dataset_exists(path, args.rows_per_dataset).await {
                println!(
                    "  Dataset exists with {} rows - loading",
                    args.rows_per_dataset
                );
                Dataset::open(path).await?
            } else {
                println!("  Dataset not found or has wrong row count - creating");
                generate_dataset(
                    path,
                    args.rows_per_dataset,
                    args.vector_dim,
                    args.batch_size,
                )
                .await?
            };

            datasets_mut.push(dataset);
        }

        // Step 2: Create indices
        println!("\n{}", "=".repeat(60));
        println!("Step 2: Loading/Creating Indices");
        println!("{}", "=".repeat(60));

        for (i, dataset) in datasets_mut.iter_mut().enumerate() {
            println!("\nIndex {}/{}...", i + 1, num_datasets);

            if has_vector_index(dataset).await? {
                println!("  Vector index already exists - skipping");
            } else {
                println!("  Creating vector index...");
                let start = Instant::now();
                create_index(dataset, args.num_partitions, args.num_sub_vectors).await?;
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
    let queries = generate_queries(args.num_queries, args.vector_dim);

    // Step 4: Warmup phase
    println!("\n{}", "=".repeat(60));
    println!("Step 4: Warmup Phase");
    println!("{}", "=".repeat(60));
    println!("\nExecuting {} queries...", args.num_queries);
    run_queries(
        datasets.clone(),
        queries.clone(),
        true,
        args.num_runtimes,
        args.concurrent_queries,
        args.query_k,
        args.nprobes,
        args.refine_factor,
    )?;

    // Step 5: Drop cache
    println!("\n{}", "=".repeat(60));
    println!("Step 5: Dropping Page Cache");
    println!("{}", "=".repeat(60));
    println!("\nDropping dataset files from kernel page cache...");
    for (i, path) in dataset_paths.iter().enumerate() {
        println!("\n  Dataset {}/{}: {}", i + 1, num_datasets, path);
        drop_dataset_cache(path)?;
    }

    // Step 6: Timed phase
    println!("\n{}", "=".repeat(60));
    println!("Step 6: Timed Phase");
    println!("{}", "=".repeat(60));
    println!("\nExecuting {} queries...", args.num_queries);
    let start = Instant::now();
    let latencies = run_queries(
        datasets,
        queries,
        false,
        args.num_runtimes,
        args.concurrent_queries,
        args.query_k,
        args.nprobes,
        args.refine_factor,
    )?;
    let elapsed = start.elapsed();

    // Step 7: Compute and display results
    println!("\n{}", "=".repeat(60));
    println!("BENCHMARK RESULTS");
    println!("{}", "=".repeat(60));

    let stats = compute_statistics(&latencies);
    let throughput = args.num_queries as f64 / elapsed.as_secs_f64();

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
