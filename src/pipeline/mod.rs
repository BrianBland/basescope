use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tracing::error;

use crate::cache::Cache;
use crate::domain::{ChunkData, ScanSpec};
use crate::error::BasescopeError;
use crate::rpc::RpcClient;

fn find_contiguous_anchor(chunk_ranges: &[(u64, u64)], cache: &Cache) -> i64 {
    let mut best_start = 0usize;
    let mut best_len = 0usize;
    let mut run_start = 0usize;
    let mut run_len = 0usize;

    for (i, (start, end)) in chunk_ranges.iter().enumerate() {
        if cache.has_chunk(*start, *end) {
            if run_len == 0 {
                run_start = i;
            }
            run_len += 1;
            if run_len > best_len {
                best_len = run_len;
                best_start = run_start;
            }
        } else {
            run_len = 0;
        }
    }

    if best_len == 0 {
        return chunk_ranges.first().map(|(s, _)| *s as i64).unwrap_or(0);
    }

    let mid_idx = best_start + best_len / 2;
    let (s, e) = chunk_ranges[mid_idx];
    ((s + e) / 2) as i64
}

#[derive(Debug, Clone)]
pub enum PipelineEvent {
    ChunkCached(ChunkData),
    ChunkStarted { start: u64 },
    ChunkProgress { start: u64, fetched: u64, total: u64 },
    ChunkComplete(ChunkData),
    ChunkFailed { start: u64, end: u64, error: String },
    Done,
}

#[derive(Clone)]
pub struct Pipeline {
    rpc_client: RpcClient,
    cache: Cache,
    concurrency: usize,
    cancel_token: CancellationToken,
}

impl Pipeline {
    pub fn new(rpc_client: RpcClient, cache: Cache, concurrency: usize) -> Self {
        Self {
            rpc_client,
            cache,
            concurrency: concurrency.max(1),
            cancel_token: CancellationToken::new(),
        }
    }

    pub async fn run(
        &self,
        spec: &ScanSpec,
        event_tx: mpsc::UnboundedSender<PipelineEvent>,
    ) -> Result<(), BasescopeError> {
        let chunk_ranges = spec.chunk_ranges();

        for (start, end) in chunk_ranges.iter().copied() {
            if self.cache.has_chunk(start, end) {
                match self.cache.load_chunk(start, end) {
                    Ok(chunk) => {
                        let _ = event_tx.send(PipelineEvent::ChunkCached(chunk));
                    }
                    Err(err) => {
                        let _ = event_tx.send(PipelineEvent::ChunkFailed {
                            start,
                            end,
                            error: err.to_string(),
                        });
                    }
                }
            }
        }

        let mut uncached: Vec<(u64, u64)> = chunk_ranges
            .iter()
            .copied()
            .filter(|(start, end)| !self.cache.has_chunk(*start, *end))
            .collect();
        if uncached.is_empty() {
            let _ = event_tx.send(PipelineEvent::Done);
            return Ok(());
        }

        let anchor = find_contiguous_anchor(&chunk_ranges, &self.cache);
        uncached.sort_by_key(|(start, _)| {
            let mid = *start as i64;
            (mid - anchor).unsigned_abs()
        });

        let semaphore = Arc::new(Semaphore::new(self.concurrency));

        let mut collector_handles: Vec<JoinHandle<()>> = Vec::with_capacity(uncached.len());

        for (start, end) in &uncached {
            let _ = event_tx.send(PipelineEvent::ChunkStarted {
                start: *start,
            });
            let total = end - start + 1;
            let fetched = Arc::new(AtomicU64::new(0));
            let mut block_handles = Vec::with_capacity(total as usize);

            for block_number in *start..=*end {
                let permit = semaphore
                    .clone()
                    .acquire_owned()
                    .await
                    .map_err(|_| BasescopeError::Cancelled)?;
                let rpc = self.rpc_client.clone();
                let cancel = self.cancel_token.clone();
                let tx = event_tx.clone();
                let fetched = fetched.clone();
                let chunk_start = *start;

                let handle = tokio::spawn(async move {
                    let _permit = permit;
                    if cancel.is_cancelled() {
                        return Err(BasescopeError::Cancelled);
                    }
                    let block = rpc.fetch_block_with_retry(block_number, 5).await?;
                    let done = fetched.fetch_add(1, Ordering::Relaxed) + 1;
                    let _ = tx.send(PipelineEvent::ChunkProgress {
                        start: chunk_start,
                        fetched: done,
                        total,
                    });
                    Ok(block)
                });

                block_handles.push(handle);
            }

            let tx = event_tx.clone();
            let cache = self.cache.clone();
            let start = *start;
            let end = *end;

            let collector = tokio::spawn(async move {
                let mut blocks = Vec::with_capacity(block_handles.len());
                let mut chunk_err = None;

                for handle in block_handles {
                    match handle.await {
                        Ok(Ok(block)) => blocks.push(block),
                        Ok(Err(err)) => {
                            chunk_err = Some(err.to_string());
                            break;
                        }
                        Err(err) => {
                            chunk_err = Some(format!("task join failed: {err}"));
                            break;
                        }
                    }
                }

                if let Some(err) = chunk_err {
                    let _ = tx.send(PipelineEvent::ChunkFailed {
                        start,
                        end,
                        error: err,
                    });
                    return;
                }

                blocks.sort_by_key(|b| b.number);
                let chunk = ChunkData {
                    start_block: start,
                    end_block: end,
                    blocks,
                };
                if let Err(err) = cache.save_chunk(&chunk) {
                    error!(%err, "failed to cache chunk {start}-{end}");
                }
                let _ = tx.send(PipelineEvent::ChunkComplete(chunk));
            });

            collector_handles.push(collector);
        }

        for handle in collector_handles {
            let _ = handle.await;
        }

        let _ = event_tx.send(PipelineEvent::Done);
        Ok(())
    }
}
