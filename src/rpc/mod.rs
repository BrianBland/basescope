use std::sync::Arc;
use std::time::Instant;

use alloy::consensus::Transaction;
use alloy::eips::BlockNumberOrTag;
use alloy::network::{AnyNetwork, BlockResponse, TransactionResponse};
use alloy::providers::{Provider, ProviderBuilder, RootProvider};
use parking_lot::Mutex;
use tracing::warn;
use url::Url;

use crate::domain::{BlockRecord, TxRecord};
use crate::error::SpamscanError;

const AI_INCREMENT: f64 = 0.05;
const MD_FACTOR: f64 = 0.5;
const HEAL_RATE: f64 = 0.01;
const HEAL_INTERVAL_SECS: f64 = 1.0;
const MIN_USABLE_SCORE: f64 = 0.1;

struct EndpointState {
    score: f64,
    last_update: Instant,
    url: String,
}

struct Scorer {
    endpoints: Vec<EndpointState>,
}

impl Scorer {
    fn new(urls: &[String]) -> Self {
        let now = Instant::now();
        Self {
            endpoints: urls
                .iter()
                .map(|url| EndpointState {
                    score: 1.0,
                    last_update: now,
                    url: url.clone(),
                })
                .collect(),
        }
    }

    fn heal_scores(&mut self) {
        let now = Instant::now();
        for ep in &mut self.endpoints {
            let elapsed = now.duration_since(ep.last_update).as_secs_f64();
            let ticks = (elapsed / HEAL_INTERVAL_SECS).floor();
            if ticks > 0.0 {
                ep.score = (ep.score + HEAL_RATE * ticks).min(1.0);
                ep.last_update = now;
            }
        }
    }

    fn pick_best(&mut self) -> usize {
        self.heal_scores();
        let mut best_usable_idx: Option<usize> = None;
        let mut best_usable_score = f64::NEG_INFINITY;
        let mut best_any_idx = 0;
        let mut best_any_score = f64::NEG_INFINITY;
        for (i, ep) in self.endpoints.iter().enumerate() {
            if ep.score > best_any_score {
                best_any_score = ep.score;
                best_any_idx = i;
            }
            if ep.score >= MIN_USABLE_SCORE && ep.score > best_usable_score {
                best_usable_score = ep.score;
                best_usable_idx = Some(i);
            }
        }
        best_usable_idx.unwrap_or(best_any_idx)
    }

    fn record_success(&mut self, idx: usize) {
        if let Some(ep) = self.endpoints.get_mut(idx) {
            ep.score = (ep.score + AI_INCREMENT).min(1.0);
            ep.last_update = Instant::now();
        }
    }

    fn record_failure(&mut self, idx: usize) {
        if let Some(ep) = self.endpoints.get_mut(idx) {
            ep.score *= MD_FACTOR;
            ep.last_update = Instant::now();
            warn!(
                endpoint = %ep.url,
                new_score = format!("{:.3}", ep.score),
                "endpoint score decreased"
            );
        }
    }

}

#[derive(Clone)]
pub struct RpcClient {
    providers: Arc<Vec<RootProvider<AnyNetwork>>>,
    scorer: Arc<Mutex<Scorer>>,
}

impl RpcClient {
    pub fn new(rpc_urls: &[String]) -> Result<Self, SpamscanError> {
        if rpc_urls.is_empty() {
            return Err(SpamscanError::Rpc("no rpc urls provided".into()));
        }
        let mut providers = Vec::with_capacity(rpc_urls.len());
        for rpc_url in rpc_urls {
            let url: Url = rpc_url
                .parse()
                .map_err(|e| SpamscanError::Rpc(format!("invalid rpc url {rpc_url}: {e}")))?;
            let provider = ProviderBuilder::new()
                .disable_recommended_fillers()
                .network::<AnyNetwork>()
                .connect_http(url);
            providers.push(provider);
        }
        let scorer = Scorer::new(rpc_urls);
        Ok(Self {
            providers: Arc::new(providers),
            scorer: Arc::new(Mutex::new(scorer)),
        })
    }

    fn pick_provider(&self) -> (usize, &RootProvider<AnyNetwork>) {
        let idx = self.scorer.lock().pick_best();
        (idx, &self.providers[idx])
    }

    pub async fn fetch_block(&self, block_number: u64) -> Result<BlockRecord, SpamscanError> {
        let (idx, provider) = self.pick_provider();
        let result = Self::do_fetch(provider, block_number).await;
        match &result {
            Ok(_) => self.scorer.lock().record_success(idx),
            Err(_) => self.scorer.lock().record_failure(idx),
        }
        result
    }

    async fn do_fetch(
        provider: &RootProvider<AnyNetwork>,
        block_number: u64,
    ) -> Result<BlockRecord, SpamscanError> {
        let block = provider
            .get_block_by_number(BlockNumberOrTag::Number(block_number))
            .full()
            .await
            .map_err(|e: alloy::transports::RpcError<alloy::transports::TransportErrorKind>| {
                SpamscanError::Rpc(e.to_string())
            })?
            .ok_or_else(|| SpamscanError::Rpc(format!("block {block_number} not found")))?;

        let header = block.header();
        let number = header.number;
        let timestamp = header.timestamp;
        let base_fee = header.base_fee_per_gas.unwrap_or(0) as u128;

        let txs = block
            .transactions()
            .as_transactions()
            .ok_or_else(|| SpamscanError::Rpc("block missing full transactions".into()))?;

        let mut transactions = Vec::with_capacity(txs.len());
        for tx in txs {
            let gas_used = tx.inner.inner.gas_limit();
            let max_priority_fee = tx.inner.inner.max_priority_fee_per_gas().unwrap_or(0);
            let max_fee = tx.inner.inner.max_fee_per_gas();
            let to_addr = tx.inner.inner.to();
            transactions.push(TxRecord {
                hash: tx.tx_hash().0,
                from: tx.from().0 .0,
                to: to_addr.map(|a| a.0 .0),
                gas_used,
                max_priority_fee,
                max_fee,
            });
        }

        Ok(BlockRecord {
            number,
            timestamp,
            base_fee,
            transactions,
        })
    }

    pub async fn fetch_block_with_retry(
        &self,
        block_number: u64,
        max_retries: u32,
    ) -> Result<BlockRecord, SpamscanError> {
        let mut last_err = None;
        for attempt in 0..=max_retries {
            match self.fetch_block(block_number).await {
                Ok(record) => return Ok(record),
                Err(err) => {
                    if attempt < max_retries {
                        let delay = std::time::Duration::from_millis(100 * 2u64.pow(attempt));
                        let delay = delay.min(std::time::Duration::from_secs(10));
                        warn!(
                            block_number,
                            attempt,
                            error = %err,
                            "retrying block fetch in {:?}", delay
                        );
                        tokio::time::sleep(delay).await;
                    }
                    last_err = Some(err);
                }
            }
        }
        Err(last_err.unwrap())
    }
}
