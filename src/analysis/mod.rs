use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::domain::{AnalysisSnapshot, ChunkData, FilterId, TxFilter};

const WEI_PER_GWEI: f64 = 1e9;

/// Lightweight record of a matched transaction for aggregate deduplication.
struct MatchedTx {
    hash: [u8; 32],
    gas_used: u64,
    rlp_size: u64,
}

pub struct Analyzer {
    snapshot: AnalysisSnapshot,
    cached: Option<Arc<AnalysisSnapshot>>,
    matches_by_filter: HashMap<FilterId, HashMap<u64, Vec<MatchedTx>>>,
    base_fee_by_block: HashMap<u64, f64>,
    block_order: Vec<u64>,
}

impl Analyzer {
    pub fn new(filters: &[TxFilter]) -> Self {
        let mut matches_by_filter = HashMap::new();
        for filter in filters {
            matches_by_filter.insert(filter.id, HashMap::new());
        }
        Self {
            snapshot: AnalysisSnapshot::new(filters),
            cached: None,
            matches_by_filter,
            base_fee_by_block: HashMap::new(),
            block_order: Vec::new(),
        }
    }

    pub fn process_chunk(&mut self, chunk: &ChunkData) {
        for block in &chunk.blocks {
            let block_number = block.number;
            let base_fee_gwei = block.base_fee as f64 / WEI_PER_GWEI;
            self.base_fee_by_block.insert(block_number, base_fee_gwei);
            self.block_order.push(block_number);
            self.snapshot
                .base_fee_series
                .push((block_number as f64, base_fee_gwei));

            // Compute per-block totals for gas and tx size.
            let block_gas: u64 = block.transactions.iter().map(|tx| tx.gas_used).sum();
            let block_tx_size: u64 = block.transactions.iter().map(|tx| tx.rlp_size).sum();
            self.snapshot
                .block_gas_series
                .push((block_number as f64, block_gas as f64));
            self.snapshot
                .block_tx_size_series
                .push((block_number as f64, block_tx_size as f64));

            let mut aggregate_hashes: HashSet<[u8; 32]> = HashSet::new();
            let mut aggregate_gas: u64 = 0;
            let mut aggregate_tx_size: u64 = 0;

            for filter in &self.snapshot.filters {
                let Some(entry) = self.matches_by_filter.get_mut(&filter.id) else {
                    continue;
                };
                let mut matches = Vec::new();
                let mut filter_gas: u64 = 0;
                let mut filter_tx_size: u64 = 0;
                for tx in &block.transactions {
                    if filter.kind.matches(tx.sender(), tx.to_addr()) {
                        filter_gas += tx.gas_used;
                        filter_tx_size += tx.rlp_size;
                        if filter.enabled && aggregate_hashes.insert(tx.hash) {
                            aggregate_gas += tx.gas_used;
                            aggregate_tx_size += tx.rlp_size;
                        }
                        matches.push(MatchedTx {
                            hash: tx.hash,
                            gas_used: tx.gas_used,
                            rlp_size: tx.rlp_size,
                        });
                    }
                }
                let count = matches.len() as f64;
                entry.insert(block_number, matches);
                if let Some(series) = self.snapshot.filter_series.get_mut(&filter.id) {
                    series.push((block_number as f64, count));
                }
                if let Some(series) = self.snapshot.gas_series.get_mut(&filter.id) {
                    series.push((block_number as f64, filter_gas as f64));
                }
                if let Some(series) = self.snapshot.tx_size_series.get_mut(&filter.id) {
                    series.push((block_number as f64, filter_tx_size as f64));
                }
            }

            let aggregate_count = aggregate_hashes.len() as f64;
            self.snapshot
                .aggregate_series
                .push((block_number as f64, aggregate_count));
            self.snapshot
                .aggregate_gas_series
                .push((block_number as f64, aggregate_gas as f64));
            self.snapshot
                .aggregate_tx_size_series
                .push((block_number as f64, aggregate_tx_size as f64));
            self.snapshot.blocks_fetched += 1;
        }
        self.cached = None;
    }

    pub fn snapshot(&mut self) -> Arc<AnalysisSnapshot> {
        if let Some(ref cached) = self.cached {
            return Arc::clone(cached);
        }
        let arc = Arc::new(self.snapshot.clone());
        self.cached = Some(Arc::clone(&arc));
        arc
    }

    pub fn toggle_filter(&mut self, filter_id: FilterId) {
        if let Some(filter) = self.snapshot.filters.iter_mut().find(|f| f.id == filter_id) {
            filter.enabled = !filter.enabled;
        }
        self.recompute_aggregate();
        self.cached = None;
    }

    pub fn toggle_aggregate(&mut self) {
        self.snapshot.show_aggregate = !self.snapshot.show_aggregate;
        self.cached = None;
    }

    fn recompute_aggregate(&mut self) {
        self.snapshot.aggregate_series.clear();
        self.snapshot.aggregate_gas_series.clear();
        self.snapshot.aggregate_tx_size_series.clear();

        let enabled_filters: Vec<FilterId> = self
            .snapshot
            .filters
            .iter()
            .filter(|f| f.enabled)
            .map(|f| f.id)
            .collect();

        for block_number in &self.block_order {
            let mut aggregate_hashes: HashSet<[u8; 32]> = HashSet::new();
            let mut aggregate_gas: u64 = 0;
            let mut aggregate_tx_size: u64 = 0;

            for filter_id in &enabled_filters {
                if let Some(matches) = self
                    .matches_by_filter
                    .get(filter_id)
                    .and_then(|map| map.get(block_number))
                {
                    for mtx in matches {
                        if aggregate_hashes.insert(mtx.hash) {
                            aggregate_gas += mtx.gas_used;
                            aggregate_tx_size += mtx.rlp_size;
                        }
                    }
                }
            }

            let count = aggregate_hashes.len() as f64;
            self.snapshot
                .aggregate_series
                .push((*block_number as f64, count));
            self.snapshot
                .aggregate_gas_series
                .push((*block_number as f64, aggregate_gas as f64));
            self.snapshot
                .aggregate_tx_size_series
                .push((*block_number as f64, aggregate_tx_size as f64));
        }
    }
}
