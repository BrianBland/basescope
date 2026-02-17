use std::collections::{HashMap, HashSet};

use crate::domain::{AnalysisSnapshot, ChunkData, FilterId, TxFilter};

pub struct Analyzer {
    snapshot: AnalysisSnapshot,
    matches_by_filter: HashMap<FilterId, HashMap<u64, Vec<[u8; 32]>>>,
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
            matches_by_filter,
            base_fee_by_block: HashMap::new(),
            block_order: Vec::new(),
        }
    }

    pub fn process_chunk(&mut self, chunk: &ChunkData) {
        for block in &chunk.blocks {
            let block_number = block.number;
            let base_fee_gwei = block.base_fee as f64 / 1e9; // wei -> gwei
            self.base_fee_by_block.insert(block_number, base_fee_gwei);
            self.block_order.push(block_number);
            self.snapshot
                .base_fee_series
                .push((block_number as f64, base_fee_gwei));

            let mut aggregate_matches: HashSet<[u8; 32]> = HashSet::new();
            for filter in &self.snapshot.filters {
                let Some(entry) = self.matches_by_filter.get_mut(&filter.id) else {
                    continue;
                };
                let mut matches = Vec::new();
                for tx in &block.transactions {
                    if filter.kind.matches(tx.sender(), tx.to_addr()) {
                        matches.push(tx.hash);
                        if filter.enabled {
                            aggregate_matches.insert(tx.hash);
                        }
                    }
                }
                let count = matches.len() as f64;
                entry.insert(block_number, matches);
                if let Some(series) = self.snapshot.filter_series.get_mut(&filter.id) {
                    series.push((block_number as f64, count));
                }
            }

            let aggregate_count = aggregate_matches.len() as f64;
            self.snapshot
                .aggregate_series
                .push((block_number as f64, aggregate_count));
            self.snapshot.blocks_fetched += 1;
        }
    }

    pub fn snapshot(&self) -> AnalysisSnapshot {
        self.snapshot.clone()
    }

    pub fn toggle_filter(&mut self, filter_id: FilterId) {
        if let Some(filter) = self.snapshot.filters.iter_mut().find(|f| f.id == filter_id) {
            filter.enabled = !filter.enabled;
        }
        self.recompute_aggregate();
    }

    pub fn toggle_aggregate(&mut self) {
        self.snapshot.show_aggregate = !self.snapshot.show_aggregate;
    }

    fn recompute_aggregate(&mut self) {
        self.snapshot.aggregate_series.clear();

        let enabled_filters: Vec<FilterId> = self
            .snapshot
            .filters
            .iter()
            .filter(|f| f.enabled)
            .map(|f| f.id)
            .collect();

        for block_number in &self.block_order {
            let mut aggregate_matches: HashSet<[u8; 32]> = HashSet::new();
            for filter_id in &enabled_filters {
                if let Some(matches) = self
                    .matches_by_filter
                    .get(filter_id)
                    .and_then(|map| map.get(block_number))
                {
                    for hash in matches {
                        aggregate_matches.insert(*hash);
                    }
                }
            }

            let count = aggregate_matches.len() as f64;
            self.snapshot
                .aggregate_series
                .push((*block_number as f64, count));
        }
    }
}
