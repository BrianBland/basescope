use alloy::primitives::Address;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

/// Block 0 timestamp on Base mainnet (2023-06-15T00:35:47Z).
pub const BASE_GENESIS_TIMESTAMP: u64 = 1686789347;
pub const BASE_BLOCK_TIME_SECS: u64 = 2;
pub const CHUNK_SIZE: u64 = 200;

pub fn timestamp_to_block(timestamp: u64) -> u64 {
    timestamp.saturating_sub(BASE_GENESIS_TIMESTAMP) / BASE_BLOCK_TIME_SECS
}

pub fn approx_head_block() -> u64 {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    timestamp_to_block(now)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FilterId(pub usize);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterKind {
    To(Address),
    From(Address),
    ToOrFrom(Address),
}

impl FilterKind {
    pub fn matches(&self, from: Address, to: Option<Address>) -> bool {
        match self {
            FilterKind::To(addr) => to == Some(*addr),
            FilterKind::From(addr) => from == *addr,
            FilterKind::ToOrFrom(addr) => from == *addr || to == Some(*addr),
        }
    }
}

impl std::fmt::Display for FilterKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FilterKind::To(addr) => write!(f, "to={}", addr),
            FilterKind::From(addr) => write!(f, "from={}", addr),
            FilterKind::ToOrFrom(addr) => write!(f, "to/from={}", addr),
        }
    }
}

impl FromStr for FilterKind {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (prefix, addr_str) = s.split_once(':').ok_or_else(|| {
            format!("expected `to:0x...`, `from:0x...`, or `addr:0x...`, got `{s}`")
        })?;

        let addr = Address::from_str(addr_str)
            .map_err(|e| format!("invalid address `{addr_str}`: {e}"))?;

        match prefix {
            "to" => Ok(FilterKind::To(addr)),
            "from" => Ok(FilterKind::From(addr)),
            "addr" => Ok(FilterKind::ToOrFrom(addr)),
            other => Err(format!(
                "unknown filter prefix `{other}`, expected `to`, `from`, or `addr`"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxFilter {
    pub id: FilterId,
    pub label: String,
    pub kind: FilterKind,
    pub enabled: bool,
    pub color_index: usize,
}

impl TxFilter {
    pub fn new(id: usize, kind: FilterKind, label: Option<String>) -> Self {
        let label = label.unwrap_or_else(|| kind.to_string());
        Self {
            id: FilterId(id),
            label,
            kind,
            enabled: true,
            color_index: id,
        }
    }
}

pub fn parse_filter(s: &str) -> Result<(FilterKind, Option<String>), String> {
    if let Some((label, rest)) = s.split_once('=') {
        let label = label.trim();
        if label.is_empty() {
            return Err("empty label before '='".to_string());
        }
        let kind = FilterKind::from_str(rest.trim())?;
        Ok((kind, Some(label.to_string())))
    } else {
        let kind = FilterKind::from_str(s)?;
        Ok((kind, None))
    }
}

#[derive(Debug, Clone)]
pub struct ScanSpec {
    pub start_block: u64,
    pub end_block: u64,
    pub filters: Vec<TxFilter>,
}

impl ScanSpec {
    pub fn chunk_ranges(&self) -> Vec<(u64, u64)> {
        let mut ranges = Vec::new();
        let mut start = self.start_block - (self.start_block % CHUNK_SIZE);
        while start <= self.end_block {
            let end = (start + CHUNK_SIZE - 1).min(self.end_block);
            ranges.push((start, end));
            start += CHUNK_SIZE;
        }
        ranges
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockRecord {
    pub number: u64,
    pub timestamp: u64,
    /// Wei.
    pub base_fee: u128,
    pub transactions: Vec<TxRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TxRecord {
    pub hash: [u8; 32],
    pub from: [u8; 20],
    pub to: Option<[u8; 20]>,
    pub gas_used: u64,
    /// Wei.
    pub max_priority_fee: u128,
    /// Wei.
    pub max_fee: u128,
}

impl TxRecord {
    pub fn from_addr(&self) -> Address {
        Address::from(self.from)
    }

    pub fn to_addr(&self) -> Option<Address> {
        self.to.map(Address::from)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkData {
    pub start_block: u64,
    pub end_block: u64,
    pub blocks: Vec<BlockRecord>,
}

#[derive(Debug, Clone)]
pub struct AnalysisSnapshot {
    /// (block_number, matching_tx_count) per filter.
    pub filter_series: HashMap<FilterId, Vec<(f64, f64)>>,
    /// (base_fee_gwei_bucket, matching_tx_count) per filter.
    pub filter_histograms: HashMap<FilterId, Vec<(f64, f64)>>,
    /// (block_number, base_fee_gwei) for all blocks — the bottom chart.
    pub base_fee_series: Vec<(f64, f64)>,
    /// Union of all enabled filters: (block_number, matching_tx_count).
    pub aggregate_series: Vec<(f64, f64)>,
    pub aggregate_histogram: Vec<(f64, f64)>,
    pub all_blocks_histogram: Vec<(f64, f64)>,
    pub blocks_fetched: usize,
    pub filters: Vec<TxFilter>,
    pub show_aggregate: bool,
}

impl AnalysisSnapshot {
    pub fn new(filters: &[TxFilter]) -> Self {
        let mut filter_series = HashMap::new();
        let mut filter_histograms = HashMap::new();
        for f in filters {
            filter_series.insert(f.id, Vec::new());
            filter_histograms.insert(f.id, Vec::new());
        }
        Self {
            filter_series,
            filter_histograms,
            base_fee_series: Vec::new(),
            aggregate_series: Vec::new(),
            aggregate_histogram: Vec::new(),
            all_blocks_histogram: Vec::new(),
            blocks_fetched: 0,
            filters: filters.to_vec(),
            show_aggregate: false,
        }
    }
}
