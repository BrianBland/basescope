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
    pub fn sender(&self) -> Address {
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

/// Columnar serialization layout for ChunkData. Groups identical fields together
/// so zstd can exploit redundancy across transactions (repeated addresses,
/// similar gas values, etc.). Hashes are isolated so their entropy doesn't
/// pollute the compressor window for other fields.
#[derive(Debug, Serialize, Deserialize)]
pub struct ColumnarChunkData {
    pub start_block: u64,
    pub end_block: u64,
    pub block_numbers: Vec<u64>,
    pub block_timestamps: Vec<u64>,
    pub block_base_fees: Vec<u128>,
    /// Number of transactions in each block (needed to reconstruct nesting).
    pub block_tx_counts: Vec<u32>,
    pub tx_hashes: Vec<[u8; 32]>,
    pub tx_froms: Vec<[u8; 20]>,
    pub tx_tos: Vec<Option<[u8; 20]>>,
    pub tx_gas_used: Vec<u64>,
    pub tx_max_priority_fees: Vec<u128>,
    pub tx_max_fees: Vec<u128>,
}

impl From<&ChunkData> for ColumnarChunkData {
    fn from(chunk: &ChunkData) -> Self {
        let block_count = chunk.blocks.len();
        let tx_count: usize = chunk.blocks.iter().map(|b| b.transactions.len()).sum();

        let mut col = ColumnarChunkData {
            start_block: chunk.start_block,
            end_block: chunk.end_block,
            block_numbers: Vec::with_capacity(block_count),
            block_timestamps: Vec::with_capacity(block_count),
            block_base_fees: Vec::with_capacity(block_count),
            block_tx_counts: Vec::with_capacity(block_count),
            tx_hashes: Vec::with_capacity(tx_count),
            tx_froms: Vec::with_capacity(tx_count),
            tx_tos: Vec::with_capacity(tx_count),
            tx_gas_used: Vec::with_capacity(tx_count),
            tx_max_priority_fees: Vec::with_capacity(tx_count),
            tx_max_fees: Vec::with_capacity(tx_count),
        };

        for block in &chunk.blocks {
            col.block_numbers.push(block.number);
            col.block_timestamps.push(block.timestamp);
            col.block_base_fees.push(block.base_fee);
            col.block_tx_counts.push(block.transactions.len() as u32);
            for tx in &block.transactions {
                col.tx_hashes.push(tx.hash);
                col.tx_froms.push(tx.from);
                col.tx_tos.push(tx.to);
                col.tx_gas_used.push(tx.gas_used);
                col.tx_max_priority_fees.push(tx.max_priority_fee);
                col.tx_max_fees.push(tx.max_fee);
            }
        }

        col
    }
}

impl From<ColumnarChunkData> for ChunkData {
    fn from(col: ColumnarChunkData) -> Self {
        let mut blocks = Vec::with_capacity(col.block_numbers.len());
        let mut tx_offset = 0usize;

        for i in 0..col.block_numbers.len() {
            let tx_count = col.block_tx_counts[i] as usize;
            let mut transactions = Vec::with_capacity(tx_count);
            for j in tx_offset..tx_offset + tx_count {
                transactions.push(TxRecord {
                    hash: col.tx_hashes[j],
                    from: col.tx_froms[j],
                    to: col.tx_tos[j],
                    gas_used: col.tx_gas_used[j],
                    max_priority_fee: col.tx_max_priority_fees[j],
                    max_fee: col.tx_max_fees[j],
                });
            }
            tx_offset += tx_count;
            blocks.push(BlockRecord {
                number: col.block_numbers[i],
                timestamp: col.block_timestamps[i],
                base_fee: col.block_base_fees[i],
                transactions,
            });
        }

        ChunkData {
            start_block: col.start_block,
            end_block: col.end_block,
            blocks,
        }
    }
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
