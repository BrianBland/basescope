use std::fs;
use std::path::{Path, PathBuf};

use crate::domain::{ChunkData, ColumnarChunkData};
use crate::error::SpamscanError;

const ZSTD_LEVEL: i32 = 3;
const V1_MAGIC: [u8; 4] = *b"BSv1";

#[derive(Clone)]
pub struct Cache {
    data_dir: PathBuf,
}

impl Cache {
    pub fn new(data_dir: PathBuf) -> Result<Self, SpamscanError> {
        let chunks_dir = Self::chunks_dir(&data_dir);
        fs::create_dir_all(&chunks_dir)?;
        Ok(Self { data_dir })
    }

    pub fn chunk_path(&self, start_block: u64, end_block: u64) -> PathBuf {
        let filename = format!("{:012}_{:012}.bin.zst", start_block, end_block);
        Self::chunks_dir(&self.data_dir).join(filename)
    }

    pub fn has_chunk(&self, start_block: u64, end_block: u64) -> bool {
        self.chunk_path(start_block, end_block).exists()
    }

    pub fn load_chunk(&self, start_block: u64, end_block: u64) -> Result<ChunkData, SpamscanError> {
        let path = self.chunk_path(start_block, end_block);
        let bytes = fs::read(path)?;
        let decompressed = zstd::bulk::decompress(&bytes, 64 * 1024 * 1024)
            .map_err(|e| SpamscanError::Serialization(format!("zstd decompress: {e}")))?;
        if decompressed.len() < V1_MAGIC.len() || decompressed[..4] != V1_MAGIC {
            return Err(SpamscanError::Serialization(
                "unsupported chunk version".into(),
            ));
        }
        let col: ColumnarChunkData = bincode::deserialize(&decompressed[V1_MAGIC.len()..])
            .map_err(|e| SpamscanError::Serialization(e.to_string()))?;
        ChunkData::try_from(col).map_err(SpamscanError::Serialization)
    }

    pub fn save_chunk(&self, chunk: &ChunkData) -> Result<(), SpamscanError> {
        let path = self.chunk_path(chunk.start_block, chunk.end_block);
        let tmp_path = path.with_extension("zst.tmp");
        let columnar = ColumnarChunkData::from(chunk);
        let payload = bincode::serialize(&columnar)
            .map_err(|e| SpamscanError::Serialization(e.to_string()))?;
        let mut out = Vec::with_capacity(V1_MAGIC.len() + payload.len());
        out.extend_from_slice(&V1_MAGIC);
        out.extend_from_slice(&payload);
        let compressed = zstd::bulk::compress(&out, ZSTD_LEVEL)
            .map_err(|e| SpamscanError::Serialization(format!("zstd compress: {e}")))?;
        fs::write(&tmp_path, compressed)?;
        fs::rename(tmp_path, path)?;
        Ok(())
    }

    fn chunks_dir(base: &Path) -> PathBuf {
        base.join("base").join("chunks")
    }
}
