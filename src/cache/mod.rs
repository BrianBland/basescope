use std::fs;
use std::path::{Path, PathBuf};

use crate::domain::ChunkData;
use crate::error::SpamscanError;

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
        let filename = format!("{:012}_{:012}.bin", start_block, end_block);
        Self::chunks_dir(&self.data_dir).join(filename)
    }

    pub fn has_chunk(&self, start_block: u64, end_block: u64) -> bool {
        self.chunk_path(start_block, end_block).exists()
    }

    pub fn load_chunk(&self, start_block: u64, end_block: u64) -> Result<ChunkData, SpamscanError> {
        let path = self.chunk_path(start_block, end_block);
        let bytes = fs::read(path)?;
        bincode::deserialize(&bytes).map_err(|e| SpamscanError::Serialization(e.to_string()))
    }

    pub fn save_chunk(&self, chunk: &ChunkData) -> Result<(), SpamscanError> {
        let path = self.chunk_path(chunk.start_block, chunk.end_block);
        let tmp_path = path.with_extension("bin.tmp");
        let bytes =
            bincode::serialize(chunk).map_err(|e| SpamscanError::Serialization(e.to_string()))?;
        fs::write(&tmp_path, bytes)?;
        fs::rename(tmp_path, path)?;
        Ok(())
    }

    fn chunks_dir(base: &Path) -> PathBuf {
        base.join("base").join("chunks")
    }
}
