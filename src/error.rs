use thiserror::Error;

#[derive(Error, Debug)]
pub enum SpamscanError {
    #[error("RPC error: {0}")]
    Rpc(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Pipeline cancelled")]
    Cancelled,
}
