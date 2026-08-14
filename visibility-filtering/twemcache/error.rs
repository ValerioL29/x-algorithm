use thiserror::Error;

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum TwemcacheError {
    #[error("timeout")]
    Timeout,
    #[error("backpressure: too many in-flight requests")]
    Backpressure,
    #[error("unavailable")]
    Unavailable,
    #[error("io: {0}")]
    Io(String),
}

pub type Result<T> = std::result::Result<T, TwemcacheError>;
