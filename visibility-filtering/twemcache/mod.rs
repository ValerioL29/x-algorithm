mod accrual;
pub(crate) mod client;
pub(crate) mod connection;
pub(crate) mod discovery;
pub(crate) mod error;
pub(crate) mod host_pool;
pub(crate) mod key;
mod metrics;
pub(crate) mod ring;

pub use client::{TwemcacheClient, default_connections_per_host, default_depth_cap};
pub use error::{Result, TwemcacheError};
pub use key::{Key, KeyError, Value};
