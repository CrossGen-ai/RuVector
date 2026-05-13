use thiserror::Error;

#[derive(Debug, Error)]
pub enum NsgError {
    #[error("dataset is empty")]
    Empty,
    #[error("dimension mismatch: index={index}, query={query}")]
    DimMismatch { index: usize, query: usize },
    #[error("k must be in (0, n]; got k={k}, n={n}")]
    BadK { k: usize, n: usize },
    #[error("parameter `{name}` must be > 0")]
    BadParam { name: &'static str },
}

pub type Result<T> = std::result::Result<T, NsgError>;
