use std::fmt;
use std::io;

use arrow_schema::ArrowError;

#[derive(Debug)]
pub enum Error {
    /// The data or options cannot be written as a valid .dta file.
    Invalid(String),
    Io(io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

impl Error {
    pub fn invalid(msg: impl Into<String>) -> Self {
        Error::Invalid(msg.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Invalid(msg) => f.write_str(msg),
            Error::Io(e) => e.fmt(f),
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<ArrowError> for Error {
    fn from(e: ArrowError) -> Self {
        match e {
            ArrowError::IoError(_, e) => Error::Io(e),
            e => Error::Invalid(e.to_string()),
        }
    }
}
