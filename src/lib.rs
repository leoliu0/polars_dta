//! Writing Arrow data (any object exporting `__arrow_c_stream__`, such as a Polars DataFrame) to
//! Stata .dta files.

mod error;
mod plan;
mod write;

use std::collections::HashMap;
use std::path::PathBuf;

use arrow_array::ffi_stream::{FFI_ArrowArrayStream, RecordBatchStreamReader};
use arrow_array::{RecordBatch, RecordBatchReader};
use arrow_schema::SchemaRef;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

use crate::error::Error;

impl From<Error> for PyErr {
    fn from(e: Error) -> Self {
        match e {
            Error::Invalid(msg) => PyValueError::new_err(msg),
            Error::Io(e) => e.into(),
        }
    }
}

/// Imports every batch of `data`'s Arrow C stream; the buffers stay shared with the exporter.
fn import_stream(data: &Bound<'_, PyAny>) -> PyResult<(SchemaRef, Vec<RecordBatch>)> {
    // Polars 1.3 requires `requested_schema` positionally; None asks for the native schema.
    let capsule = data.call_method1("__arrow_c_stream__", (data.py().None(),))?;
    let capsule = capsule.cast::<PyCapsule>()?;
    let ptr = capsule.pointer_checked(Some(c"arrow_array_stream"))?;
    // SAFETY: a capsule named "arrow_array_stream" holds an FFI_ArrowArrayStream; `from_raw` moves
    // it out and leaves a released stream, so the capsule's destructor does not release it again.
    let stream = unsafe { FFI_ArrowArrayStream::from_raw(ptr.as_ptr().cast()) };
    let reader = RecordBatchStreamReader::try_new(stream).map_err(Error::from)?;
    let schema = reader.schema();
    let batches = reader.collect::<Result<Vec<_>, _>>().map_err(Error::from)?;
    Ok((schema, batches))
}

#[pyfunction]
#[pyo3(signature = (data, path, *, variable_labels=None, value_labels=None, data_label=None, timestamp=None))]
fn write_dta(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    path: PathBuf,
    variable_labels: Option<HashMap<String, String>>,
    value_labels: Option<HashMap<String, HashMap<i64, String>>>,
    data_label: Option<String>,
    timestamp: Option<String>,
) -> PyResult<()> {
    let (schema, batches) = import_stream(data)?;
    let opts = plan::Options {
        variable_labels: variable_labels.unwrap_or_default(),
        value_labels: value_labels.unwrap_or_default(),
        data_label: data_label.unwrap_or_default(),
        timestamp: timestamp.unwrap_or_default(),
    };
    py.detach(|| {
        let plan = plan::plan(&schema, &batches, &opts)?;
        write::write_file(&path, &plan, &batches)
    })?;
    Ok(())
}

#[pymodule]
fn _polars_dta(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(write_dta, m)?)
}
