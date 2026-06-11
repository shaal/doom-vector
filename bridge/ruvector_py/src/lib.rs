//! Minimal PyO3 binding over `ruvector-core` (verified sync API, v2.2.0).
//!
//! Exposes one class, `RuVectorMemory`, with `insert` and `search`. Metadata is
//! restricted to float values (enough for episodic control: action index +
//! discounted return). The store is sync (no tokio) and `&self`-method based,
//! so we wrap it in `Arc` for thread-safe access from Python.
//!
//! NOTE: if the build fails on the import paths below, the public re-exports in
//! your installed 2.2.0 differ — try `ruvector_core::vector_db::VectorDB` and
//! confirm `types` paths with `cargo doc --open`.
mod world_model;

use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use ruvector_core::types::{DbOptions, HnswConfig, SearchQuery, VectorEntry};
use ruvector_core::VectorDB;

#[pyclass]
struct RuVectorMemory {
    db: Arc<VectorDB>,
}

#[pymethods]
impl RuVectorMemory {
    #[new]
    #[pyo3(signature = (dimensions, storage_path=None, max_elements=100_000))]
    fn new(dimensions: usize, storage_path: Option<String>, max_elements: usize) -> PyResult<Self> {
        // HnswConfig::default() pre-allocates for 10M elements (~661 MB) — fatal
        // on a 512 MB device. Bound it; the graph grows as needed within this cap.
        let opts = DbOptions {
            dimensions,
            storage_path: storage_path.unwrap_or_else(|| "./ruvector_store".to_string()),
            hnsw_config: Some(HnswConfig { max_elements, ..Default::default() }),
            ..Default::default()
        };
        let db = VectorDB::new(opts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ruvector: {e}")))?;
        Ok(Self { db: Arc::new(db) })
    }

    /// Insert a vector with optional float metadata. Returns the assigned id.
    #[pyo3(signature = (vector, id=None, metadata=None))]
    fn insert(
        &self,
        vector: Vec<f32>,
        id: Option<String>,
        metadata: Option<HashMap<String, f64>>,
    ) -> PyResult<String> {
        let md = metadata.map(|m| {
            m.into_iter()
                .map(|(k, v)| (k, serde_json::json!(v)))
                .collect::<HashMap<String, serde_json::Value>>()
        });
        let entry = VectorEntry { id, vector, metadata: md };
        self.db
            .insert(entry)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ruvector: {e}")))
    }

    /// k-NN search. Returns a list of (id, score, metadata-dict) tuples.
    #[pyo3(signature = (vector, k=8))]
    fn search(&self, py: Python<'_>, vector: Vec<f32>, k: usize) -> PyResult<Py<PyList>> {
        let query = SearchQuery { vector, k, filter: None, ef_search: None };
        let results = self
            .db
            .search(query)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ruvector: {e}")))?;

        let out = PyList::empty(py);
        for r in results {
            let md = PyDict::new(py);
            if let Some(meta) = r.metadata {
                for (key, val) in meta {
                    if let Some(f) = val.as_f64() {
                        md.set_item(key, f)?;
                    }
                }
            }
            out.append((r.id, r.score, md))?;
        }
        Ok(out.into())
    }

    /// Delete an entry by id. Returns whether it existed. Used by the memory
    /// store's value-based eviction to keep the index bounded.
    fn delete(&self, id: &str) -> PyResult<bool> {
        self.db
            .delete(id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ruvector: {e}")))
    }
}

#[pymodule]
fn ruvector_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RuVectorMemory>()?;
    m.add_class::<world_model::WorldModel>()?;
    Ok(())
}
