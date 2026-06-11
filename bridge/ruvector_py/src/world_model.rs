//! RuVector world model + value-bootstrapped planner (Phase 2, Option B v2).
//!
//! v1 (a reward-summing rollout) collapsed to no-op on dense-reward scenarios:
//! the immediate reward was action-invariant (constant living reward), so every
//! rollout tied and argmax fell to index 0. v2 scores each action by a
//! value-bootstrapped 1-step Bellman backup using a LEARNED VALUE (the
//! discounted return-to-go recorded with each transition) instead of immediate
//! reward:
//!     score(a) = predicted_reward(s, a) + gamma * V(predicted_next_state)
//!     V(s)     = max over actions of the nearest-neighbour return-to-go at s
//! This gives a gradient even when immediate rewards are flat: actions that lead
//! to higher-value states (e.g. toward a health kit, away from death) win. It is
//! model-based (predicts the next state) and value-optimising, at ~n + n^2
//! searches per decision (cheaper than the old horizon-H rollout).
use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::json;

use ruvector_core::types::{DbOptions, HnswConfig, SearchQuery, VectorEntry};
use ruvector_core::VectorDB;

#[pyclass]
pub struct WorldModel {
    dbs: Vec<Arc<VectorDB>>, // one transition index per action
    n_actions: usize,
    gamma: f32,
}

impl WorldModel {
    /// Nearest transition for (action, state): (next_state, reward, return_to_go).
    fn predict(&self, action: usize, state: &[f32]) -> Option<(Vec<f32>, f32, f32)> {
        let q = SearchQuery { vector: state.to_vec(), k: 1, filter: None, ef_search: None };
        let hit = self.dbs[action].search(q).ok()?.into_iter().next()?;
        let md = hit.metadata?;
        let reward = md.get("reward")?.as_f64()? as f32;
        let next = md
            .get("next_state")?
            .as_array()?
            .iter()
            .filter_map(|v| v.as_f64().map(|x| x as f32))
            .collect::<Vec<f32>>();
        let value = md.get("value").and_then(|v| v.as_f64()).unwrap_or(0.0) as f32;
        Some((next, reward, value))
    }

    /// V(s) ~= max over actions of the nearest-neighbour return-to-go at s.
    fn state_value(&self, state: &[f32]) -> f32 {
        let mut best = f32::NEG_INFINITY;
        for a in 0..self.n_actions {
            if let Some((_ns, _r, v)) = self.predict(a, state) {
                if v > best {
                    best = v;
                }
            }
        }
        if best.is_finite() {
            best
        } else {
            0.0
        }
    }
}

#[pymethods]
impl WorldModel {
    #[new]
    #[pyo3(signature = (n_actions, dim, storage_path=None, gamma=0.99, max_elements=100_000))]
    fn new(
        n_actions: usize,
        dim: usize,
        storage_path: Option<String>,
        gamma: f32,
        max_elements: usize,
    ) -> PyResult<Self> {
        let prefix = storage_path.unwrap_or_else(|| "./wm_store".to_string());
        let mut dbs = Vec::with_capacity(n_actions);
        for a in 0..n_actions {
            // bound HNSW pre-allocation (default is 10M elems ~= 661 MB per index)
            let opts = DbOptions {
                dimensions: dim,
                storage_path: format!("{prefix}_a{a}"),
                hnsw_config: Some(HnswConfig { max_elements, ..Default::default() }),
                ..Default::default()
            };
            let db = VectorDB::new(opts)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ruvector: {e}")))?;
            dbs.push(Arc::new(db));
        }
        Ok(Self { dbs, n_actions, gamma })
    }

    /// Record a transition with its discounted return-to-go (the learned value).
    fn observe(
        &self,
        action: usize,
        state: Vec<f32>,
        reward: f32,
        next_state: Vec<f32>,
        value: f32,
    ) -> PyResult<()> {
        if action >= self.n_actions {
            return Err(pyo3::exceptions::PyValueError::new_err("action out of range"));
        }
        let md = HashMap::from([
            ("reward".to_string(), json!(reward)),
            ("next_state".to_string(), json!(next_state)),
            ("value".to_string(), json!(value)),
        ]);
        let entry = VectorEntry { id: None, vector: state, metadata: Some(md) };
        self.dbs[action]
            .insert(entry)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ruvector: {e}")))?;
        Ok(())
    }

    /// One-step value-bootstrapped backup: argmax_a [ r(s,a) + gamma * V(s') ].
    /// Unexplored actions are skipped; if none are known, returns 0.
    fn plan(&self, state: Vec<f32>) -> usize {
        let mut best_a = 0usize;
        let mut best = f32::NEG_INFINITY;
        for a in 0..self.n_actions {
            if let Some((ns, r, _v)) = self.predict(a, &state) {
                let score = r + self.gamma * self.state_value(&ns);
                if score > best {
                    best = score;
                    best_a = a;
                }
            }
        }
        best_a
    }
}
