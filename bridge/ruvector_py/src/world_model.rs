//! RuVector-as-world-model + an all-Rust rollout planner (Phase 2, Option B).
//!
//! One `VectorDB` per action maps `state -> (next_state, reward)` via k-NN
//! (next_state + reward ride along in metadata). `plan` does short
//! receding-horizon rollouts entirely in Rust — for each candidate first action
//! it simulates the chosen action then continues greedily for `horizon` steps,
//! and returns the best first action. ~n_actions^2 * horizon searches per
//! decision: kept out of Python on purpose so the searches don't cross the
//! PyO3 boundary thousands of times per episode.
use std::collections::HashMap;
use std::sync::Arc;

use pyo3::prelude::*;
use serde_json::json;

use ruvector_core::types::{DbOptions, SearchQuery, VectorEntry};
use ruvector_core::VectorDB;

#[pyclass]
pub struct WorldModel {
    dbs: Vec<Arc<VectorDB>>, // one transition index per action
    n_actions: usize,
    gamma: f32,
    horizon: usize,
    unknown_dist: f32, // score above which the model is treated as ignorant
}

impl WorldModel {
    /// Nearest-neighbour transition for (action, state): (next_state, reward, score).
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
        Some((next, reward, hit.score))
    }

    /// Discounted predicted return of taking `first_action` now, then continuing
    /// greedily (best predicted immediate reward) for the remaining horizon.
    fn rollout(&self, first_action: usize, state: &[f32]) -> f32 {
        let mut total = 0.0f32;
        let mut disc = 1.0f32;
        let mut s: Vec<f32> = state.to_vec();

        match self.predict(first_action, &s) {
            Some((ns, r, d)) if d <= self.unknown_dist => {
                total += disc * r;
                disc *= self.gamma;
                s = ns;
            }
            // unexplored first action: neutral score (epsilon-greedy drives exploration)
            _ => return 0.0,
        }

        for _ in 1..self.horizon {
            let mut best_r = f32::NEG_INFINITY;
            let mut best_ns: Option<Vec<f32>> = None;
            for a in 0..self.n_actions {
                if let Some((ns, r, d)) = self.predict(a, &s) {
                    if d <= self.unknown_dist && r > best_r {
                        best_r = r;
                        best_ns = Some(ns);
                    }
                }
            }
            match best_ns {
                Some(ns) => {
                    total += disc * best_r;
                    disc *= self.gamma;
                    s = ns;
                }
                None => break,
            }
        }
        total
    }
}

#[pymethods]
impl WorldModel {
    #[new]
    #[pyo3(signature = (n_actions, dim, storage_path=None, gamma=0.99, horizon=4, unknown_dist=None))]
    fn new(
        n_actions: usize,
        dim: usize,
        storage_path: Option<String>,
        gamma: f32,
        horizon: usize,
        unknown_dist: Option<f32>,
    ) -> PyResult<Self> {
        let prefix = storage_path.unwrap_or_else(|| "./wm_store".to_string());
        let mut dbs = Vec::with_capacity(n_actions);
        for a in 0..n_actions {
            let opts = DbOptions {
                dimensions: dim,
                storage_path: format!("{prefix}_a{a}"),
                ..Default::default()
            };
            let db = VectorDB::new(opts)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ruvector: {e}")))?;
            dbs.push(Arc::new(db));
        }
        Ok(Self {
            dbs,
            n_actions,
            gamma,
            horizon,
            unknown_dist: unknown_dist.unwrap_or(f32::MAX),
        })
    }

    /// Record a real transition: state --action--> next_state, with reward.
    fn observe(
        &self,
        action: usize,
        state: Vec<f32>,
        reward: f32,
        next_state: Vec<f32>,
    ) -> PyResult<()> {
        if action >= self.n_actions {
            return Err(pyo3::exceptions::PyValueError::new_err("action out of range"));
        }
        let md = HashMap::from([
            ("reward".to_string(), json!(reward)),
            ("next_state".to_string(), json!(next_state)),
        ]);
        let entry = VectorEntry { id: None, vector: state, metadata: Some(md) };
        self.dbs[action]
            .insert(entry)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("ruvector: {e}")))?;
        Ok(())
    }

    /// Plan one action: roll out each candidate first action, return the best.
    fn plan(&self, state: Vec<f32>) -> usize {
        let mut best_a = 0usize;
        let mut best_val = f32::NEG_INFINITY;
        for a in 0..self.n_actions {
            let v = self.rollout(a, &state);
            if v > best_val {
                best_val = v;
                best_a = a;
            }
        }
        best_a
    }
}
