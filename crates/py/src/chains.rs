//! `llmt_kv.ChainsClient`: engine side of the lookup chains. Never blocks: calls queue frames for a
//! background sender; a receiver thread keeps the latest hint per request.
use kvchains::client::{Client as Inner, Config};
use kvchains::engine::{ArmCfg, Tags};
use std::sync::Mutex;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};
use std::time::Duration;

/// Tokens from a list of ints or from bytes (u32 little-endian, e.g. numpy uint32 .tobytes()).
fn toks(o: &Bound<'_, PyAny>) -> PyResult<Vec<u32>> {
    if let Ok(b) = o.downcast::<PyBytes>() {
        let b = b.as_bytes();
        if b.len() % 4 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err("token bytes: length not a multiple of 4"));
        }
        return Ok(b.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect());
    }
    o.extract::<Vec<u32>>()
}

#[pyclass(module = "llmt_kv")]
pub struct ChainsClient {
    c: Inner,
    /// chains v2 engine config (`configure`): phase tag ids + arming gates
    eng: Mutex<Option<(Tags, ArmCfg)>>,
}

#[pymethods]
impl ChainsClient {
    #[new]
    #[pyo3(signature = (socket="/run/llmt-kv/chains.sock", engine_id=0, model_tag="", gate=16, max_chain=64,
                        queue_frames=4096, reconnect_ms=1000))]
    fn new(socket: &str, engine_id: u16, model_tag: &str, gate: u16, max_chain: u16, queue_frames: usize,
           reconnect_ms: u64) -> Self {
        let mut cfg = Config::new(socket, engine_id);
        cfg.model_tag = model_tag.into();
        cfg.gate = gate;
        cfg.max_chain = max_chain;
        cfg.queue_frames = queue_frames;
        cfg.reconnect_ms = reconnect_ms;
        ChainsClient { c: Inner::new(cfg), eng: Mutex::new(None) }
    }

    /// A new request with its prompt tokens (list of ints or u32 bytes).
    #[pyo3(signature = (req, tokens, thinking=false))]
    fn open(&self, py: Python<'_>, req: u64, tokens: &Bound<'_, PyAny>, thinking: bool) -> PyResult<()> {
        let t = toks(tokens)?;
        let c = &self.c;
        py.allow_threads(|| c.open(req, thinking as u8, &t));
        Ok(())
    }

    /// Chains v2: tag token ids for the phase tracker and the arming gates used by `step`.
    #[pyo3(signature = (think0, think1, tool0, tool1, gate_tool=16, gate_answer=16, confirm=1, max_draft=24))]
    fn configure(&self, think0: u32, think1: u32, tool0: u32, tool1: u32, gate_tool: u16, gate_answer: u16,
                 confirm: u32, max_draft: usize) {
        *self.eng.lock().unwrap() = Some((Tags { think0, think1, tool0, tool1 },
                                          ArmCfg { gate_tool, gate_answer, confirm, max_draft }));
    }

    /// One engine step for the batch: reqs[i] gained counts[i] tokens (flat `new_tokens`, list or u32 bytes) and now
    /// has lens[i] tokens. Appends, tracks phases, aligns hints; returns the armed rows only, as
    /// [(row, chain_ref, match_len, chain tokens as u32 bytes)]. Runs without the GIL.
    fn step<'py>(&self, py: Python<'py>, reqs: Vec<u64>, lens: Vec<u32>, new_tokens: &Bound<'py, PyAny>,
                 counts: Vec<u32>) -> PyResult<Vec<(u32, u64, u16, Bound<'py, PyBytes>)>> {
        let Some((tags, cfg)) = *self.eng.lock().unwrap() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("ChainsClient.step: call configure() first"));
        };
        if lens.len() != reqs.len() || counts.len() != reqs.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("step: reqs, lens, counts must have equal length"));
        }
        let t = toks(new_tokens)?;
        let c = &self.c;
        let armed = py.allow_threads(|| c.step(&tags, &cfg, &reqs, &lens, &t, &counts));
        Ok(armed.into_iter().map(|a| {
            let b: Vec<u8> = a.tokens.iter().flat_map(|x| x.to_le_bytes()).collect();
            (a.row, a.chain_ref, a.match_len, PyBytes::new(py, &b))
        }).collect())
    }

    /// Tokens appended to a request; known_len = its length after them; phase 0 think / 1 answer / 2 tool.
    fn append(&self, req: u64, known_len: u32, phase: u8, tokens: &Bound<'_, PyAny>) -> PyResult<()> {
        let t = toks(tokens)?;
        self.c.append(req, known_len, phase, &t);
        Ok(())
    }

    #[pyo3(signature = (req, aborted=false))]
    fn close(&self, req: u64, aborted: bool) {
        self.c.close(req, aborted);
    }

    /// [(req, offered, accepted, chain_ref), ...]: what the hints earned.
    fn feedback(&self, items: Vec<(u64, u16, u16, u64)>) {
        self.c.feedback(items);
    }

    /// Corpus / template tokens for the broker's global store (permanent: never evicted by recency).
    #[pyo3(signature = (tokens, permanent=true))]
    fn seed(&self, tokens: &Bound<'_, PyAny>, permanent: bool) -> PyResult<()> {
        let t = toks(tokens)?;
        self.c.seed(&t, permanent);
        Ok(())
    }

    /// None, or (known_len, match_len, source, chain_ref, tokens) for the request's latest hint. Use it only when
    /// known_len equals the request's current length.
    fn hint(&self, req: u64) -> Option<(u32, u16, u8, u64, Vec<u32>)> {
        self.c.hint(req).map(|h| (h.known_len, h.match_len, h.source, h.chain_ref, h.tokens))
    }

    /// Broker-side chains stats as a JSON string (None on timeout).
    #[pyo3(signature = (timeout_ms=2000))]
    fn stats_json(&self, py: Python<'_>, timeout_ms: u64) -> Option<String> {
        let c = &self.c;
        py.allow_threads(|| c.stats(Duration::from_millis(timeout_ms)))
    }

    fn counters<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        for (k, v) in self.c.counters() {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    #[pyo3(signature = (timeout_ms=5000))]
    fn flush(&self, py: Python<'_>, timeout_ms: u64) -> bool {
        let c = &self.c;
        py.allow_threads(|| c.flush(Duration::from_millis(timeout_ms)))
    }
}
