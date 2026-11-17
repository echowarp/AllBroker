//! Engine-side chains logic (chains v2): what the
//! vLLM scheduler did in Python per request and per step (phase scan, hint alignment, arming) as one batch call.
//!
//! - **Phase** (think / answer / tool call) from the model's tag token ids, incrementally per appended token.
//! - **Alignment:** a hint was computed for an earlier length `hk` (async); it lines up if the tokens produced since
//!   then equal its head. `d = len - hk` of its tokens are consumed; the rest is the chain to draft.
//! - **Confirm-then-arm:** arm only when at least `confirm` produced tokens already matched the hint (d >= confirm) and
//!   the suffix match passes the phase's gate. Thinking never arms. Structured-output requests are the caller's call.
use crate::proto::{PHASE_ANSWER, PHASE_THINK, PHASE_TOOL};

/// Tag token ids that switch the phase (Qwen3.x: <think> </think> <tool_call> </tool_call>).
#[derive(Clone, Copy, Debug)]
pub struct Tags {
    pub think0: u32,
    pub think1: u32,
    pub tool0: u32,
    pub tool1: u32,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PhaseState {
    pub thinking: bool,
    pub tool: bool,
}

impl PhaseState {
    /// Scan tokens (prompt tail at open, then each append).
    pub fn feed(&mut self, tags: &Tags, toks: &[u32]) {
        for &t in toks {
            if t == tags.think0 {
                self.thinking = true;
            } else if t == tags.think1 {
                self.thinking = false;
            } else if t == tags.tool0 {
                self.tool = true;
            } else if t == tags.tool1 {
                self.tool = false;
            }
        }
    }
    pub fn phase(&self) -> u8 {
        if self.thinking {
            PHASE_THINK
        } else if self.tool {
            PHASE_TOOL
        } else {
            PHASE_ANSWER
        }
    }
}

/// Gates for arming: minimum suffix-match length per phase, produced tokens that must confirm the hint, and the
/// longest chain handed to the engine.
#[derive(Clone, Copy, Debug)]
pub struct ArmCfg {
    pub gate_tool: u16,
    pub gate_answer: u16,
    pub confirm: u32,
    pub max_draft: usize,
}

/// The chain to draft for a request whose history is `seq` (all known tokens), given a hint computed at length
/// `hk` with `toks` its continuation. None if it doesn't line up, is too short, or fails the gates.
pub fn align<'a>(cfg: &ArmCfg, phase: u8, seq: &[u32], hk: u32, match_len: u16, toks: &'a [u32]) -> Option<&'a [u32]> {
    if phase == PHASE_THINK {
        return None;
    }
    let gate = if phase == PHASE_TOOL { cfg.gate_tool } else { cfg.gate_answer };
    if match_len < gate {
        return None;
    }
    let l = seq.len();
    let hk = hk as usize;
    if hk > l {
        return None;
    }
    let d = l - hk;
    if (d as u32) < cfg.confirm || d >= toks.len() || seq[hk..] != toks[..d] {
        return None;
    }
    let rest = &toks[d..];
    if rest.len() < 2 {
        return None;
    }
    Some(&rest[..rest.len().min(cfg.max_draft)])
}

#[cfg(test)]
mod tests {
    use super::*;
    const TAGS: Tags = Tags { think0: 1, think1: 2, tool0: 3, tool1: 4 };
    const CFG: ArmCfg = ArmCfg { gate_tool: 16, gate_answer: 24, confirm: 1, max_draft: 24 };

    #[test]
    fn phases() {
        let mut p = PhaseState::default();
        p.feed(&TAGS, &[1, 9, 9]);
        assert_eq!(p.phase(), PHASE_THINK);
        p.feed(&TAGS, &[2, 9]);
        assert_eq!(p.phase(), PHASE_ANSWER);
        p.feed(&TAGS, &[3]);
        assert_eq!(p.phase(), PHASE_TOOL);
        p.feed(&TAGS, &[9, 4]);
        assert_eq!(p.phase(), PHASE_ANSWER);
    }

    #[test]
    fn alignment_and_gates() {
        let seq: Vec<u32> = (100..120).collect(); // len 20
        let toks: Vec<u32> = (118..150).collect(); // hint computed at hk = 18: continuation 118, 119, 120, ...
        // two produced tokens confirm it: chain starts at 120
        assert_eq!(align(&CFG, PHASE_TOOL, &seq, 18, 20, &toks).unwrap()[0], 120);
        // max_draft caps the chain
        assert_eq!(align(&CFG, PHASE_TOOL, &seq, 18, 20, &toks).unwrap().len(), 24);
        // unconfirmed (d = 0) does not arm
        assert!(align(&CFG, PHASE_TOOL, &seq, 20, 20, &[120, 121, 122]).is_none());
        // gate per phase: 20 passes tool (16), not answer (24)
        assert!(align(&CFG, PHASE_ANSWER, &seq, 18, 20, &toks).is_none());
        // a produced token that differs from the hint's head
        let bad: Vec<u32> = [118, 777].iter().copied().chain(120..140).collect();
        assert!(align(&CFG, PHASE_TOOL, &seq, 18, 20, &bad).is_none());
        // never while thinking
        assert!(align(&CFG, PHASE_THINK, &seq, 18, 40, &toks).is_none());
        // hint from the future (stale caller) / exhausted hint
        assert!(align(&CFG, PHASE_TOOL, &seq, 25, 20, &toks).is_none());
        assert!(align(&CFG, PHASE_TOOL, &seq, 18, 20, &[118, 119, 120]).is_none());
    }
}
