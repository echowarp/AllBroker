//! Offline replay of real requests through the chains logic (the server's rules, no transport): own context first,
//! global store on a miss, gate, max_chain, no hints while thinking, finished outputs feed the global store.
//! Input: binary records `u32 n_prompt, u32 n_out, prompt[n_prompt] u32, out[n_out] u32`, in arrival order.
//! Prints numbers only (the token ids are customer-derived).
//!   replay FILE [gate 16] [max_chain 64] [global 1]
use kvchains::index::RequestIndex;
use kvchains::store::{self, Store};
use std::io::Read;

const THINK0: u32 = 248068;
const THINK1: u32 = 248069;
const TC0: u32 = 248058;
const TC1: u32 = 248059;

#[derive(Default, Clone, Copy)]
struct Acc {
    positions: u64,
    hinted: u64,
    hinted_own: u64,
    hinted_global: u64,
    matched_any: u64,
    // greedy walk: tokens the chain would have supplied
    covered: u64,
    covered_own: u64,
    covered_global: u64,
    walk_tokens: u64,
}

fn main() -> std::io::Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let gate: usize = a.get(2).and_then(|x| x.parse().ok()).unwrap_or(16);
    let max_chain: usize = a.get(3).and_then(|x| x.parse().ok()).unwrap_or(64);
    let global = a.get(4).map_or(true, |x| x != "0");
    let mut b = Vec::new();
    std::fs::File::open(&a[1])?.read_to_end(&mut b)?;
    let rd = |p: &mut usize| -> u32 {
        let v = u32::from_le_bytes(b[*p..*p + 4].try_into().unwrap());
        *p += 4;
        v
    };
    let mut store = Store::new(512 << 20);
    let mut acc = [Acc::default(); 3]; // by phase: 0 think, 1 answer, 2 tool
    // matched-prefix length of the hint at armed positions (hinted, at a walk step), per phase answer/tool
    let mut hist = [[0u64; 65]; 3];
    let (mut p, mut nreq, mut t_open, mut t_step) = (0usize, 0u64, 0f64, 0f64);
    let t0 = std::time::Instant::now();
    while p + 8 <= b.len() {
        let (np, no) = (rd(&mut p) as usize, rd(&mut p) as usize);
        let prompt: Vec<u32> = (0..np).map(|_| rd(&mut p)).collect();
        let out: Vec<u32> = (0..no).map(|_| rd(&mut p)).collect();
        nreq += 1;
        let ts = std::time::Instant::now();
        let mut ix = RequestIndex::new(prompt.clone());
        t_open += ts.elapsed().as_secs_f64();
        // phase at the start of the output: thinking if the prompt's tail opened <think> without closing it
        let (mut think, mut tool) = (false, false);
        for &t in prompt.iter().rev().take(16).collect::<Vec<_>>().into_iter().rev() {
            if t == THINK0 { think = true } else if t == THINK1 { think = false }
        }
        let mut hint: Option<(Vec<u32>, bool)> = None; // (tokens, is_global), aligned to the current length
        let mut skip_until = 0usize; // greedy walk: positions covered by an accepted chain
        let ts = std::time::Instant::now();
        for i in 0..out.len() {
            let ph = if think { 0 } else if tool { 2 } else { 1 };
            let a = &mut acc[ph];
            a.positions += 1;
            if let Some((h, g)) = &hint {
                let m = h.iter().zip(&out[i..]).take_while(|(x, y)| x == y).count();
                a.hinted += 1;
                if *g { a.hinted_global += 1 } else { a.hinted_own += 1 }
                if m > 0 {
                    a.matched_any += 1;
                }
                if i >= skip_until {
                    hist[ph][m.min(64)] += 1;
                    a.covered += m as u64;
                    if *g { a.covered_global += m as u64 } else { a.covered_own += m as u64 }
                    skip_until = i + m + 1;
                }
            }
            if i >= skip_until.saturating_sub(1) {
                a.walk_tokens += 1;
            }
            let t = out[i];
            if t == THINK0 { think = true } else if t == THINK1 { think = false } else if t == TC0 { tool = true } else if t == TC1 { tool = false }
            let m = ix.append(&[t], 1024);
            hint = None;
            if !think {
                if let Some(m) = m.filter(|m| m.len >= gate) {
                    hint = Some((ix.continuation(m.pos, max_chain).to_vec(), false));
                } else if global {
                    let tk = &ix.tokens;
                    if let Some(gh) = store.lookup(&tk[tk.len().saturating_sub(store::LEFT)..], gate, max_chain, 0) {
                        hint = Some((gh.tokens, true));
                    }
                }
            }
        }
        t_step += ts.elapsed().as_secs_f64();
        if global {
            store.insert_doc(&out, store::SRC_OUTPUT, 1, false, 0);
            if nreq % 200 == 0 {
                store.gc(0);
            }
        }
    }
    let tot: u64 = acc.iter().map(|a| a.positions).sum();
    println!("{nreq} requests, {tot} output positions; gate {gate}, max_chain {max_chain}, global {global}; \
              open {:.2} s, steps {:.2} s ({:.2} us/token), wall {:.1} s",
             t_open, t_step, t_step / tot as f64 * 1e6, t0.elapsed().as_secs_f64());
    for (name, a) in ["think", "answer", "tool", "all non-think"].iter().zip([acc[0], acc[1], acc[2], {
        let mut s = acc[1];
        for f in [&mut s.positions, &mut s.hinted, &mut s.hinted_own, &mut s.hinted_global, &mut s.matched_any,
                  &mut s.covered, &mut s.covered_own, &mut s.covered_global] {
            let _ = f;
        }
        Acc { positions: acc[1].positions + acc[2].positions, hinted: acc[1].hinted + acc[2].hinted,
              hinted_own: acc[1].hinted_own + acc[2].hinted_own, hinted_global: acc[1].hinted_global + acc[2].hinted_global,
              matched_any: acc[1].matched_any + acc[2].matched_any, covered: acc[1].covered + acc[2].covered,
              covered_own: acc[1].covered_own + acc[2].covered_own, covered_global: acc[1].covered_global + acc[2].covered_global,
              walk_tokens: 0 }
    }]) {
        if a.positions == 0 {
            continue;
        }
        println!("{name:>13}: positions {:>8} | hinted {:5.1}% (own {:4.1}%, global {:4.1}%) | first token right {:5.1}% of hints | \
                  chain-covered tokens {:5.1}% (own {:4.1}%, global {:4.1}%)",
                 a.positions, 100.0 * a.hinted as f64 / a.positions as f64, 100.0 * a.hinted_own as f64 / a.positions as f64,
                 100.0 * a.hinted_global as f64 / a.positions as f64,
                 100.0 * a.matched_any as f64 / a.hinted.max(1) as f64, 100.0 * a.covered as f64 / a.positions as f64,
                 100.0 * a.covered_own as f64 / a.positions as f64, 100.0 * a.covered_global as f64 / a.positions as f64);
    }
    // expected accepted chain tokens per armed step if the engine drafts L chain tokens: E[min(m, L)]
    for (name, ph) in [("answer", 1usize), ("tool", 2usize)] {
        let n: u64 = hist[ph].iter().sum();
        if n == 0 {
            continue;
        }
        let mut line = format!("{name:>6}: {n} armed steps; m=0 {:.1}%, median m ", 100.0 * hist[ph][0] as f64 / n as f64);
        let mut c = 0;
        for (m, &k) in hist[ph].iter().enumerate() {
            c += k;
            if c * 2 >= n {
                line += &format!("{m}");
                break;
            }
        }
        line += "; E[min(m,L)] by L:";
        for l in [4usize, 8, 12, 16, 23, 31, 48, 64] {
            let e: f64 = hist[ph].iter().enumerate().map(|(m, &k)| m.min(l) as f64 * k as f64).sum::<f64>() / n as f64;
            let full = hist[ph][l.min(64)..].iter().sum::<u64>() as f64 / n as f64;
            line += &format!(" L{l}={e:.1} (all L right {:.0}%)", 100.0 * full);
        }
        println!("{line}");
    }
    println!("global store: {} docs, {} keys, {:.0} MiB, hits {} of {} lookups", store.stats.docs_in, store.index.len(),
             store.mem_bytes() as f64 / (1 << 20) as f64, store.stats.hits, store.stats.lookups);
    Ok(())
}
