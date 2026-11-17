//! Minimal GGUF reader: metadata key/values and tensor descriptors, enough to locate the n-gram table and read the
//! PLE hash parameters. Tensor data is never read here.

use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Array(Vec<Value>),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match *self {
            Value::U64(v) => Some(v),
            Value::I64(v) if v >= 0 => Some(v as u64),
            _ => None,
        }
    }
    pub fn as_u64_array(&self) -> Option<Vec<u64>> {
        match self {
            Value::Array(a) => a.iter().map(Value::as_u64).collect(),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// Absolute byte offset of the tensor data in its file.
    pub file_offset: u64,
    pub path: PathBuf,
}

#[derive(Debug, Default)]
pub struct Gguf {
    pub kv: HashMap<String, Value>,
    pub tensors: HashMap<String, TensorInfo>,
}

fn rd<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
    let mut b = [0u8; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
fn u32(r: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(rd(r)?))
}
fn u64(r: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(rd(r)?))
}
fn string(r: &mut impl Read) -> io::Result<String> {
    let n = u64(r)? as usize;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b)?;
    String::from_utf8(b).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

fn value(r: &mut impl Read, ty: u32) -> io::Result<Value> {
    Ok(match ty {
        0 => Value::U64(rd::<1>(r)?[0] as u64),
        1 => Value::I64(rd::<1>(r)?[0] as i8 as i64),
        2 => Value::U64(u16::from_le_bytes(rd(r)?) as u64),
        3 => Value::I64(i16::from_le_bytes(rd(r)?) as i64),
        4 => Value::U64(u32(r)? as u64),
        5 => Value::I64(i32::from_le_bytes(rd(r)?) as i64),
        6 => Value::F64(f32::from_le_bytes(rd(r)?) as f64),
        7 => Value::Bool(rd::<1>(r)?[0] != 0),
        8 => Value::Str(string(r)?),
        9 => {
            let et = u32(r)?;
            let n = u64(r)?;
            let mut v = Vec::with_capacity(n.min(1 << 20) as usize);
            for _ in 0..n {
                v.push(value(r, et)?);
            }
            Value::Array(v)
        }
        10 => Value::U64(u64(r)?),
        11 => Value::I64(i64::from_le_bytes(rd(r)?)),
        12 => Value::F64(f64::from_le_bytes(rd(r)?)),
        t => return Err(io::Error::new(io::ErrorKind::InvalidData, format!("gguf value type {t}"))),
    })
}

impl Gguf {
    /// Reads one shard's header into `self` (keys from later shards don't overwrite earlier ones).
    pub fn add_shard(&mut self, path: &Path) -> io::Result<()> {
        let mut r = BufReader::new(File::open(path)?);
        if &rd::<4>(&mut r)? != b"GGUF" {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "not a GGUF file"));
        }
        let _version = u32(&mut r)?;
        let n_tensors = u64(&mut r)?;
        let n_kv = u64(&mut r)?;
        let mut kv = HashMap::new();
        for _ in 0..n_kv {
            let k = string(&mut r)?;
            let ty = u32(&mut r)?;
            kv.insert(k, value(&mut r, ty)?);
        }
        let align = kv.get("general.alignment").and_then(Value::as_u64).unwrap_or(32);
        let mut infos = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = string(&mut r)?;
            let nd = u32(&mut r)?;
            let dims = (0..nd).map(|_| u64(&mut r)).collect::<io::Result<Vec<_>>>()?;
            let ggml_type = u32(&mut r)?;
            let off = u64(&mut r)?;
            infos.push((name, dims, ggml_type, off));
        }
        let pos = r.stream_position()?;
        let data_start = pos.div_ceil(align) * align;
        for (name, dims, ggml_type, off) in infos {
            self.tensors.insert(
                name.clone(),
                TensorInfo { name, dims, ggml_type, file_offset: data_start + off, path: path.to_owned() },
            );
        }
        for (k, v) in kv {
            self.kv.entry(k).or_insert(v);
        }
        let _ = r.seek(SeekFrom::Start(0));
        Ok(())
    }

    /// Opens every shard of a split GGUF given any one of them (`...-0000N-of-0000M.gguf`), or a single file.
    pub fn open(path: &Path) -> io::Result<Gguf> {
        let mut g = Gguf::default();
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        let shards: Vec<PathBuf> = match name.rfind("-of-") {
            Some(i) if i >= 6 => {
                let total: usize = name[i + 4..i + 9].parse().unwrap_or(1);
                (1..=total)
                    .map(|k| path.with_file_name(format!("{}{:05}{}", &name[..i - 5], k, &name[i..])))
                    .collect()
            }
            _ => vec![path.to_owned()],
        };
        for s in &shards {
            g.add_shard(s)?;
        }
        Ok(g)
    }
}
