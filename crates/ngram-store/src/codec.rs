//! Row codecs. IQ4_NL (the production GGUF table, 90 bytes per 160-wide row) is implemented; the EXL3 K6 trellis
//! (122 bytes per row + per-head bias, `ngram_embedding.safetensors`) comes with the split engine.

pub const GGML_TYPE_IQ4_NL: u32 = 20;
const IQ4NL_VALUES: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];
const IQ4NL_BLOCK: usize = 32;
const IQ4NL_BLOCK_BYTES: usize = 18; // f16 scale + 16 bytes of nibbles

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Codec {
    Iq4Nl,
}

impl Codec {
    pub fn from_ggml(ty: u32) -> Option<Codec> {
        (ty == GGML_TYPE_IQ4_NL).then_some(Codec::Iq4Nl)
    }

    pub fn row_bytes(self, dim: usize) -> usize {
        match self {
            Codec::Iq4Nl => dim / IQ4NL_BLOCK * IQ4NL_BLOCK_BYTES,
        }
    }

    /// Decodes one row into `dst` (len = dim), bit-identical to gufo's `DequantizeIQ4_NL`.
    pub fn decode(self, src: &[u8], dst: &mut [f32]) {
        match self {
            Codec::Iq4Nl => {
                for (b, (blk, out)) in src.chunks_exact(IQ4NL_BLOCK_BYTES).zip(dst.chunks_exact_mut(IQ4NL_BLOCK)).enumerate() {
                    let _ = b;
                    let d = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                    let qs = &blk[2..18];
                    for i in 0..16 {
                        out[i] = d * IQ4NL_VALUES[(qs[i] & 0x0F) as usize] as f32;
                        out[i + 16] = d * IQ4NL_VALUES[(qs[i] >> 4) as usize] as f32;
                    }
                }
            }
        }
    }
}

/// IEEE half -> single, including subnormals, infinities and NaN.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: value = man * 2^-24; shift the leading 1 into the implicit-bit position
            let mut shifts: u32 = 0;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                shifts += 1;
            }
            sign | ((127 - 14 - shifts) << 23) | ((m & 0x3FF) << 13)
        }
    } else if exp == 0x1F {
        sign | 0x7F80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f16_known_values() {
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0001), 5.960_464_5e-8);
        assert_eq!(f16_to_f32(0x0200), 3.051_757_8e-5); // largest-exponent subnormal: 2^-15
        assert_eq!(f16_to_f32(0x03FF), 6.097_555e-5);
        assert!(f16_to_f32(0x7C00).is_infinite());
    }
}
