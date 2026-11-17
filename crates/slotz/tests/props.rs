//! Fuzz-style properties of in-place slot compression: a reader of a
//! compressed slot gets exactly the original bytes, or a failure, never silently wrong bytes, whatever damage the
//! slot took (a torn copy, a slot reused under a reader, random corruption); arbitrary bytes never panic.
use proptest::prelude::*;

/// Compressible raw data: runs of a byte with sparse noise, `len` bytes.
fn raw() -> impl Strategy<Value = Vec<u8>> {
    (1usize..200_000, any::<u64>()).prop_map(|(len, seed)| {
        let mut s = seed | 1;
        (0..len).map(|i| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            if s % 23 == 0 { s as u8 } else { (seed as usize + i / 300) as u8 }
        }).collect()
    })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, ..ProptestConfig::default() })]

    #[test]
    fn roundtrip(data in raw(), chunk in prop_oneof![Just(4096usize), Just(32768usize), 512usize..70_000],
                 a in any::<usize>(), b in any::<usize>()) {
        let zz = kvslotz::compress(&data, chunk, 0.0);
        prop_assert!(zz.is_some() || data.len() < 65_536, "compressible {} bytes not compressed", data.len());
        if let Some(z) = zz {
            prop_assert!(kvslotz::is_compressed(&z));
            prop_assert_eq!(kvslotz::raw_len(&z), Some(data.len()));
            let (lo, hi) = { let (x, y) = (a % (data.len() + 1), b % (data.len() + 1)); (x.min(y), x.max(y)) };
            let mut out = vec![0u8; hi - lo];
            prop_assert!(kvslotz::read_range(&z, lo, &mut out));
            prop_assert_eq!(&out[..], &data[lo..hi]);
        }
    }

    #[test]
    fn damage_never_silent(data in raw(), chunk in prop_oneof![Just(4096usize), Just(32768usize)],
                           hits in proptest::collection::vec((any::<usize>(), any::<u8>()), 1..16)) {
        if let Some(mut z) = kvslotz::compress(&data, chunk, 0.0) {
            for (at, v) in hits {
                let i = at % z.len();
                z[i] ^= v | 1;
            }
            let mut out = vec![0u8; data.len()];
            if kvslotz::read_range(&z, 0, &mut out) {
                prop_assert_eq!(&out[..], &data[..], "damaged slot decoded to different bytes without an error");
            }
        }
    }

    #[test]
    fn arbitrary_bytes_never_panic(mut junk in proptest::collection::vec(any::<u8>(), 0..4096),
                                   off in 0usize..10_000, n in 0usize..10_000) {
        if junk.len() >= 8 {
            junk[..8].copy_from_slice(kvslotz::MAGIC);
        }
        let mut out = vec![0u8; n];
        let _ = kvslotz::read_range(&junk, off, &mut out);
        let _ = kvslotz::raw_len(&junk);
    }
}
