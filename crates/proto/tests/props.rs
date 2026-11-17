//! Fuzz-style property: decoding arbitrary bytes as any request or reply never panics.
use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig { cases: 3000, ..ProptestConfig::default() })]
    #[test]
    fn decode_never_panics(op in 1u16..=17, b in proptest::collection::vec(any::<u8>(), 0..512)) {
        if let Some(op) = kvproto::Op::from_u16(op) {
            let _ = kvproto::Request::decode(op, &b);
            let _ = kvproto::Reply::decode(op, &b);
        }
    }
}
