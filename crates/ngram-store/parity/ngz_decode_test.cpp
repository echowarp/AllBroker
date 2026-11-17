// The engine's NGZ1 decoder (engine/ple_rows.h, Ngz) against the bf16 path (x 2^18 + host flush), bit for bit, and
// its speed. No GPU.
//   ngz_decode_test <rows.bin> <rows.ngz> [rows=2000000]   (an .ngz built from the first `rows` rows of rows.bin
//                                                            works too: ngram-ngz build on a prefix)
#include "../../engine/ple_rows.h"
#include <chrono>

using plerows::Ngz;
int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: ngz_decode_test <rows.bin> <rows.ngz> [rows]\n"); return 2; }
    const long n = argc > 3 ? atol(argv[3]) : 2000000;
    const int fb = open(argv[1], O_RDONLY), fz = open(argv[2], O_RDONLY);
    uint8_t h[4096];
    if (pread(fz, h, 4096, 0) != 4096 || memcmp(h, "NGZ1", 4)) { fprintf(stderr, "not NGZ1\n"); return 1; }
    Ngz z;
    if (!z.init(h + 48)) { fprintf(stderr, "bad code\n"); return 1; }
    const long B = 65536;
    std::vector<uint8_t> raw(B * 320), slots(B * Ngz::SLOT);
    std::vector<uint16_t> a(B * 160), b(B * 160);
    long bad = 0, esc = 0;
    double tz = 0, tb = 0;
    for (long r0 = 0; r0 < n; r0 += B) {
        const long m = std::min(B, n - r0);
        if (pread(fb, raw.data(), m * 320, r0 * 320) != m * 320 || pread(fz, slots.data(), m * Ngz::SLOT, 4096 + r0 * Ngz::SLOT) != m * Ngz::SLOT) { perror("read"); return 1; }
        auto t0 = std::chrono::steady_clock::now();
        for (long i = 0; i < m; ++i) if (!z.row(slots.data() + i * Ngz::SLOT, a.data() + i * 160)) { ++esc; memset(a.data() + i * 160, 0xff, 320); }
        auto t1 = std::chrono::steady_clock::now();
        for (long i = 0; i < m; ++i) {
            float v[160];
            for (int k = 0; k < 160; ++k) { uint32_t x = (uint32_t)(raw[i * 320 + 2 * k] | raw[i * 320 + 2 * k + 1] << 8) << 16; memcpy(&v[k], &x, 4); v[k] *= 262144.f; }
            plerows::f2h_n(v, b.data() + i * 160, 160);
        }
        auto t2 = std::chrono::steady_clock::now();
        tz += std::chrono::duration<double>(t1 - t0).count(); tb += std::chrono::duration<double>(t2 - t1).count();
        for (long i = 0; i < m; ++i) if (slots[i * Ngz::SLOT] && memcmp(a.data() + i * 160, b.data() + i * 160, 320)) ++bad;
    }
    printf("%ld rows: %ld mismatches vs the bf16 path, %ld escaped; decode %.0f ns/row NGZ (%.1f us per 16-row token), %.0f ns/row bf16\n",
           n, bad, esc, tz / n * 1e9, tz / n * 16e6, tb / n * 1e9);
    return bad ? 1 : 0;
}
