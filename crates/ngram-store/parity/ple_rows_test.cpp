// CPU check + timing of the split engine's PLE row source (engine/ple_rows.h), no GPU.
//   ple_rows_test <tokens.u32> <offset> <ntok> <per> [check=1] [prefetch_ahead=0] [runs=1] [gap_us=0]
// Walks ntok tokens of a token stream from `offset` in requests of `per` tokens (1 = decode, 8 = a verify,
// 2048 = a prefill chunk), with the n-gram history carried as the engine does. Each request is split into `runs`
// runs (multi-session form). With PLE_BROKER=<socket> the rows come from the broker, otherwise from pread.
// check=1 compares every row with a direct pread of the table (outside the timed part). prefetch_ahead=k sends a
// PREFETCH for the request k ahead before each gather (the engine's chunk-ahead), -1 for the request itself (its
// tokens known early: MTP drafts). gap_us sleeps that long before each gather (the GPU work in between; untimed).
// Prints host us/token, broker disk rows/token and this process's storage reads (/proc/self/io read_bytes) per token
// during the timed fetches (the pread path's real disk traffic; page-cache hits read nothing).
#include "../../engine/ple_rows.h"
#include <algorithm>
#include <fstream>
#include <sstream>

static uint64_t io_read_bytes() {
    std::ifstream f("/proc/self/io");
    std::string k; uint64_t v;
    while (f >> k >> v) if (k == "read_bytes:") return v;
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 5) { fprintf(stderr, "usage: ple_rows_test <tokens.u32> <offset> <ntok> <per> [check] [prefetch_ahead] [runs]\n"); return 2; }
    const long off = atol(argv[2]), ntok = atol(argv[3]);
    const int per = atoi(argv[4]), check = argc > 5 ? atoi(argv[5]) : 1, ahead = argc > 6 ? atoi(argv[6]) : 0,
              nruns = argc > 7 ? atoi(argv[7]) : 1, gap_us = argc > 8 ? atoi(argv[8]) : 0;
    std::vector<int32_t> all;
    {
        FILE* f = fopen(argv[1], "rb");
        if (!f) { perror(argv[1]); return 1; }
        fseek(f, (off - 2 > 0 ? off - 2 : 0) * 4, SEEK_SET);
        all.resize(ntok + 2 + (size_t)per * (std::max(ahead, 0) + 1));
        all.resize(fread(all.data(), 4, all.size(), f));
        fclose(f);
        for (auto& t : all) if (t < 0) t = 248044;   // the corpus files separate documents with -1: EOS (the engine never sends negative ids)
    }
    const int lead = off >= 2 ? 2 : (int)off;   // history tokens before the walk
    plerows::Source src;
    if (!src.init("$HOME/models/qwen3.8-flash-next/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf", 682453280)) return 1;
    Ple ple;
    for (int i = 0; i < lead; ++i) { uint32_t r[16]; ple.rows(all[i], r); }
    const int32_t* toks = all.data() + lead;
    long n = std::min<long>(ntok, (long)all.size() - lead - (long)per * std::max(ahead, 0));
    n -= n % per;
    std::vector<uint32_t> rows((size_t)per * 16);
    std::vector<uint8_t> got((size_t)per * src.tok_bytes()), want(got.size()), raw((size_t)per * 16 * src.row_bytes());
    long bad = 0, checked = 0;
    uint64_t io = 0;
    std::vector<double> lat;
    for (long i = 0; i < n; i += per) {
        if (ahead == -1) {
            plerows::Run r{{ple.prev[0], ple.prev[1]}, toks + i, per};
            src.prefetch(&r, 1);
        }
        if (ahead > 0) {   // prefetch the request `ahead` requests later (its history: the tokens before it)
            const long j = i + (long)per * ahead;
            if (j + per <= (long)all.size() - lead) {
                plerows::Run r;
                r.prev[0] = toks[j - 1]; r.prev[1] = j >= 2 ? toks[j - 2] : ple.prev[0];
                r.toks = toks + j; r.n = per;
                src.prefetch(&r, 1);
            }
        }
        if (gap_us > 0) usleep(gap_us);
        std::vector<plerows::Run> runs;
        const int each = (per + nruns - 1) / nruns;
        for (int a = 0; a < per; a += each) {
            plerows::Run r;
            r.prev[0] = ple.prev[0]; r.prev[1] = ple.prev[1];
            r.toks = toks + i + a; r.n = std::min(each, per - a);
            for (int m = 0; m < r.n; ++m) ple.rows(r.toks[m], rows.data() + (size_t)(a + m) * 16);
            runs.push_back(r);
        }
        const uint64_t io0 = io_read_bytes();
        const double t0 = plerows::now_s();
        src.fetch(runs.data(), (int)runs.size(), rows.data(), got.data());
        lat.push_back((plerows::now_s() - t0) * 1e6);
        io += io_read_bytes() - io0;
        if (check) {
            // the same rows by pread (and the same host decode for k6 / bf16), compared row by row
            src.pread_rows(rows.data(), per * 16, src.f16() ? raw.data() : want.data());
            if (src.f16()) src.decode(raw.data(), per, (uint16_t*)want.data());
            const int ob = src.tok_bytes() / 16;
            for (int r = 0; r < per * 16; ++r) bad += memcmp(got.data() + (size_t)r * ob, want.data() + (size_t)r * ob, ob) != 0;
            checked += per * 16;
        }
    }
    std::sort(lat.begin(), lat.end());
    printf("%s per=%d runs=%d ahead=%d: %ld tokens, host %.1f us/token (gather p50 %.0f us, p99 %.0f us), broker disk rows %.2f/token, storage reads %.2f KiB/token",
           src.broker() ? "broker" : "pread", per, nruns, ahead, n, src.host_s * 1e6 / n, lat[lat.size() / 2], lat[lat.size() * 99 / 100],
           (double)src.disk_rows / n, io / 1024.0 / n);
    if (check) printf(", %ld rows compared, %ld mismatches", checked, bad);
    printf("\n");
    return bad ? 1 : 0;
}
