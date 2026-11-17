// C++ client test: gufo's NgramBroker (compiled from its source) against a broker, checked byte-for-byte against
// direct pread()s of the table, with rows hashed by gufo's own HashNgramRows (extracted verbatim below).
//   broker_client_test <socket> <table_name> <table_file> <table_offset> [runs=300] [tokens=16]
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <vector>
#include "src/models/qwen38_flash_next/ngram_broker.cpp"
struct Config {
  static constexpr std::uint32_t kMaxPleHeads = 16;
  static constexpr std::uint32_t kMaxPleNgram = 3;
  std::uint32_t ple_ngram_size{3}, ple_heads_per_ngram{8}, ple_heads{16};
  std::int64_t ple_eos_token{248044};
  std::array<std::uint64_t, kMaxPleNgram> ple_multipliers{23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
  std::array<std::uint32_t, kMaxPleHeads> ple_head_offsets{0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275};
  std::array<std::uint32_t, kMaxPleHeads> ple_head_vocab{20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077, 20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171};
};
struct NgramHistory {
  static constexpr std::int32_t kNone = -1;
  std::array<std::int32_t, Config::kMaxPleNgram - 1> prev{kNone, kNone};
};
void HashNgramRows(const Config& c, NgramHistory& history,
                   std::span<const std::int32_t> tokens,
                   std::span<std::uint32_t> rows) {
  const std::uint32_t n = c.ple_ngram_size;
  const std::int64_t eos = c.ple_eos_token;
  std::array<std::uint64_t, Config::kMaxPleNgram> ctx{};
  for (std::size_t i = 0; i < tokens.size(); ++i) {
    // The token's own EOS does not cut its context, only an older one does.
    ctx[0] = static_cast<std::uint64_t>(tokens[i]);
    bool cut = false;
    for (std::uint32_t s = 1; s < n; ++s) {
      const std::int32_t t = cut ? NgramHistory::kNone : history.prev[s - 1];
      cut = cut || t < 0 || t == eos;
      ctx[s] = static_cast<std::uint64_t>(cut ? eos : t);
    }
    std::uint32_t* out = rows.data() + i * c.ple_heads;
    for (std::uint32_t order = 2; order <= n; ++order) {
      std::uint64_t mixed = ctx[0] * c.ple_multipliers[0];
      for (std::uint32_t j = 1; j < order; ++j) {
        mixed ^= ctx[j] * c.ple_multipliers[j];
      }
      const std::uint32_t base = (order - 2) * c.ple_heads_per_ngram;
      for (std::uint32_t g = 0; g < c.ple_heads_per_ngram; ++g) {
        const std::uint32_t h = base + g;
        out[h] = static_cast<std::uint32_t>(mixed % c.ple_head_vocab[h]) +
                 c.ple_head_offsets[h];
      }
    }
    for (std::size_t s = history.prev.size(); s-- > 1;) {
      history.prev[s] = history.prev[s - 1];
    }
    history.prev[0] = tokens[i];
  }
}
int main(int argc, char** argv) {
  using gufo::models::qwen38_flash_next::NgramBroker;
  if (argc < 5) { std::fprintf(stderr, "usage: broker_client_test <socket> <table> <table_file> <offset> [runs] [tokens]\n"); return 2; }
  const int runs = argc > 5 ? std::atoi(argv[5]) : 300, per = argc > 6 ? std::atoi(argv[6]) : 16;
  const std::uint64_t base = std::strtoull(argv[4], nullptr, 10);
  const std::uint32_t rb = 90;
  std::string err;
  auto b = NgramBroker::Connect(argv[1], argv[2], 320001536, rb, &err);
  if (!b) { std::fprintf(stderr, "connect: %s\n", err.c_str()); return 1; }
  const int fd = ::open(argv[3], O_RDONLY);
  Config c;
  std::mt19937_64 rng(12345);
  auto tok = [&] { return rng() % 12 == 0 ? 248044 : static_cast<std::int32_t>(rng() % 248320); };
  long checked = 0, bad = 0;
  std::vector<double> lat;
  for (int r = 0; r < runs; ++r) {
    NgramHistory h;
    h.prev = {rng() % 5 == 0 ? -1 : tok(), rng() % 5 == 0 ? -1 : tok()};
    const NgramHistory before = h;
    std::vector<std::int32_t> t(per);
    for (auto& x : t) x = tok();
    std::vector<std::uint32_t> rows(t.size() * c.ple_heads);
    HashNgramRows(c, h, t, rows);
    const auto t0 = std::chrono::steady_clock::now();
    const std::uint8_t* src = nullptr;
    if (!b->StartGather(before.prev.data(), t) || !b->WaitGather(rows.size(), &src, &err)) { std::fprintf(stderr, "gather: %s\n", err.c_str()); return 1; }
    lat.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    std::vector<std::uint8_t> want(rb);
    for (std::size_t i = 0; i < rows.size(); ++i) {
      if (::pread(fd, want.data(), rb, base + std::uint64_t{rows[i]} * rb) != rb) { std::fprintf(stderr, "pread\n"); return 1; }
      ++checked;
      bad += std::memcmp(want.data(), src + i * rb, rb) != 0;
    }
  }
  std::sort(lat.begin(), lat.end());
  std::printf("C++ client: %ld rows compared with direct reads, %ld mismatches; gather p50 %.0f us, p99 %.0f us; broker disk rows %llu\n",
              checked, bad, lat[lat.size() / 2], lat[lat.size() * 99 / 100], (unsigned long long)b->DiskRows());
  return bad == 0 ? 0 : 1;
}
