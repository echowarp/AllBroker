// Parity harness: gufo's HashNgramRows, extracted verbatim from ngram.cpp (see make_parity.sh), with stub types.
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <span>
#include <vector>
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
  Config c;
  NgramHistory h;
  std::vector<std::int32_t> toks;
  for (int i = 1; i < argc; ++i) toks.push_back(std::atoi(argv[i]));
  std::vector<std::uint32_t> rows(toks.size() * c.ple_heads);
  HashNgramRows(c, h, toks, rows);
  for (std::size_t i = 0; i < toks.size(); ++i) {
    std::printf("%d", toks[i]);
    for (std::uint32_t k = 0; k < c.ple_heads; ++k) std::printf(" %u", rows[i * c.ple_heads + k]);
    std::printf("\n");
  }
}
