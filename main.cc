#include "dgemm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

#include <boost/align/aligned_allocator.hpp>

namespace {
using i64 = std::int64_t;

template <typename Fn, typename Rng>
auto bench(const std::string& msg, const i64 n, const i64 n_iter,
           const Fn& dgemm, Rng& rng) {
  std::vector<double, dgemm::double_avx_allocator_t> A(std::pow(n, 2)),
      B(std::pow(n, 2)), C(std::pow(n, 2));
  std::generate(A.begin(), A.end(), rng);
  std::generate(B.begin(), B.end(), rng);
  std::generate(C.begin(), C.end(), rng);
  std::vector<double> dts(n_iter);

  for (i64 i_iter = 0; i_iter < n_iter; ++i_iter) {
    auto t0 = std::chrono::steady_clock::now();
    dgemm(n, A, B, C);
    auto t1 = std::chrono::steady_clock::now();
    dts[i_iter] = std::chrono::duration<double>(t1 - t0).count();
  }
  auto offset = (1 < n_iter) ? 1 : 0;
  auto mean =
      std::accumulate(dts.begin() + offset, dts.end(), static_cast<double>(0)) /
      (dts.size() - offset);
  // todo: std err
  std::cout << msg << "\t" << n << "\t" << mean << std::endl;
}
}  // namespace

int main() {
  // random_device rdv;
  // std::mt19937 gen{rdv()};
  std::normal_distribution<double> dis{0, 1};

  {
    std::mt19937 gen{42};
    auto rng = [&gen, &dis]() { return dis(gen); };
    auto run = [&rng](auto n) {
      bench(
          "dgemm", n, 5,
          [](const auto n, const auto& A, const auto& B, auto& C) {
            dgemm::dgemm(n, A, B, C);
          },
          rng);
    };
    run(12);
    run(36);
    run(108);
    run(324);
    run(972);
    // run(2430);
  }
  return 0;
}
