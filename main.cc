#include "dgemm.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

namespace {
using i64 = std::int64_t;

template <typename Fn>
auto bench(const std::string& msg, const i64 n, const i64 n_iter,
           const Fn& dgemm) {
  std::vector<double> A(std::pow(n, 2)), B(std::pow(n, 2)), C(std::pow(n, 2));
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
  std::mt19937 gen{42};
  std::normal_distribution<double> dis{0, 1};

  bench("dgemm1", 200, 5,
        [](const auto n, const auto& A, const auto& B, auto& C) {
          dgemm::dgemm1(n, A, B, C);
        });
  return 0;
}
