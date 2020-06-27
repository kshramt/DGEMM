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
    auto run = [&rng](auto n, auto nit) {
      bench(
          "dgemm_block_avx_unroll_openmp", n, nit,
          [](const auto n, const auto& A, const auto& B, auto& C) {
            dgemm::dgemm_block_avx_unroll_openmp(n, A, B, C);
          },
          rng);
    };
    // run(16, 20);
    run(32, 20);
    run(64, 10);
    run(128, 10);
    run(256, 4);
    run(512, 2);
    run(1024, 2);
  }
  {
    std::mt19937 gen{42};
    auto rng = [&gen, &dis]() { return dis(gen); };
    auto run = [&rng](auto n, auto nit) {
      bench(
          "dgemm_block_avx_unroll", n, nit,
          [](const auto n, const auto& A, const auto& B, auto& C) {
            dgemm::dgemm_block_avx_unroll(n, A, B, C);
          },
          rng);
    };
    // run(16, 20);
    run(32, 20);
    run(64, 10);
    run(128, 10);
    run(256, 4);
    run(512, 2);
    run(1024, 2);
  }
  {
    std::mt19937 gen{42};
    auto rng = [&gen, &dis]() { return dis(gen); };
    auto run = [&rng](auto n, auto nit) {
      bench(
          "dgemm_block", n, nit,
          [](const auto n, const auto& A, const auto& B, auto& C) {
            dgemm::dgemm_block(n, A, B, C);
          },
          rng);
    };
    // run(16, 20);
    run(32, 20);
    run(64, 10);
    run(128, 10);
    run(256, 4);
    run(512, 2);
    run(1024, 2);
  }
  {
    std::mt19937 gen{42};
    auto rng = [&gen, &dis]() { return dis(gen); };
    auto run = [&rng](auto n, auto nit) {
      bench(
          "dgemm_avx_unroll", n, nit,
          [](const auto n, const auto& A, const auto& B, auto& C) {
            dgemm::dgemm_avx_unroll(n, A, B, C);
          },
          rng);
    };
    // run(16, 20);
    run(32, 20);
    run(64, 10);
    run(128, 10);
    run(256, 4);
    run(512, 2);
    run(1024, 2);
  }
  {
    std::mt19937 gen{42};
    auto rng = [&gen, &dis]() { return dis(gen); };
    auto run = [&rng](auto n, auto nit) {
      bench(
          "dgemm_avx", n, nit,
          [](const auto n, const auto& A, const auto& B, auto& C) {
            dgemm::dgemm_avx(n, A, B, C);
          },
          rng);
    };
    // run(16, 20);
    run(32, 20);
    run(64, 10);
    run(128, 10);
    run(256, 4);
    run(512, 2);
    run(1024, 2);
  }
  {
    std::mt19937 gen{42};
    auto rng = [&gen, &dis]() { return dis(gen); };
    auto run = [&rng](auto n, auto nit) {
      bench(
          "dgemm_openmp", n, nit,
          [](const auto n, const auto& A, const auto& B, auto& C) {
            dgemm::dgemm_openmp(n, A, B, C);
          },
          rng);
    };
    // run(16, 20);
    run(32, 20);
    run(64, 10);
    run(128, 10);
    run(256, 4);
    run(512, 2);
    run(1024, 2);
  }
  {
    std::mt19937 gen{42};
    auto rng = [&gen, &dis]() { return dis(gen); };
    auto run = [&rng](auto n, auto nit) {
      bench(
          "dgemm", n, nit,
          [](const auto n, const auto& A, const auto& B, auto& C) {
            dgemm::dgemm(n, A, B, C);
          },
          rng);
    };
    // run(16, 20);
    run(32, 20);
    run(64, 10);
    run(128, 10);
    run(256, 4);
    run(512, 2);
    run(1024, 2);
  }
  return 0;
}
