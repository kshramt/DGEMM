#include <immintrin.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

#include <boost/align/aligned_allocator.hpp>

namespace dgemm {
using i64 = std::int64_t;
using double_avx_allocator_t =
    boost::alignment::aligned_allocator<double, alignof(__m256d)>;

constexpr i64 kUnroll{4};
constexpr i64 kBlockSize{32};

template <typename T, typename Allocator>
auto dgemm(const i64 n, const std::vector<T, Allocator>& A,
           const std::vector<T, Allocator>& B, std::vector<T, Allocator>& C) {
  assert(std::pow(n, 2) <= A.size());
  assert(std::pow(n, 2) <= B.size());
  assert(std::pow(n, 2) <= C.size());
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; ++j) {
      T cij = C[i * n + j];
      for (i64 k = 0; k < n; ++k) {
        cij += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = cij;
    }
  }
}

template <typename T, typename Allocator>
auto dgemm_openmp(const i64 n, const std::vector<T, Allocator>& A,
           const std::vector<T, Allocator>& B, std::vector<T, Allocator>& C) {
  assert(std::pow(n, 2) <= A.size());
  assert(std::pow(n, 2) <= B.size());
  assert(std::pow(n, 2) <= C.size());
  #pragma omp parallel for
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; ++j) {
      T cij = C[i * n + j];
      for (i64 k = 0; k < n; ++k) {
        cij += A[i * n + k] * B[k * n + j];
      }
      C[i * n + j] = cij;
    }
  }
}

// Figure 3.23
auto dgemm_avx(const i64 n,
               const std::vector<double, double_avx_allocator_t>& A,
               const std::vector<double, double_avx_allocator_t>& B,
               std::vector<double, double_avx_allocator_t>& C) {
  assert(std::pow(n, 2) <= A.size());
  assert(std::pow(n, 2) <= B.size());
  assert(std::pow(n, 2) <= C.size());
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; j += 4) {
      __m256d c0 = _mm256_load_pd(&C[i * n + j]);
      for (i64 k = 0; k < n; ++k) {
        // c0 += A[i][k]*B[k, j];
        c0 = _mm256_add_pd(c0, _mm256_mul_pd(_mm256_broadcast_sd(&A[i * n + k]),
                                             _mm256_load_pd(&B[k * n + j])));
      }
      // C[i][j] = c0;
      _mm256_store_pd(&C[i * n + j], c0);
    }
  }
}

// Figure 4.80
auto dgemm_avx_unroll(const i64 n,
                      const std::vector<double, double_avx_allocator_t>& A,
                      const std::vector<double, double_avx_allocator_t>& B,
                      std::vector<double, double_avx_allocator_t>& C) {
  assert(std::pow(n, 2) <= A.size());
  assert(std::pow(n, 2) <= B.size());
  assert(std::pow(n, 2) <= C.size());
  for (i64 i = 0; i < n; ++i) {
    for (i64 j = 0; j < n; j += kUnroll * 4) {
      __m256d c[kUnroll];
      for (i64 u = 0; u < kUnroll; ++u) {
        c[u] = _mm256_load_pd(&C[i * n + j + kUnroll * u]);
      }
      for (i64 k = 0; k < n; ++k) {
        __m256d a = _mm256_broadcast_sd(&A[i * n + k]);
        for (i64 u = 0; u < kUnroll; ++u) {
          c[u] = _mm256_add_pd(
              c[u],
              _mm256_mul_pd(a, _mm256_load_pd(&B[k * n + j + kUnroll * u])));
        }
      }
      for (i64 u = 0; u < kUnroll; ++u) {
        _mm256_store_pd(&C[i * n + j + kUnroll * u], c[u]);
      }
    }
  }
}

// Figure 5.21
template <typename T, typename Allocator>
auto do_block(const i64 n, i64 si, i64 sj, i64 sk,
              const std::vector<T, Allocator>& A,
              const std::vector<T, Allocator>& B,
              std::vector<T, Allocator>& C) {
  for (i64 i = si; i < si + kBlockSize; ++i) {
    for (i64 j = sj; j < sj + kBlockSize; ++j) {
      auto cij = C[i * n + j];
      for (i64 k = sk; k < sk + kBlockSize; ++k) {
        cij += A[k * n + k] * B[k * n + j];
      }
      C[i * n + j] = cij;
    }
  }
}

template <typename T, typename Allocator>
auto dgemm_block(const i64 n, const std::vector<T, Allocator>& A,
                 const std::vector<T, Allocator>& B,
                 std::vector<T, Allocator>& C) {
  assert(std::pow(n, 2) <= A.size());
  assert(std::pow(n, 2) <= B.size());
  assert(std::pow(n, 2) <= C.size());
  for (i64 si = 0; si < n; si += kBlockSize) {
    for (i64 sj = 0; sj < n; sj += kBlockSize) {
      for (i64 sk = 0; sk < n; sk += kBlockSize) {
        do_block(n, si, sj, sk, A, B, C);
      }
    }
  }
}

// Figure 5.48
template <typename T, typename Allocator>
auto do_block_avx_unroll(const i64 n, i64 si, i64 sj, i64 sk,
                         const std::vector<T, Allocator>& A,
                         const std::vector<T, Allocator>& B,
                         std::vector<T, Allocator>& C) {
  for (i64 i = si; i < si + kBlockSize; ++i) {
    for (i64 j = sj; j < sj + kBlockSize; j += kUnroll * 4) {
      __m256d c[kUnroll];
      for (i64 u = 0; u < kUnroll; ++u) {
        c[u] = _mm256_load_pd(&C[i * n + j + kUnroll * u]);
      }
      for (i64 k = sk; k < sk + kBlockSize; ++k) {
        __m256d a = _mm256_broadcast_sd(&A[i * n + k]);
        for (i64 u = 0; u < kUnroll; ++u) {
          c[u] = _mm256_add_pd(
              c[u],
              _mm256_mul_pd(a, _mm256_load_pd(&B[k * n + j + kUnroll * u])));
        }
      }
      for (i64 u = 0; u < kUnroll; ++u) {
        _mm256_store_pd(&C[i * n + j + kUnroll * u], c[u]);
      }
    }
  }
}

template <typename T, typename Allocator>
auto dgemm_block_avx_unroll(const i64 n, const std::vector<T, Allocator>& A,
                            const std::vector<T, Allocator>& B,
                            std::vector<T, Allocator>& C) {
  assert(std::pow(n, 2) <= A.size());
  assert(std::pow(n, 2) <= B.size());
  assert(std::pow(n, 2) <= C.size());
  for (i64 si = 0; si < n; si += kBlockSize) {
    for (i64 sj = 0; sj < n; sj += kBlockSize) {
      for (i64 sk = 0; sk < n; sk += kBlockSize) {
        do_block_avx_unroll(n, si, sj, sk, A, B, C);
      }
    }
  }
}

template <typename T, typename Allocator>
auto dgemm_block_avx_unroll_openmp(const i64 n,
                                   const std::vector<T, Allocator>& A,
                                   const std::vector<T, Allocator>& B,
                                   std::vector<T, Allocator>& C) {
  assert(std::pow(n, 2) <= A.size());
  assert(std::pow(n, 2) <= B.size());
  assert(std::pow(n, 2) <= C.size());
#pragma omp parallel for
  for (i64 si = 0; si < n; si += kBlockSize) {
    for (i64 sj = 0; sj < n; sj += kBlockSize) {
      for (i64 sk = 0; sk < n; sk += kBlockSize) {
        do_block_avx_unroll(n, si, sj, sk, A, B, C);
      }
    }
  }
}
}  // namespace dgemm
