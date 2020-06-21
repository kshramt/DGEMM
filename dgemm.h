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
}  // namespace dgemm
