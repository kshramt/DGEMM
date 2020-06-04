#include <cassert>
#include <cmath>
#include <cstdint>
#include <vector>

namespace dgemm {
using i64 = std::int64_t;

template <typename T>
auto dgemm1(const i64 n, const std::vector<T>& A, const std::vector<T>& B,
            std::vector<T>& C) {
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
