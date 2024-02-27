//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// bitset<N>& operator<<=(size_t pos); // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>


#include "../bitset_test_cases.h"
#include "test_macros.h"

_CCCL_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N>
__host__ __device__
TEST_CONSTEXPR_CXX23 bool test_left_shift() {
    span_stub<const char *> const cases = get_test_cases<N>();
    for (cuda::std::size_t c = 0; c != cases.size(); ++c) {
        for (cuda::std::size_t s = 0; s <= N+1; ++s) {
            cuda::std::bitset<N> v1(cases[c]);
            cuda::std::bitset<N> v2 = v1;
            v1 <<= s;
            for (cuda::std::size_t i = 0; i < v1.size(); ++i)
                if (i < s)
                    assert(v1[i] == 0);
                else
                    assert(v1[i] == v2[i-s]);
        }
    }
    return true;
}

int main(int, char**) {
  test_left_shift<0>();
  test_left_shift<1>();
  test_left_shift<31>();
  test_left_shift<32>();
  test_left_shift<33>();
  test_left_shift<63>();
  test_left_shift<64>();
  test_left_shift<65>();
  test_left_shift<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 2020
  static_assert(test_left_shift<0>());
  static_assert(test_left_shift<1>());
  static_assert(test_left_shift<31>());
  static_assert(test_left_shift<32>());
  static_assert(test_left_shift<33>());
  static_assert(test_left_shift<63>());
  static_assert(test_left_shift<64>());
  static_assert(test_left_shift<65>());
#endif

  return 0;
}
