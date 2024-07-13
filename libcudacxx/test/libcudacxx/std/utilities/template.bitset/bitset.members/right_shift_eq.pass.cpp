//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=15000000

// bitset<N>& operator<<=(size_t pos); // constexpr since C++23

#include <cuda/std/bitset>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>

#include "../bitset_test_cases.h"
#include "test_macros.h"

_CCCL_NV_DIAG_SUPPRESS(186)

template <cuda::std::size_t N, cuda::std::size_t Start = 0, cuda::std::size_t End = static_cast<cuda::std::size_t>(-1)>
__host__ __device__ BITSET_TEST_CONSTEXPR bool test_right_shift()
{
  span_stub<const char*> const cases = get_test_cases<N>();
  if (Start >= 9)
  {
    assert(End >= cases.size());
  }
  for (cuda::std::size_t c = Start; c != cases.size() && c != End; ++c)
  {
    for (cuda::std::size_t s = 0; s <= N + 1; ++s)
    {
      cuda::std::bitset<N> v1(cases[c]);
      cuda::std::bitset<N> v2 = v1;
      v1 >>= s;
      _CCCL_DIAG_PUSH
      _CCCL_DIAG_SUPPRESS_ICC(186)
      for (cuda::std::size_t i = 0; i < v1.size(); ++i)
      {
        if (i + s < v1.size())
        {
          _CCCL_DIAG_POP
          {
            assert(v1[i] == v2[i + s]);
          }
        }
        else
        {
          assert(v1[i] == 0);
        }
      }
    }
  }
  return true;
}

__host__ __device__ int main(int, char**)
{
  test_right_shift<0>();
  test_right_shift<1>();
  test_right_shift<31>();
  test_right_shift<32>();
  test_right_shift<33>();
  test_right_shift<63>();
  test_right_shift<64>();
  test_right_shift<65>();
  test_right_shift<1000>(); // not in constexpr because of constexpr evaluation step limits
#if TEST_STD_VER > 2011 && !defined(_CCCL_CUDACC_BELOW_11_4) // 11.4 added support for constexpr device vars
                                                             // needed here
  static_assert(test_right_shift<0>(), "");
  static_assert(test_right_shift<1>(), "");
  static_assert(test_right_shift<31>(), "");
  static_assert(test_right_shift<32>(), "");
  static_assert(test_right_shift<33>(), "");
  static_assert(test_right_shift<63, 0, 3>(), "");
  static_assert(test_right_shift<63, 3, 6>(), "");
  static_assert(test_right_shift<63, 6, 9>(), "");
  static_assert(test_right_shift<63, 9>(), "");
  static_assert(test_right_shift<64, 0, 3>(), "");
  static_assert(test_right_shift<64, 3, 6>(), "");
  static_assert(test_right_shift<64, 6, 9>(), "");
  static_assert(test_right_shift<64, 9>(), "");
  static_assert(test_right_shift<65, 0, 3>(), "");
  static_assert(test_right_shift<65, 3, 6>(), "");
  static_assert(test_right_shift<65, 6, 9>(), "");
  static_assert(test_right_shift<65, 9>(), "");
#endif

  return 0;
}