//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16

// <cuda/std/variant>

// template <class ...Types> class variant;

// constexpr size_t index() const noexcept;

#include <cuda/std/cassert>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "archetypes.h"
#include "test_macros.h"
#include "variant_test_helpers.h"


int main(int, char**) {
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v;
    static_assert(v.index() == 0, "");
  }
  {
    using V = cuda::std::variant<int, long>;
    V v;
    assert(v.index() == 0);
  }
  {
    using V = cuda::std::variant<int, long>;
    constexpr V v(cuda::std::in_place_index<1>);
    static_assert(v.index() == 1, "");
  }
  /*{
    using V = cuda::std::variant<int, cuda::std::string>;
    V v("abc");
    assert(v.index() == 1);
    v = 42;
    assert(v.index() == 0);
  }*/
#ifndef TEST_HAS_NO_EXCEPTIONS
  {
    using V = cuda::std::variant<int, MakeEmptyT>;
    V v;
    assert(v.index() == 0);
    makeEmpty(v);
    assert(v.index() == cuda::std::variant_npos);
  }
#endif

  return 0;
}
