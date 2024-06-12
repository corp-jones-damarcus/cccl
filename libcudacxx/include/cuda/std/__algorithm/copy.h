//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023-24 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ALGORITHM_COPY_H
#define _LIBCUDACXX___ALGORITHM_COPY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/iterator_operations.h>
#include <cuda/std/__algorithm/unwrap_iter.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/detail/libcxx/include/cstdint>
#include <cuda/std/detail/libcxx/include/cstdlib>
#include <cuda/std/detail/libcxx/include/cstring>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _AlgPolicy, class _InputIterator, class _OutputIterator>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14
  pair<_InputIterator, _OutputIterator>
  __copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  for (; __first != __last; ++__first, (void) ++__result)
  {
    *__result = *__first;
  }
  return {__last, __result};
}

template <class _Tp, class _Up>
inline _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 bool
__dispatch_memmove(_Up* __result, _Tp* __first, const size_t __n)
{
  // This is a pessimisation, but there's no way to do the code path detection correctly before GCC 9.0.
  // __builtin_memmove is also illegal in constexpr there, so... just always assume we are constant evaluated,
  // and let the optimizer *maybe* recover some of the perf.
#if defined(_CCCL_COMPILER_GCC) && _GNUC_VER < 900
  return false;
#endif

  if (__libcpp_is_constant_evaluated())
  {
    return false;
  }
  else
  {
    // For now, we only ever use memmove on host
    // clang-format off
    NV_IF_ELSE_TARGET(NV_IS_HOST, (
      _CUDA_VSTD::memmove(__result, __first, __n * sizeof(_Up));
      return true;
    ),(
      return false;
    ))
    // clang-format on
  }
}

template <class _Tp, class _Up>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 bool
__constexpr_tail_overlap_fallback(_Tp* __first, _Up* __needle, _Tp* __last)
{
  while (__first != __last)
  {
    if (__first == __needle)
    {
      return true;
    }
    ++__first;
  }
  return false;
}

template <class _Tp, class _Up>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 bool
__constexpr_tail_overlap(_Tp* __first, _Up* __needle, _Tp* __last)
{
#if __has_builtin(__builtin_constant_p) || defined(_CCCL_COMPILER_GCC)
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return __builtin_constant_p(__first < __needle) && __first < __needle;),
                    (return __constexpr_tail_overlap_fallback(__first, __needle, __last);))
#else
  return __constexpr_tail_overlap_fallback(__first, __needle, __last);
#endif
}

template <class _AlgPolicy,
          class _Tp,
          class _Up,
          __enable_if_t<_CCCL_TRAIT(is_same, __remove_const_t<_Tp>, _Up), int> = 0,
          __enable_if_t<_CCCL_TRAIT(is_trivially_copy_assignable, _Up), int>   = 0>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 pair<_Tp*, _Up*>
__copy(_Tp* __first, _Tp* __last, _Up* __result)
{
  const ptrdiff_t __n = __last - __first;
  if (__n > 0)
  {
    if (__dispatch_memmove(__result, __first, __n))
    {
      return {__last, __result + __n};
    }
    if ((!__libcpp_is_constant_evaluated() && __first < __result)
        || __constexpr_tail_overlap(__first, __result, __last))
    {
      for (ptrdiff_t __i = __n; __i > 0; --__i)
      {
        *(__result + __i - 1) = *(__first + __i - 1);
      }
    }
    else
    {
      for (ptrdiff_t __i = 0; __i < __n; ++__i)
      {
        *(__result + __i) = *(__first + __i);
      }
    }
  }
  return {__last, __result + __n};
}

template <class _InputIterator, class _OutputIterator>
inline _LIBCUDACXX_HIDE_FROM_ABI _LIBCUDACXX_INLINE_VISIBILITY _CCCL_CONSTEXPR_CXX14 _OutputIterator
copy(_InputIterator __first, _InputIterator __last, _OutputIterator __result)
{
  return _CUDA_VSTD::__copy<_ClassicAlgPolicy>(__unwrap_iter(__first), __unwrap_iter(__last), __unwrap_iter(__result))
    .second;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ALGORITHM_COPY_H
