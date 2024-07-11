//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef HEADER_H
#define HEADER_H

#include <string>

static void FormatHeader(std::ostream& out)
{
  std::string header = R"XXX(
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This is an autogenerated file, we want to ensure that it contains exactly the contents we want to generate
// clang-format off

#ifndef _LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_PTX_GENERATED_H
#define _LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_PTX_GENERATED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/is_unsigned.h>

#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/functions/common.h>
#include <cuda/std/__atomic/functions/cuda_ptx_generated_helper.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_CUDA_COMPILER)
)XXX";

  out << header;
}

static void FormatTail(std::ostream& out)
{
  std::string tail = R"XXX(
#endif // defined(_CCCL_CUDA_COMPILER)

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_PTX_GENERATED_H

// clang-format on
)XXX";

  out << tail;
}

#endif // HEADER_H
