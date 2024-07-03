/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "catch2_test_helper.h"

CUB_TEST("Device inclusive scan works", "[scan][device]")
{
  // Determine temporary device storage requirements for inclusive scan
  cudaStream_t stream{};
  REQUIRE(cudaSuccess == cudaStreamCreate(&stream));
  // example-begin device-inclusive-scan
  thrust::device_vector<int> input{0, -1, 2, -3, 4, -5, 6};
  thrust::device_vector<int> out(input.size());

  void* d_temp_storage{};
  size_t temp_storage_bytes{};

  int init = 1;

  cub::DeviceScan::InclusiveScanInit(
    d_temp_storage,
    temp_storage_bytes,
    input.begin(),
    out.begin(),
    cub::Max{},
    static_cast<int>(input.size()),
    stream,
    init);

  // Allocate temporary storage for inclusive scan
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // Run inclusive prefix sum
  cub::DeviceScan::InclusiveScanInit(
    d_temp_storage,
    temp_storage_bytes,
    input.begin(),
    out.begin(),
    cub::Max{},
    static_cast<int>(input.size()),
    stream,
    init);

  thrust::host_vector<int> expected{1, 1, 2, 2, 4, 4, 6};
  // example-end device-inclusive-scan

  REQUIRE(expected == out);
}
