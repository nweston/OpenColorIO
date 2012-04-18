/*
Copyright (c) 2012 Sony Pictures Imageworks Inc., et al.
All Rights Reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:
* Redistributions of source code must retain the above copyright
  notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
  notice, this list of conditions and the following disclaimer in the
  documentation and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef INCLUDE_OCIO_CUDA_SUPPORT_H
#define INCLUDE_OCIO_CUDA_SUPPORT_H

#if OCIO_BUILD_CUDA
#include <cuda_runtime_api.h>
#endif

#include <OpenColorIO/OpenColorIO.h>

#ifdef __CUDACC__

#define DEVICE __device__
#define HOST __host__
#define CUDASTATIC static

#else

#define DEVICE
#define HOST
#define CUDASTATIC

#endif

OCIO_NAMESPACE_ENTER
{
  // std::max isn't available as a device function in CUDA, so we need to
  // reimplement it
  template <class T>
  HOST DEVICE const T& max ( const T& a, const T& b ) {
#ifdef __CUDACC__
      return (a<b)?b:a;
#else
      return std::max(a, b);
#endif
  }

#if OCIO_BUILD_CUDA
  // Check error code and throw an exception if != cudaSuccess
  void CheckCudaError(cudaError_t err);

#endif
}
OCIO_NAMESPACE_EXIT

#endif