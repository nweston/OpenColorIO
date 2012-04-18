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

#include "CudaOps.h"

#include "CudaSupport.h"
#include "ExponentOpsInternal.h"

OCIO_NAMESPACE_ENTER
{
    CudaOp::~CudaOp()
    {}

    namespace
    {
        class CudaExponentOp : public CudaOp
        {
        public:
            DEVICE CudaExponentOp(const float * exp4);
            DEVICE virtual ~CudaExponentOp();
            DEVICE virtual void apply(float* rgbaBuffer) const;
        private:
            float m_exp4[4];
        };
    }

    namespace {
        CudaExponentOp::CudaExponentOp(const float * exp4) : CudaOp()
        {
            for(int i=0; i<4; ++i)
                m_exp4[i] = exp4[i];
        }

        CudaExponentOp::~CudaExponentOp()
        { }

        void CudaExponentOp::apply(float* rgbaBuffer) const
        {
            if(!rgbaBuffer) return;

            ApplyClampExponent(rgbaBuffer, 1, m_exp4);
        }
    }

    __device__ volatile static CudaExponentOp * newExponentOp;
    __global__ void makeExponentOpKernel(float4 exp4)
    {
        newExponentOp = new CudaExponentOp(reinterpret_cast<float *>(&exp4));
    }

    CudaOp * makeCudaExponentOp(const float * exp4)
    {
        // Create a new op on the device heap
        makeExponentOpKernel<<<1, 1>>>(*reinterpret_cast<const float4 *>(exp4));

        // Copy the address of the new object back to the CPU
        CudaExponentOp *host_result = NULL;
        CheckCudaError(cudaMemcpyFromSymbol(&host_result, newExponentOp,
                                            sizeof(CudaExponentOp *)));

        return host_result;
    }
 }
OCIO_NAMESPACE_EXIT
