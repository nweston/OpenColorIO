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

#include "CudaSupport.h"
#include "Processor.h"

// Include implementations of all other CUDA code
#include "CudaOps.cuh"
#include "ImagePacking.cpp"

OCIO_NAMESPACE_ENTER
{
    __global__ void ApplyKernel(CudaOp ** ops, size_t opCount,
                                GenericImageDesc img)
    {
        int x = blockDim.x * blockIdx.x + threadIdx.x;
        int y = blockDim.y * blockIdx.y + threadIdx.y;
        int pixelIndex = y * img.width + x;

        if (x >= img.width || y >= img.height)
            return;

        // Get one pixel
        // TODO: for packed layouts, we'd get better memory access patterns if
        // each thread did a single channel (or fetched all channels in one
        // transaction as a float4). But that requires digging into the guts of
        // PackRGBAFromImageDesc.
        float pixelData[4];
        int numCopied = 0;

        PackRGBAFromImageDesc(img, pixelData, &numCopied, 1, pixelIndex);

        // Apply ops
        for (int i = 0; i < opCount; i++)
            ops[i]->apply(pixelData);

        // Write result back to global memory
        UnpackRGBAToImageDesc(img, pixelData, 1, pixelIndex);
    }

    __global__ void FreeOpsKernel(CudaOp ** ops, size_t opCount)
    {
        for (int i = 0; i < opCount; i++)
            delete ops[i];
    }

    void ApplyCudaOps(const std::vector<CudaOp *> &cudaOps,
                      GenericImageDesc &img)
    {
        // The CudaOps are already in device memory, but the vector of their
        // pointers isn't. Copy it over to the device.
        size_t opListSize = cudaOps.size() * sizeof(CudaOp *);
        CudaOp **deviceOpList;
        CheckCudaError(cudaMalloc(&deviceOpList, opListSize));
        // Elements of std::vector are guaranteed to be contiguous
        CheckCudaError(cudaMemcpy(deviceOpList, &cudaOps[0], opListSize,
                                  cudaMemcpyHostToDevice));

        // Setup and launch kernel
        dim3 threads(32, 16, 1);
        dim3 blocks((img.width + threads.x - 1) / threads.x,
                    (img.height + threads.y - 1) / threads.y,
                    1);
        ApplyKernel<<<blocks, threads>>>(deviceOpList, cudaOps.size(), img);

        // Wait for kernel to finish before freeing ops
        cudaThreadSynchronize();
        CheckCudaError(cudaGetLastError());

        // Clean up
        // TODO: could keep ops around in the processor. Will speed things up
        // especially if they have LUTs, etc.
        FreeOpsKernel<<<1, 1>>>(deviceOpList, cudaOps.size());
        cudaThreadSynchronize();
        CheckCudaError(cudaGetLastError());
        CheckCudaError(cudaFree(deviceOpList));
    }
}
OCIO_NAMESPACE_EXIT
