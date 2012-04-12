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

#include "Op.h"

OCIO_NAMESPACE_ENTER
{
    namespace
    {
        DEVICE void ApplyClampExponent(float* rgbaBuffer, long numPixels,
                                const float* exp4)
        {
            for(long pixelIndex=0; pixelIndex<numPixels; ++pixelIndex)
            {
                rgbaBuffer[0] = powf( max(0.0f, rgbaBuffer[0]), exp4[0]);
                rgbaBuffer[1] = powf( max(0.0f, rgbaBuffer[1]), exp4[1]);
                rgbaBuffer[2] = powf( max(0.0f, rgbaBuffer[2]), exp4[2]);
                rgbaBuffer[3] = powf( max(0.0f, rgbaBuffer[3]), exp4[3]);

                rgbaBuffer += 4;
            }
        }

        const int FLOAT_DECIMALS = 7;
    }
}
OCIO_NAMESPACE_EXIT
