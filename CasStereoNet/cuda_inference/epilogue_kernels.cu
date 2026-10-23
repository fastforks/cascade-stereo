// Conv epilogue kernels (NHWC fp16, size-independent), from the IGEV-Stereo fork's core/igev_kernels.cu.
// Loaded by kernels.py (prebuilt fatbin), launched on torch's current stream.
#include <cuda_fp16.h>
#include <stdint.h>

#define GRID_STRIDE(i, n) for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < (n); i += (long long)gridDim.x * blockDim.x)

__device__ __forceinline__ float act_f(float v, int mode) {
    switch (mode) {
        case 1: return fmaxf(v, 0.f);                      // relu
        case 2: return v >= 0.f ? v : 0.01f * v;           // LeakyReLU(0.01)
        case 3: return tanhf(v);
        case 4: return 1.f / (1.f + __expf(-v));
        case 5: return fminf(fmaxf(v, 0.f), 6.f);          // relu6
        default: return v;
    }
}

// y[p, c] = act(y[p, c] * s[c] + t[c] (+ r[p, c]))   (NHWC rows; s / t / r may be null). In place.
extern "C" __global__ void affine_act(__half* y, int ystride, const float* __restrict__ s, const float* __restrict__ t,
                                      const __half* __restrict__ r, int rstride, long long P, int C, int mode) {
    GRID_STRIDE(i, P * C) {
        long long p = i / C;
        int c = (int)(i - p * C);
        float v = __half2float(y[p * ystride + c]);
        if (s) v *= s[c];
        if (t) v += t[c];
        if (r) v += __half2float(r[p * rstride + c]);
        y[p * ystride + c] = __float2half(act_f(v, mode));
    }
}

// 8 channels per thread, 16-byte accesses (C, strides multiple of 8, 16-byte aligned base pointers)
extern "C" __global__ void affine_act8(__half* y, int ystride, const float* __restrict__ s, const float* __restrict__ t,
                                       const __half* __restrict__ r, int rstride, long long P, int C, int mode) {
    const int CV = C / 8;
    GRID_STRIDE(i, P * CV) {
        long long p = i / CV;
        int c0 = (int)(i - p * CV) * 8;
        uint4 raw = *reinterpret_cast<const uint4*>(y + p * ystride + c0);
        __half* hv = reinterpret_cast<__half*>(&raw);
        uint4 rraw;
        __half* rv = reinterpret_cast<__half*>(&rraw);
        if (r) rraw = *reinterpret_cast<const uint4*>(r + p * rstride + c0);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            float v = __half2float(hv[k]);
            if (s) v *= s[c0 + k];
            if (t) v += t[c0 + k];
            if (r) v += __half2float(rv[k]);
            hv[k] = __float2half(act_f(v, mode));
        }
        *reinterpret_cast<uint4*>(y + p * ystride + c0) = raw;
    }
}
