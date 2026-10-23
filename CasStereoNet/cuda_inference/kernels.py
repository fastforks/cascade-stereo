"""Custom CUDA kernels of epilogue_kernels.cu for PyTorch (loader adapted from RAFT-Stereo exp 002 cuda_kernels.py):
shipped prebuilt as a fatbin (prebuilt/, sm_80/86/89/90/120 SASS + compute_80 PTX, so no nvcc is needed); only if
it is missing (kernel source edited) a per-arch cubin is compiled once with CUDA_HOME's nvcc into
~/.cache/cascade-stereo-torch. Loaded through the driver API, launched on torch's current stream. Size-independent.
Rebuild the shipped fatbin after editing epilogue_kernels.cu:  python -m cuda_inference.kernels  (from CasStereoNet/)
"""
import ctypes
import hashlib
import os
import subprocess

import torch

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "epilogue_kernels.cu")
_PREBUILT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prebuilt")
_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "cascade-stereo-torch")

# p: pointer (tensor or int address), i: int32, l: int64, f: float32
SIGNATURES = {
    "affine_act": "pipppilii",
    "affine_act8": "pipppilii",
}
_CTYPE = {"i": ctypes.c_int32, "l": ctypes.c_int64, "f": ctypes.c_float, "u": ctypes.c_uint32}


def _nvcc(arch_args, out):
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    subprocess.run([os.path.join(cuda_home, "bin", "nvcc"), *arch_args, "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                    f"-I{cuda_home}/include", "-o", out + ".tmp", _SRC], check=True)
    os.replace(out + ".tmp", out)


def _src_tag():
    return hashlib.sha1(open(_SRC, "rb").read()).hexdigest()[:16]


def build_prebuilt(archs=(80, 86, 89, 90, 120)):
    """Build the shipped fatbin for the current kernel source (run after editing epilogue_kernels.cu)."""
    os.makedirs(_PREBUILT, exist_ok=True)
    for f in os.listdir(_PREBUILT):
        if f.startswith("epilogue_kernels_"):
            os.remove(os.path.join(_PREBUILT, f))
    out = os.path.join(_PREBUILT, f"epilogue_kernels_{_src_tag()}.fatbin")
    gencode = [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs] + ["-gencode=arch=compute_80,code=compute_80"]
    _nvcc(["-fatbin", *gencode], out)
    return out


def _check(res, what):
    if res != 0:
        raise RuntimeError(f"CUDA driver error {res} in {what}")


class Kernels:
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        torch.cuda.init()
        major, minor = torch.cuda.get_device_capability()
        # shipped fatbin (sm_80..sm_120 SASS + compute_80 PTX), else a per-arch cubin compiled once with nvcc
        cubin = os.path.join(_PREBUILT, f"epilogue_kernels_{_src_tag()}.fatbin")
        if not os.path.exists(cubin):
            tag = hashlib.sha1(open(_SRC, "rb").read() + f"sm_{major}{minor}".encode()).hexdigest()[:16]
            cubin = os.path.join(_CACHE, f"epilogue_kernels_{tag}.cubin")
        if not os.path.exists(cubin):
            os.makedirs(_CACHE, exist_ok=True)
            _nvcc(["-cubin", f"-arch=sm_{major}{minor}"], cubin)
        self.lib = ctypes.CDLL("libcuda.so.1")
        self.lib.cuLaunchKernel.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 7 + [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        torch.zeros(1, device="cuda")  # make sure the primary context is current
        self.module = ctypes.c_void_p()
        self._image = open(cubin, "rb").read()
        _check(self.lib.cuModuleLoadData(ctypes.byref(self.module), self._image), "cuModuleLoadData")
        self.funcs = {}
        for name in SIGNATURES:
            f = ctypes.c_void_p()
            _check(self.lib.cuModuleGetFunction(ctypes.byref(f), self.module, name.encode()), f"cuModuleGetFunction {name}")
            self.funcs[name] = f

    @staticmethod
    def linear(n):
        """Rust's Launch::linear: 256 threads, grid-stride loops, at most 65535*8 blocks."""
        return (min(max((n + 255) // 256, 1), 65535 * 8), 1, 1), (256, 1, 1)

    def launch(self, name, cfg, *args):
        grid, block = cfg
        sig = SIGNATURES[name]
        assert len(sig) == len(args), f"{name}: expected {len(sig)} args, got {len(args)}"
        holders = []
        for t, a in zip(sig, args):
            if t == "p":
                holders.append(ctypes.c_void_p(a.data_ptr() if isinstance(a, torch.Tensor) else (int(a) or None)))
            else:
                holders.append(_CTYPE[t](a))
        arr = (ctypes.c_void_p * len(holders))(*[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders])
        stream = torch.cuda.current_stream().cuda_stream
        _check(self.lib.cuLaunchKernel(self.funcs[name], *grid, *block, 0, stream, arr, None), f"launch {name}")


if __name__ == "__main__":
    print(build_prebuilt())
