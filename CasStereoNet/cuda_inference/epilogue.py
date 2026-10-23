"""Conv + epilogue helpers for the generic fp16 path (copied from the IGEV-Stereo fork's core/inference.py):
conv_ep(conv, x, s, t, act, r) = act(conv(x) * s + t + r) as fp16-accumulate cuDNN conv + one epilogue kernel, or one
CUTLASS kernel with fused epilogue for >= 128 output channels. CAS_FUSED=0: cuDNN + separate epilogue pass only.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

FUSED = os.environ.get("CAS_FUSED", "1") == "1"
_FC = [None]


def fused():
    if _FC[0] is None:
        from .cutlass_conv import FusedConvs, EpiArgs
        _FC[0] = (FusedConvs.get(), EpiArgs)
    return _FC[0]


def fused_cfg(P):
    """Fixed size rule (measured once on an RTX 5060 Ti): 128x128x32/3st tile, 64x128x64/4st for small problems."""
    return 0 if P >= 7000 else 1


ACT = {"none": 0, "relu": 1, "leaky": 2, "tanh": 3, "sigmoid": 4, "relu6": 5}
_K = [None]


def kern():
    if _K[0] is None:
        from .kernels import Kernels
        _K[0] = Kernels.get()
    return _K[0]


def _mf(dim):
    return torch.channels_last if dim == 4 else torch.channels_last_3d


def _rows(x):
    """(P, C, row stride) of a dense channels-last tensor, else None."""
    if x.dim() not in (4, 5) or not x.is_contiguous(memory_format=_mf(x.dim())) or x.dtype != torch.float16:
        return None
    C = x.shape[1]
    return x.numel() // C, C, C


def epilogue(y, s=None, t=None, act=0, r=None):
    """In place: y = act(y * s + t + r), y dense channels-last fp16, s/t fp32 [C] or None, r like y or None."""
    ry = _rows(y)
    if ry is None or (r is not None and (_rows(r) is None or r.shape != y.shape)):
        return _epilogue_torch(y, s, t, act, r)
    P, C, st = ry
    k = kern()
    vec = C % 8 == 0
    name = "affine_act8" if vec else "affine_act"
    n = P * (C // 8) if vec else P * C
    k.launch(name, k.linear(n), y, st, 0 if s is None else s, 0 if t is None else t, 0 if r is None else r,
             0 if r is None else C, P, C, act)
    return y


def _epilogue_torch(y, s, t, act, r):
    shp = (1, -1) + (1,) * (y.dim() - 2)
    yf = y.float()
    if s is not None:
        yf = yf * s.view(shp)
    if t is not None:
        yf = yf + t.view(shp)
    if r is not None:
        yf = yf + r.float()
    yf = [lambda v: v, F.relu, lambda v: F.leaky_relu(v, 0.01), torch.tanh, torch.sigmoid, F.relu6][act](yf)
    return y.copy_(yf)


def conv_ep(m, x, s=None, t=None, act=0, r=None, keep_pad=False):
    """conv module m (Conv/ConvTranspose 2d/3d, any groups) without bias, then one epilogue: act(conv*s + t + r).
    t defaults to the conv bias; with s (BN), pass t = BN shift + bias * s (precomputed in prepare)."""
    if x.dtype != m.weight.dtype:  # fp32 island upstream (e.g. an fp32-allocated cost volume) feeding an fp16 conv
        x = x.to(m.weight.dtype).contiguous(memory_format=_mf(x.dim()))
    if t is None and s is None:
        t = m._bias32
    mf = _mf(x.dim())
    if getattr(m, "_fuse", False) and r is None and act in (0, 1) and x.dim() == 4 and x.dtype == torch.float16 \
            and x.shape[0] == 1 and x.is_contiguous(memory_format=torch.channels_last):
        y = _conv_fused(m, x, s, t, act)
        if y is not None:
            return y if y.shape[1] == m.out_channels or keep_pad else y[:, :m.out_channels]
    if getattr(m, "_hacc", False):
        from .cudnn_conv import conv_half_acc
        y = conv_half_acc(x, m.weight, None, list(m.padding), list(m.stride))
    elif getattr(m, "_hacc_t", False):
        from .cudnn_conv import deconv_half_acc
        y = deconv_half_acc(x, m.weight, list(m.padding), list(m.stride))
    elif isinstance(m, (nn.ConvTranspose2d, nn.ConvTranspose3d)):
        f = F.conv_transpose2d if x.dim() == 4 else F.conv_transpose3d
        y = f(x, m.weight, None, m.stride, m.padding, m.output_padding, m.groups, m.dilation).contiguous(memory_format=mf)
    else:
        f = F.conv2d if x.dim() == 4 else F.conv3d
        y = f(x, m.weight, None, m.stride, m.padding, m.dilation, m.groups).contiguous(memory_format=mf)
    if s is not None or t is not None or act or r is not None:
        y = epilogue(y, s, t, act, r)
    if y.shape[1] != m.out_channels and not keep_pad:  # weights zero-padded to a multiple of 8 output channels
        y = y[:, :m.out_channels]
    return y


def _conv_fused(m, x, s, t, act, out=None, out2=None):
    """conv + epilogue as one CUTLASS kernel. out / out2: (buffer, channel offset) destinations inside wider
    channels-last buffers (out2: a second copy with the same row stride)."""
    fc, E = fused()
    _, C, H, W = x.shape
    K = m.weight.shape[0]
    (R, S), (ph, pw), (st, _) = m.kernel_size, m.padding, m.stride
    Ho, Wo = (H + 2 * ph - R) // st + 1, (W + 2 * pw - S) // st + 1
    if out is None:
        y = torch.empty((1, K, Ho, Wo), device=x.device, dtype=x.dtype, memory_format=torch.channels_last)
        yp, ys = y.data_ptr(), K
    else:
        y, c0 = out
        assert y.is_contiguous(memory_format=torch.channels_last) and tuple(y.shape[2:]) == (Ho, Wo)
        yp, ys = y.data_ptr() + 2 * c0, y.shape[1]
    y2p = None if out2 is None else out2[0].data_ptr() + 2 * out2[1]
    e = E(2, act, None, 0, yp, ys, None, ys, y2p, None, 0 if s is None else s.data_ptr(),
          0 if t is None else t.data_ptr(), None, 0, 0)
    rc = fc.run(fused_cfg(Ho * Wo), x, m.weight, H, W, C, K, e, padding=(ph, pw), stride=st)
    if rc == 1:
        m._fuse = False  # geometry not supported -> cuDNN path from now on
        return None
    if rc:
        raise RuntimeError(f"fused conv failed ({rc})")
    return y


def _conv_forward(self, x):
    if x.dtype != self.weight.dtype:  # an fp32 island upstream (e.g. upcast probabilities) feeding an fp16 conv
        x = x.to(self.weight.dtype).contiguous(memory_format=_mf(x.dim()))
    return conv_ep(self, x)


def _bn_st(bn, conv):
    s = bn.weight.float() * torch.rsqrt(bn.running_var.float() + bn.eps)
    t = bn.bias.float() - bn.running_mean.float() * s
    if conv is not None and conv.bias is not None:
        t = t + conv.bias.float() * s
    return s.cuda().contiguous(), t.cuda().contiguous()


def _in_stats(x):
    var, mean = torch.var_mean(x.float(), dim=tuple(range(2, x.dim())), keepdim=True, unbiased=False)
    return mean, var


def _in_module_forward(self, x):  # bare InstanceNorm2d(affine=False) (followed by ReLU in the Sequential)
    mean, var = _in_stats(x)
    rstd = torch.rsqrt(var + self.eps)
    return epilogue(x.contiguous(memory_format=torch.channels_last), rstd.view(-1).contiguous(),
                    (-mean * rstd).view(-1).contiguous(), 0)


def _bn_module_forward(self, x):  # standalone BN (or timm BatchNormAct2d) on an NHWC tensor
    return epilogue(x, self._s, self._t, self._act)
