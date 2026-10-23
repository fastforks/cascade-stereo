"""CasStereoNet (Cas-GwcNet) inference with custom kernels (generic fp16 NHWC convs with fused conv+BN epilogues).
Same math as models/gwcnet.py GwcNet.forward in eval mode (2 stages, using_ns), with:
  - disparity hypotheses kept in fp32 (upstream uses left.dtype; fp16 would round x - d to 0.5-1 px at x > 1024);
  - stage-1 hypotheses are spatially constant (cur_disp None) -> computed on a [1, 1, D, 1, 1] tensor and broadcast;
  - cost volumes built one disparity plane at a time (2D grid_sample in fp32 per plane, same sampling as the [D*H, W]
    grid of upstream) -> no [B, C, D, H, W] warped-feature tensors (stage 2 at Midd-F: 11 GB fp32 upstream);
  - "trilinear upsample of the cost to full resolution -> softmax -> sum(p * d)" row-chunked (no full-res cost / prob
    volumes), fp32.
Weights / convs: fp16, BN in fp32 epilogues.

    from cuda_inference import prepare   # cwd CasStereoNet/
    prepare(model)                       # model = __models__["gwcnet-c" | "gwcnet"](...), weights loaded, eval
"""
import math
import os

import torch
import torch.nn.functional as F

from .wrap import prepare_wrapped


def _upsample_softmax_regress(cost, D, H, W, dv, rows=64):
    """cost [B, 1, d, h, w] -> fp32 [B, H, W] = sum(softmax(trilinear(cost -> [D, H, W]), 1) * dv); dv [1, D, 1, 1]
    (broadcast) or [B, D, H, W] (full-res hypotheses). Same values as the full computation: each row block upsamples
    only the low-res rows it depends on (+1 row margin, so no clamping at the block borders differs)."""
    B, _, d, h, w = cost.shape
    sd, sh, sw = D // d, H // h, W // w
    assert sd * d == D and sh * h == H and sw * w == W
    out = torch.empty((B, H, W), device=cost.device, dtype=torch.float32)
    full = dv.shape[-1] != 1
    for y0 in range(0, H, rows):
        y1 = min(H, y0 + rows)
        i0 = max(0, math.floor((y0 + 0.5) / sh - 0.5) - 1)
        i1 = min(h, math.floor((y1 - 0.5) / sh - 0.5) + 2)
        up = F.interpolate(cost[:, :, :, i0:i1].float(), scale_factor=(sd, sh, sw), mode="trilinear", align_corners=False,
                           recompute_scale_factor=False).squeeze(1)
        a = y0 - i0 * sh
        p = F.softmax(up[:, :, a:a + (y1 - y0)], dim=1)
        out[:, y0:y1] = torch.sum(p * (dv[:, :, y0:y1] if full else dv), 1)
    return out


VOL_SCALE = 1.0 / 16  # cost volumes are stored scaled (folded back into the first 3D conv's fp32 epilogue scale):
# Cas-GwcNet-gc stage-2 features reach ~1e4 and gwc products ~6e4 (fp16 max 65504) -> inf / fp16-accumulation overflow


def _cost_volume(fl, fr, disp_lo, num_groups, k=VOL_SCALE):
    """gwcnet (dict features {"gwc_feature", "concat_feature"}): GwcNet GetCostVolume = [gwc (x zeroed where x < d),
    concat(x zeroed, y_warped)] -> [B, G + 2 Cc, d, h, w]; gwcnet-c (tensor features, num_groups None): PSMNet GetCostVolume
    = [x (not zeroed), y_warped] -> [B, 2 C, d, h, w]. Features fp16 NHWC; disp_lo fp32 [B or 1, d, h or 1, w or 1];
    one 2D fp32 grid_sample per plane (same sampling as upstream's [D*H, W] grid)."""
    gwc = isinstance(fl, dict)
    if gwc:
        xg, yg, xc, yc = fl["gwc_feature"], fr["gwc_feature"], fl["concat_feature"], fr["concat_feature"]
        y_both = torch.cat((yg, yc), 1).float()
        C, Cc = xg.shape[1], xc.shape[1]
        ch = num_groups + 2 * Cc
    else:
        xc, y_both = fl, fr.float()
        C, Cc = 0, fl.shape[1]
        ch = 2 * Cc
    B, _, h, w = xc.shape
    d = disp_lo.shape[1]
    vol = torch.empty((B, ch, d, h, w), device=xc.device, dtype=torch.float16)
    mw = torch.arange(w, device=xc.device, dtype=torch.float32).view(1, 1, w)
    gy = (torch.arange(h, device=xc.device, dtype=torch.float32) / ((h - 1.0) / 2.0) - 1.0).view(1, h, 1).expand(B, h, w)
    for j in range(d):
        dk = disp_lo[:, j].expand(B, h, w)
        gx = (mw - dk) / ((w - 1.0) / 2.0) - 1.0
        yw = F.grid_sample(y_both, torch.stack((gx, gy), 3), mode="bilinear", padding_mode="zeros", align_corners=True)
        if gwc:
            valid = (mw >= dk).unsqueeze(1)  # upstream: x_warped[:, mw < d] = 0
            vol[:, :num_groups, j] = (xg.float() * valid * yw[:, :C]).view(B, num_groups, C // num_groups, h, w).mean(2) * k
            vol[:, num_groups:num_groups + Cc, j] = xc.float() * valid * k
            vol[:, num_groups + Cc:, j] = yw[:, C:] * k
        else:
            vol[:, :Cc, j] = xc.float() * k
            vol[:, Cc:, j] = yw * k
    return vol


def _ranges_ns(cur_disp, ndisp, interval, ns_size, max_disp):
    """get_cur_disp_range_samples(using_ns=True) up to the low-res hypotheses: -> fp32 [B, ndisp//4, H/4, W/4]."""
    B, H, W = cur_disp.shape
    s = F.interpolate((cur_disp / 4.0).unsqueeze(1), [H // 4, W // 4], mode="bilinear", align_corners=False).squeeze(1)
    dmin = torch.abs(F.max_pool2d(-s, stride=1, kernel_size=ns_size, padding=ns_size // 2))
    dmax = F.max_pool2d(s, stride=1, kernel_size=ns_size, padding=ns_size // 2)
    comp = (ndisp // 4 * interval - torch.abs(dmax - dmin)).clamp(min=0) / 2.0
    lo = (dmin - comp).clamp(min=0, max=max_disp)
    hi = (dmax + comp).clamp(min=0, max=max_disp)
    n = ndisp // 4
    step = (hi - lo) / (n - 1)
    return lo.unsqueeze(1) + torch.arange(n, device=lo.device, dtype=lo.dtype).view(1, -1, 1, 1) * step.unsqueeze(1)


def _forward_eval(self, left, right):
    B, _, H, W = left.shape
    fl_all, fr_all = self.feature_extraction(left), self.feature_extraction(right)
    outputs, pred = {}, None
    for si in range(self.num_stage):
        scale = int(self.stage_infos[f"stage{si + 1}"]["scale"])
        nd = self.ndisps[si]
        if pred is None:  # spatially constant hypotheses
            lo = max(0.0, -nd / 2 * self.disp_interval_pixel[si])
            hi = min(lo + (nd - 1) * self.disp_interval_pixel[si], float(self.maxdisp))
            full = (lo + torch.arange(nd, device=left.device, dtype=torch.float32) * ((hi - lo) / (nd - 1))).view(1, nd, 1, 1)
            low = F.interpolate((full / scale).view(1, 1, nd, 1, 1), [nd // scale, 1, 1], mode="trilinear",
                                align_corners=False).view(1, nd // scale, 1, 1)
        else:
            r4 = _ranges_ns(pred, nd, self.disp_interval_pixel[si], self.ns_size, self.maxdisp)
            full = F.interpolate((r4 * 4.0).unsqueeze(1), [nd, H, W], mode="trilinear", align_corners=False).squeeze(1)
            low = F.interpolate((full / scale).unsqueeze(1), [nd // scale, H // scale, W // scale], mode="trilinear",
                                align_corners=False).squeeze(1)
        fl, fr = fl_all[f"stage{si + 1}"], fr_all[f"stage{si + 1}"]
        cost = _cost_volume(fl, fr, low, self.num_groups[si] if hasattr(self, "num_groups") else None)
        del low
        agg = self.cost_agg[si]
        c0 = agg.dres0(cost)
        del cost
        c0 = agg.dres1(c0) + c0
        out3 = agg.dres4(agg.dres3(agg.dres2(c0)))
        del c0
        pred = _upsample_softmax_regress(agg.classif3(out3), nd, H, W, full)
        del out3, full
        outputs[f"stage{si + 1}"] = {"pred3": pred, "pred": pred}
    return outputs


def prepare(model):
    model.forward = _forward_eval.__get__(model)
    n = prepare_wrapped(model, repo_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for agg in model.cost_agg:  # undo VOL_SCALE in the first conv's epilogue: s * (W * (x k)) = (s / k) * ... -> s / k
        f = agg.dres0[0][0]
        assert f.s is not None, type(f)
        f.s = (f.s / VOL_SCALE).contiguous()
    return n
