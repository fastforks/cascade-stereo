# Cascade-Stereo (CasStereoNet): faster inference fork

> **You are on `portable`.** Research fork of [alibaba/cascade-stereo](https://github.com/alibaba/cascade-stereo) (base `2ae7da6`, 2022-12-24; `main` mirrors upstream). Two branches:
> - [`portable`](https://github.com/fastforks/cascade-stereo/tree/portable): plain PyTorch, exports to ONNX / TensorRT, no build step.
> - [`cuda-kernels`](https://github.com/fastforks/cascade-stereo/tree/cuda-kernels): `portable` plus custom CUDA kernels (prebuilt for sm_80/86/89/90/120 + PTX; without a matching prebuilt they are JIT-compiled with the `nvcc` of `CUDA_HOME`; fp16-accumulate convs need `pip install nvidia-cudnn-frontend`).
>
> Both branches include `grid_sample(align_corners=True)`, the PyTorch 1.2 default the code was written for (upstream on current PyTorch shifts the warp by half a pixel).
> Licence, as upstream: code MIT · weights: no licence stated upstream

Cas-GwcNet, released SceneFlow checkpoints, ndisps 48,24, disp_inter_r 4,1; upstream = fp32 (with the align_corners fix); portable = fp16 autocast, fp32 cost volume; cuda-kernels = fp16 convs with fused fp32 BN epilogues. RTX 5060 Ti 16 GB, torch 2.14.1+cu130, 2026-10-23. Median of 10 runs after warm-up; memory = torch peak allocated / reserved; startup = process launch to model ready; EPE on all 27 ETH3D / 15 Middlebury-H training pairs. TensorRT: not measured.

Cas-GwcNet-c (`--model gwcnet-c`):

| version | KITTI ms | Midd-H ms | speed-up at H | peak MB at H (alloc / reserved) | startup / new size | ETH3D EPE | Midd-H EPE |
|---|---|---|---|---|---|---|---|
| upstream | 194.2 | 655.6 | 1.00x | 4839 / 5454 | 1.3 s / 0 ms | 0.513 | 8.568 |
| **portable** | 114.3 | 387.5 | 1.69x | 1948 / 2116 | 1.3 s / 38 ms | 0.513 | 8.569 |
| **cuda-kernels** | 67.1 | 227.9 | 2.88x | 1948 / 2130 | 1.4 s / 29 ms | 0.513 | 8.572 |

Cas-GwcNet-gc (`--model gwcnet`):

| version | KITTI ms | Midd-H ms | speed-up at H | peak MB at H (alloc / reserved) | startup / new size | ETH3D EPE | Midd-H EPE |
|---|---|---|---|---|---|---|---|
| upstream | 250.7 | 861.4 | 1.00x | 10101 / 10792 | 1.3 s / 0 ms | 0.821 | 6.541 |
| **portable** | 158.1 | 561.0 | 1.54x | 2405 / 2836 | 1.3 s / 43 ms | 0.821 | 6.542 |
| **cuda-kernels** | 113.2 | 377.1 | 2.28x | 2284 / 2452 | 1.4 s / 16 ms | 0.824 | 6.541 |

```python
# cwd CasStereoNet/; model = __models__["gwcnet-c" | "gwcnet"](...) with the checkpoint loaded, .cuda().eval()
from models.gwcnet import prepare_half
prepare_half(model)                      # portable: fp16 autocast, fp32 cost volume
pred = model(left, right)["stage2"]["pred"]  # ImageNet-normalised images padded to /32, as upstream
```

---

# Casacde-Stereo(CVPR2020 Oral)
Official source code of paper Cascade Cost Volume for High-Resolution Multi-View Stereo and Stereo Matching, [arxiv](https://arxiv.org/pdf/1912.06378.pdf)

# Installation
## Requirements
* python 3.6
* Pytorch == 1.2 （The default interpolation of the high version is different from the low version, which will cause the results to be different）
* CUDA >= 9.0

```
pip install -r requirements.txt  --user
```

## Option: Apex 
install apex to enable synchronized batch normalization 
```
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

# CasMVSNet(Multi View Stereo)
Refer to [MVS-README.md](CasMVSNet/README.md)
# CasStereoNet(Stereo Matching)
Refer to [Stereo-Matching-README.md](CasStereoNet/README.md)

# Citation
If you find this code useful in your research, please cite:

```
@inproceedings{gu2020cascade,
  title={Cascade cost volume for high-resolution multi-view stereo and stereo matching},
  author={Gu, Xiaodong and Fan, Zhiwen and Zhu, Siyu and Dai, Zuozhuo and Tan, Feitong and Tan, Ping},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2495--2504},
  year={2020}
}
```

# Acknowledgements
Thanks to Xiaoyang Guo for opening source of his excellent works [GwcNet](https://github.com/xy-guo/GwcNet)
and [MVSNet-pytorch](https://github.com/xy-guo/MVSNet_pytorch). Thanks to Yao Yao for opening source of 
his excellent work [MVSNet](https://github.com/YoYo000/MVSNet).
