

import torch
import random
import numpy as np
import torchaudio
import torch.nn.functional as F



'''
tensor: the tensor of the video, 
noise_strength: the strength of the noise,
brightness: the brightness of the video,
contrast: the contrast of the video,
'''

class video_augmentation():
    def __init__(
        self,
        p_noise=0.4, noise_strength=(0.01, 0.08),
        p_bc=0.4, brightness=(0.8, 1.2), contrast=(0.7, 1.3), gamma=(0.8, 1.2),
        p_blur=0.35, blur_ks=(3, 5),  # 高斯/盒滤简化
        p_downup=0.5, min_scale=0.4,  # 随机降采样再上采样(模拟压缩/低分辨)
        p_cutout=0.3, cutout_frac=(0.1, 0.25),
        p_temporal=0.3, drop_frac=(0.05, 0.15),  # 随机丢帧并重复补齐
        seed=None
    ):
        self.p_noise = p_noise; self.noise_strength = noise_strength
        self.p_bc = p_bc; self.brightness = brightness; self.contrast = contrast; self.gamma = gamma
        self.p_blur = p_blur; self.blur_ks = blur_ks
        self.p_downup = p_downup; self.min_scale = min_scale
        self.p_cutout = p_cutout; self.cutout_frac = cutout_frac
        self.p_temporal = p_temporal; self.drop_frac = drop_frac
        if seed is not None:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    def rand(self, a, b):
        return a + (b - a) * random.random()

    def ensure_5d(self, x: torch.Tensor):

        squeezed = False

        if x.dim() == 3:  # [C,H,W]
            x = x.unsqueeze(1)  #  [C,1,H,W]
            squeezed = True
        if x.dim() == 4:  # [C,T,H,W]
            x = x.unsqueeze(0)  #  [1,C,T,H,W]
            squeezed = True
        elif x.dim() != 5:
            raise ValueError(f"Expected [C,T,H,W] or [B,C,T,H,W] or [C,H,W], got {x.shape}")

        return x, squeezed

    def add_noise(self, x):
        if random.random() < self.p_noise:
            s = self.rand(*self.noise_strength)
            x = torch.clamp(x + torch.randn_like(x) * s, 0, 1)
        return x

    def bc_gamma(self, x):
        if random.random() < self.p_bc:

            c = self.rand(*self.contrast)
            x = torch.clamp(x * c, 0, 1)

            b = self.rand(*self.brightness)
            x = torch.clamp(x + (b - 1.0), 0, 1)

            # gamma
            g = self.rand(*self.gamma)
            x = torch.clamp(x ** g, 0, 1)
        return x

    def blur(self, x):
        if random.random() < self.p_blur:
            k = random.choice(self.blur_ks)
            kernel = torch.ones((1, 1, k, k), device=x.device, dtype=x.dtype) / (k * k)
            B, C, T, H, W = x.shape
            x = x.contiguous()
            x2 = x.view(B*C*T, 1, H, W)
            x2 = F.conv2d(F.pad(x2, (k//2, k//2, k//2, k//2), mode='reflect'), kernel)
            x = x2.view(B, C, T, H, W)
        return x

    def down_up(self, x):
        if random.random() < self.p_downup:
            B, C, T, H, W = x.shape
            scale = self.rand(self.min_scale, 1.0)
            h2 = max(8, int(H * scale))
            w2 = max(8, int(W * scale))
            x = F.interpolate(x, size=(T, h2, w2), mode='trilinear', align_corners=False)
            x = F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=False)
        return x

    def cutout(self, x):
        if random.random() < self.p_cutout:
            B, C, T, H, W = x.shape
            frac = self.rand(*self.cutout_frac)
            ch, cw = int(H * frac), int(W * frac)
            cy = random.randint(0, max(0, H - ch))
            cx = random.randint(0, max(0, W - cw))
            mask = torch.ones((B, 1, 1, H, W), device=x.device, dtype=x.dtype)
            mask[:, :, :, cy:cy+ch, cx:cx+cw] = 0.0
            x = x * mask
        return x

    def temporal_droprepeat(self, x):
        if random.random() < self.p_temporal:
            B, C, T, H, W = x.shape
            drop_ratio = self.rand(*self.drop_frac)
            keep = int(T * (1 - drop_ratio))
            idx = sorted(random.sample(range(T), keep))
            kept = x[:, :, idx, :, :]
            rep = T - keep

            if rep > 0:
                rep_idx = np.random.choice(np.arange(kept.shape[2]), size=rep, replace=True)
                rep_frames = kept[:, :, rep_idx, :, :]
                x = torch.cat([kept, rep_frames], dim=2)
                # 打乱回到时间顺序
                perm = torch.randperm(T, device=x.device)
                x = x[:, :, perm, :, :]
            else:
                x = kept
        return x

    def __call__(self, video):
        if isinstance(video, np.ndarray):
            video = torch.from_numpy(video).float()
        x, squeezed = self.ensure_5d(video)

        x = self.down_up(x)
        x = self.blur(x)
        x = self.add_noise(x)
        x = self.bc_gamma(x)
        x = self.cutout(x)
        x = self.temporal_droprepeat(x)

        return x.squeeze(0) if squeezed else x





