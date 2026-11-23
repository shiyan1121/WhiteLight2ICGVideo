from __future__ import annotations

import math

import numpy as np
import torch
from tqdm import tqdm


def img_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Open-Sora/Fundus2Video PSNR for images in [0, 1]."""
    mse = np.mean((img1 / 1.0 - img2 / 1.0) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(float(mse)))


def calculate_psnr(videos1: torch.Tensor, videos2: torch.Tensor) -> dict:
    """Calculate per-frame PSNR for videos shaped [B, T, C, H, W]."""
    print("calculate_psnr...")
    if videos1.shape != videos2.shape:
        raise ValueError("videos1 and videos2 must have the same shape")

    psnr_results = []
    for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        psnr_results_of_a_video = []
        for frame_idx in range(len(video1)):
            psnr_results_of_a_video.append(img_psnr(video1[frame_idx].numpy(), video2[frame_idx].numpy()))
        psnr_results.append(psnr_results_of_a_video)

    psnr_results = np.array(psnr_results)
    psnr = {}
    psnr_std = {}
    for frame_idx in range(videos1.shape[1]):
        psnr[frame_idx] = float(np.mean(psnr_results[:, frame_idx]))
        psnr_std[frame_idx] = float(np.std(psnr_results[:, frame_idx]))

    return {
        "value": psnr,
        "value_std": psnr_std,
        "video_setting": videos1[0].shape,
        "video_setting_name": "time, channel, height, width",
    }
