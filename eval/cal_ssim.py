from __future__ import annotations

import cv2
import numpy as np
import torch
from tqdm import tqdm


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim_map.mean())


def calculate_ssim_function(img1: np.ndarray, img2: np.ndarray) -> float:
    """Open-Sora/Fundus2Video SSIM for images in [0, 1]."""
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")
    if img1.ndim == 2:
        return ssim(img1, img2)
    if img1.ndim == 3:
        if img1.shape[0] == 3:
            return float(np.array([ssim(img1[i], img2[i]) for i in range(3)]).mean())
        if img1.shape[0] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    raise ValueError("Wrong input image dimensions.")


def calculate_ssim(videos1: torch.Tensor, videos2: torch.Tensor) -> dict:
    """Calculate per-frame SSIM for videos shaped [B, T, C, H, W]."""
    print("calculate_ssim...")
    if videos1.shape != videos2.shape:
        raise ValueError("videos1 and videos2 must have the same shape")

    ssim_results = []
    for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        ssim_results_of_a_video = []
        for frame_idx in range(len(video1)):
            ssim_results_of_a_video.append(calculate_ssim_function(video1[frame_idx].numpy(), video2[frame_idx].numpy()))
        ssim_results.append(ssim_results_of_a_video)

    ssim_results = np.array(ssim_results)
    ssim_values = {}
    ssim_std = {}
    for frame_idx in range(videos1.shape[1]):
        ssim_values[frame_idx] = float(np.mean(ssim_results[:, frame_idx]))
        ssim_std[frame_idx] = float(np.std(ssim_results[:, frame_idx]))

    return {
        "value": ssim_values,
        "value_std": ssim_std,
        "video_setting": videos1[0].shape,
        "video_setting_name": "time, channel, height, width",
    }
