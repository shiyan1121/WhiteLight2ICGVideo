from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm


def trans(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-3] == 1:
        x = x.repeat(1, 1, 3, 1, 1)
    return x * 2.0 - 1.0


def build_lpips_model(device: torch.device) -> torch.nn.Module:
    import lpips  # type: ignore

    loss_fn = lpips.LPIPS(net="alex", spatial=True).to(device)
    loss_fn.eval()
    return loss_fn


def calculate_lpips(videos1: torch.Tensor, videos2: torch.Tensor, device: torch.device, loss_fn: torch.nn.Module | None = None) -> dict:
    """Calculate Open-Sora/Fundus2Video LPIPS for videos shaped [B, T, C, H, W]."""
    print("calculate_lpips...")
    if videos1.shape != videos2.shape:
        raise ValueError("videos1 and videos2 must have the same shape")

    videos1 = trans(videos1)
    videos2 = trans(videos2)
    if loss_fn is None:
        loss_fn = build_lpips_model(device)

    lpips_results = []
    for video_num in tqdm(range(videos1.shape[0])):
        video1 = videos1[video_num]
        video2 = videos2[video_num]
        lpips_results_of_a_video = []
        for frame_idx in range(len(video1)):
            img1 = video1[frame_idx].unsqueeze(0).to(device)
            img2 = video2[frame_idx].unsqueeze(0).to(device)
            with torch.no_grad():
                lpips_results_of_a_video.append(float(loss_fn.forward(img1, img2).mean().detach().cpu()))
        lpips_results.append(lpips_results_of_a_video)

    lpips_results = np.array(lpips_results)
    lpips_values = {}
    lpips_std = {}
    for frame_idx in range(videos1.shape[1]):
        lpips_values[frame_idx] = float(np.mean(lpips_results[:, frame_idx]))
        lpips_std[frame_idx] = float(np.std(lpips_results[:, frame_idx]))

    return {
        "value": lpips_values,
        "value_std": lpips_std,
        "video_setting": videos1[0].shape,
        "video_setting_name": "time, channel, height, width",
    }


def frames_to_video_tensor(frames: list[torch.Tensor]) -> torch.Tensor:
    if not frames:
        raise ValueError("LPIPS requires paired prediction and target frames.")
    return torch.stack([frame.detach().cpu().clamp(0.0, 1.0) for frame in frames], dim=0).unsqueeze(0)


def calculate_lpips_frames(
    pred_frames: list[torch.Tensor],
    target_frames: list[torch.Tensor],
    device: torch.device,
    loss_fn: torch.nn.Module,
) -> float:
    result = calculate_lpips(frames_to_video_tensor(pred_frames), frames_to_video_tensor(target_frames), device, loss_fn=loss_fn)
    values = list(result["value"].values())
    return float(sum(values) / len(values))
