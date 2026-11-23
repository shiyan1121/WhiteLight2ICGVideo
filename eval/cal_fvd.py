from __future__ import annotations

import numpy as np
import torch


def _tensor_to_uint8(frame: torch.Tensor) -> np.ndarray:
    array = frame.detach().cpu().clamp(0.0, 1.0).numpy()
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    array = (array * 255.0).round().astype(np.uint8)
    return array


def _gray_uint8_to_rgb(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return np.repeat(frame[:, :, None], 3, axis=2)
    if frame.ndim == 3 and frame.shape[2] == 1:
        return np.repeat(frame, 3, axis=2)
    return frame


def frames_to_uint8_videos(pred_frames: list[torch.Tensor], target_frames: list[torch.Tensor]) -> tuple[np.ndarray, np.ndarray]:
    """Return real/fake videos as uint8 arrays shaped [B, T, H, W, C]."""
    if not pred_frames:
        raise ValueError("FVD requires paired prediction and target frames.")
    if len(pred_frames) != len(target_frames):
        raise ValueError("pred_frames and target_frames must have the same length.")

    fake_frames = []
    real_frames = []
    for pred, target in zip(pred_frames, target_frames):
        fake_frames.append(_gray_uint8_to_rgb(_tensor_to_uint8(pred)))
        real_frames.append(_gray_uint8_to_rgb(_tensor_to_uint8(target)))
    return (
        np.stack(real_frames, axis=0)[None].astype(np.uint8),
        np.stack(fake_frames, axis=0)[None].astype(np.uint8),
    )


def build_fvd_metric(device: torch.device, model_name: str = "i3d", ckpt_path: str | None = None):
    from cdfvd import fvd  # type: ignore

    return fvd.cdfvd(
        model=model_name,
        n_real="full",
        n_fake="full",
        ckpt_path=ckpt_path,
        device=str(device),
    )


def calculate_fvd(real_videos: np.ndarray, fake_videos: np.ndarray, metric) -> float:
    """Calculate cd-fvd for uint8 videos shaped [B, T, H, W, C]."""
    metric.empty_real_stats()
    metric.empty_fake_stats()
    return float(metric.compute_fvd(real_videos, fake_videos))


def calculate_fvd_frames(pred_frames: list[torch.Tensor], target_frames: list[torch.Tensor], metric) -> float:
    real_videos, fake_videos = frames_to_uint8_videos(pred_frames, target_frames)
    return calculate_fvd(real_videos, fake_videos, metric)
