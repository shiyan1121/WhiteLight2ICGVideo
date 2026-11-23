"""Reconstruction, temporal and auxiliary vessel losses for ConvGRU-ResUNet."""
from __future__ import annotations
import torch
import torch.nn.functional as F

def ssim_index(pred: torch.Tensor, target: torch.Tensor, window_size: int=11) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    pad = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=pad)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=pad)
    sigma_x = F.avg_pool2d(pred * pred, window_size, stride=1, padding=pad) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, window_size, stride=1, padding=pad) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(pred * target, window_size, stride=1, padding=pad) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    return (num / den.clamp_min(1e-06)).mean()

def multi_scale_ssim_loss(pred: torch.Tensor, target: torch.Tensor, weights: tuple[float, ...]=(0.5, 0.3, 0.2), window_size: int=11) -> torch.Tensor:
    losses = []
    x = pred
    y = target
    for scale_weight in weights:
        losses.append(float(scale_weight) * (1.0 - ssim_index(x, y, window_size=window_size)))
        if min(x.shape[-2:]) <= 16:
            break
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        y = F.avg_pool2d(y, kernel_size=2, stride=2)
    return torch.stack(losses).sum() / max(sum((float(w) for w in weights[:len(losses)])), 1e-06)

def image_gradient_l1(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_dx = pred[..., :, 1:] - pred[..., :, :-1]
    target_dx = target[..., :, 1:] - target[..., :, :-1]
    pred_dy = pred[..., 1:, :] - pred[..., :-1, :]
    target_dy = target[..., 1:, :] - target[..., :-1, :]
    return 0.5 * (torch.abs(pred_dx - target_dx).mean() + torch.abs(pred_dy - target_dy).mean())

def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, vessel_mask: torch.Tensor, l1_weight: float=1.0, ssim_weight: float=0.2, vessel_weight: float=2.0, ms_ssim_weight: float=0.0, vessel_ssim_weight: float=0.0, vessel_ms_ssim_weight: float=0.0, gradient_weight: float=0.0, vessel_gradient_weight: float=0.0) -> tuple[torch.Tensor, dict[str, float]]:
    abs_err = torch.abs(pred - target)
    l1 = abs_err.mean()
    masked_l1 = (abs_err * vessel_mask).sum() / vessel_mask.sum().clamp_min(1.0)
    ssim_loss = 1.0 - ssim_index(pred, target)
    vessel_pred = pred * vessel_mask
    vessel_target = target * vessel_mask
    ms_ssim = multi_scale_ssim_loss(pred, target) if ms_ssim_weight > 0.0 else pred.new_tensor(0.0)
    vessel_ssim = 1.0 - ssim_index(vessel_pred, vessel_target) if vessel_ssim_weight > 0.0 else pred.new_tensor(0.0)
    vessel_ms_ssim = multi_scale_ssim_loss(vessel_pred, vessel_target) if vessel_ms_ssim_weight > 0.0 else pred.new_tensor(0.0)
    gradient = image_gradient_l1(pred, target) if gradient_weight > 0.0 else pred.new_tensor(0.0)
    vessel_gradient = image_gradient_l1(vessel_pred, vessel_target) if vessel_gradient_weight > 0.0 else pred.new_tensor(0.0)
    total = l1_weight * l1 + ssim_weight * ssim_loss + vessel_weight * masked_l1 + ms_ssim_weight * ms_ssim + vessel_ssim_weight * vessel_ssim + vessel_ms_ssim_weight * vessel_ms_ssim + gradient_weight * gradient + vessel_gradient_weight * vessel_gradient
    parts = {'loss': float(total.detach().cpu()), 'l1': float(l1.detach().cpu()), 'masked_l1': float(masked_l1.detach().cpu()), 'ssim_loss': float(ssim_loss.detach().cpu()), 'ms_ssim_loss': float(ms_ssim.detach().cpu()), 'vessel_ssim_loss': float(vessel_ssim.detach().cpu()), 'vessel_ms_ssim_loss': float(vessel_ms_ssim.detach().cpu()), 'gradient_loss': float(gradient.detach().cpu()), 'vessel_gradient_loss': float(vessel_gradient.detach().cpu())}
    return (total, parts)

def aux_keyframe_vessel_loss(pred_frame: torch.Tensor, target_frame: torch.Tensor, pred_mask: torch.Tensor, target_mask: torch.Tensor, available: torch.Tensor | None=None, frame_weight: float=2.0, mask_weight: float=0.5, bce_weight: float=1.0, dice_weight: float=1.0, focal_gamma: float=2.0, focal_alpha: float | None=0.75, frame_time: torch.Tensor | None=None, frame_time_weighting_enabled: bool=False, early_time_threshold: float=0.35, early_frame_l1_weight: float=2.0, late_frame_l1_weight: float=1.0, pred_mask_frame_weight: float=0.0, target_threshold: float=0.5, smooth: float=1.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Auxiliary key-frame vessel supervision for a two-head generator.

    The frame branch is supervised inside the GT key-frame mask. The mask branch
    predicts the key-frame vessel mask with focal BCE + Dice. The optional
    pred_mask_frame term is kept off by default to avoid mask-shrinking shortcuts.
    """
    if target_mask.shape[-2:] != pred_frame.shape[-2:]:
        target_mask = F.interpolate(target_mask.float(), size=pred_frame.shape[-2:], mode='nearest')
    if target_mask.shape[1] != 1:
        target_mask = target_mask.mean(dim=1, keepdim=True)
    target_mask = (target_mask.float() >= float(target_threshold)).to(dtype=pred_frame.dtype)
    pred_mask = pred_mask.float()
    if pred_mask.shape[-2:] != target_mask.shape[-2:]:
        pred_mask = F.interpolate(pred_mask, size=target_mask.shape[-2:], mode='bilinear', align_corners=False)
    if available is None:
        valid = torch.ones(pred_frame.shape[0], 1, 1, 1, dtype=pred_frame.dtype, device=pred_frame.device)
    else:
        valid = available.view(available.shape[0], 1, 1, 1).to(dtype=pred_frame.dtype, device=pred_frame.device)
    valid_sum = valid.sum().clamp_min(1.0)
    bsz = pred_frame.shape[0]
    valid_flat = valid.reshape(bsz).float()
    valid_sum_flat = valid_flat.sum().clamp_min(1.0)
    sample_weights = torch.ones(bsz, dtype=pred_frame.dtype, device=pred_frame.device)
    frame_time_weight_mean = pred_frame.new_tensor(1.0)
    frame_time_early_fraction = pred_frame.new_tensor(0.0)
    if frame_time_weighting_enabled and frame_time is not None:
        time_flat = frame_time.reshape(bsz).to(dtype=pred_frame.dtype, device=pred_frame.device)
        early = time_flat <= float(early_time_threshold)
        sample_weights = torch.where(early, torch.full_like(time_flat, float(early_frame_l1_weight)), torch.full_like(time_flat, float(late_frame_l1_weight)))
        frame_time_weight_mean = (sample_weights.float() * valid_flat).sum() / valid_sum_flat
        frame_time_early_fraction = (early.float() * valid_flat).sum() / valid_sum_flat
    mask_for_frame = target_mask * valid
    frame_denom = mask_for_frame.sum().clamp_min(1.0)
    abs_frame_err = torch.abs(pred_frame - target_frame)
    frame_l1 = (abs_frame_err * mask_for_frame).sum() / frame_denom
    per_sample_num = (abs_frame_err * mask_for_frame).reshape(bsz, -1).sum(dim=1)
    per_sample_den = mask_for_frame.reshape(bsz, -1).sum(dim=1).clamp_min(1.0)
    per_sample_frame_l1 = per_sample_num / per_sample_den
    weighted_frame_l1 = (per_sample_frame_l1 * sample_weights.float() * valid_flat).sum() / valid_sum_flat
    pred_mask_f = pred_mask.float().clamp(1e-06, 1.0 - 1e-06)
    target_mask_f = target_mask.float()
    valid_f = valid.float()
    with torch.cuda.amp.autocast(enabled=False):
        bce_map = F.binary_cross_entropy(pred_mask_f, target_mask_f, reduction='none')
        pt = torch.where(target_mask_f > 0.5, pred_mask_f, 1.0 - pred_mask_f)
        focal_map = bce_map * (1.0 - pt).pow(float(focal_gamma))
        if focal_alpha is not None:
            alpha = float(focal_alpha)
            alpha_t = torch.where(target_mask_f > 0.5, torch.full_like(target_mask_f, alpha), torch.full_like(target_mask_f, 1.0 - alpha))
            focal_map = focal_map * alpha_t
    spatial_size = bce_map.shape[-1] * bce_map.shape[-2]
    per_sample_plain_bce = bce_map.reshape(bsz, -1).mean(dim=1)
    per_sample_mask_bce = focal_map.reshape(bsz, -1).mean(dim=1)
    plain_bce_raw = (per_sample_plain_bce * valid_flat).sum() / valid_sum_flat
    mask_bce_raw = (per_sample_mask_bce * valid_flat).sum() / valid_sum_flat
    plain_bce = (per_sample_plain_bce * sample_weights.float() * valid_flat).sum() / valid_sum_flat
    mask_bce = (per_sample_mask_bce * sample_weights.float() * valid_flat).sum() / valid_sum_flat
    dims = tuple(range(1, pred_mask.ndim))
    intersection = (pred_mask_f * target_mask_f * valid_f).sum(dims)
    denom = (pred_mask_f * valid_f).sum(dims) + (target_mask_f * valid_f).sum(dims)
    dice_per_sample = 1.0 - (2.0 * intersection + smooth) / (denom + smooth).clamp_min(1e-06)
    dice_raw = (dice_per_sample * valid_flat).sum() / valid_sum_flat
    dice = (dice_per_sample * sample_weights.float() * valid_flat).sum() / valid_sum_flat
    mask_loss = float(bce_weight) * mask_bce + float(dice_weight) * dice
    pred_mask_frame_l1 = pred_frame.new_tensor(0.0)
    pred_mask_frame_l1_weighted = pred_frame.new_tensor(0.0)
    if pred_mask_frame_weight > 0.0:
        pred_mask_detached = pred_mask.detach()
        pred_mask_frame_abs = torch.abs(pred_frame * pred_mask_detached - target_frame * target_mask).mul(valid)
        pred_mask_frame_l1 = pred_mask_frame_abs.sum() / (valid_sum_flat * spatial_size)
        per_sample_pred_mask_frame_l1 = pred_mask_frame_abs.reshape(bsz, -1).mean(dim=1)
        pred_mask_frame_l1_weighted = (per_sample_pred_mask_frame_l1 * sample_weights.float() * valid_flat).sum() / valid_sum_flat
    total = float(frame_weight) * weighted_frame_l1 + float(mask_weight) * mask_loss + float(pred_mask_frame_weight) * pred_mask_frame_l1_weighted
    return (total, {'aux_vessel_loss': float(total.detach().cpu()), 'aux_vessel_frame_l1': float(frame_l1.detach().cpu()), 'aux_vessel_frame_l1_weighted': float(weighted_frame_l1.detach().cpu()), 'aux_vessel_frame_time_weight_mean': float(frame_time_weight_mean.detach().cpu()), 'aux_vessel_frame_early_fraction': float(frame_time_early_fraction.detach().cpu()), 'aux_vessel_mask_bce': float(mask_bce.detach().cpu()), 'aux_vessel_mask_bce_raw': float(mask_bce_raw.detach().cpu()), 'aux_vessel_mask_plain_bce': float(plain_bce.detach().cpu()), 'aux_vessel_mask_plain_bce_raw': float(plain_bce_raw.detach().cpu()), 'aux_vessel_mask_dice': float(dice.detach().cpu()), 'aux_vessel_mask_dice_raw': float(dice_raw.detach().cpu()), 'aux_vessel_pred_mask_frame_l1': float(pred_mask_frame_l1.detach().cpu()), 'aux_vessel_pred_mask_frame_l1_weighted': float(pred_mask_frame_l1_weighted.detach().cpu()), 'aux_vessel_pred_mean': float(pred_mask.detach().mean().cpu()), 'aux_vessel_target_mean': float(target_mask.detach().mean().cpu()), 'aux_vessel_available': float(valid.detach().mean().cpu())})

def sequence_vessel_curve_loss(pred_seq: torch.Tensor, target_seq: torch.Tensor, vessel_mask: torch.Tensor, value_weight: float=1.0, delta_weight: float=1.0, corr_weight: float=0.0, eps: float=1e-06) -> tuple[torch.Tensor, dict[str, float]]:
    """Match vessel-region intensity curves over a full patch sequence.

    pred_seq and target_seq are [T, 1, H, W]. vessel_mask is [1, H, W] or
    [T, 1, H, W]. The loss is computed on the vessel-mean intensity curve.
    """
    if pred_seq.ndim != 4 or target_seq.ndim != 4:
        raise ValueError('pred_seq and target_seq must have shape [T, C, H, W].')
    if pred_seq.shape != target_seq.shape:
        raise ValueError(f'pred_seq and target_seq shape mismatch: {pred_seq.shape} vs {target_seq.shape}')
    mask = vessel_mask.float()
    if mask.ndim == 3:
        mask = mask.unsqueeze(0).expand(pred_seq.shape[0], -1, -1, -1)
    elif mask.ndim == 4 and mask.shape[0] == 1:
        mask = mask.expand(pred_seq.shape[0], -1, -1, -1)
    if mask.shape[-2:] != pred_seq.shape[-2:]:
        mask = F.interpolate(mask, size=pred_seq.shape[-2:], mode='nearest')
    if mask.shape[1] != pred_seq.shape[1]:
        mask = mask.mean(dim=1, keepdim=True).expand(-1, pred_seq.shape[1], -1, -1)
    denom = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    pred_curve = (pred_seq * mask).sum(dim=(1, 2, 3)) / denom
    target_curve = (target_seq * mask).sum(dim=(1, 2, 3)) / denom
    value_loss = torch.abs(pred_curve - target_curve).mean()
    if pred_curve.numel() > 1:
        pred_delta = pred_curve[1:] - pred_curve[:-1]
        target_delta = target_curve[1:] - target_curve[:-1]
        delta_loss = torch.abs(pred_delta - target_delta).mean()
    else:
        delta_loss = pred_seq.new_tensor(0.0)
    corr_loss = pred_seq.new_tensor(0.0)
    if corr_weight > 0.0 and pred_curve.numel() > 1:
        pred_centered = pred_curve - pred_curve.mean()
        target_centered = target_curve - target_curve.mean()
        corr = (pred_centered * target_centered).sum() / (pred_centered.square().sum().sqrt() * target_centered.square().sum().sqrt()).clamp_min(eps)
        corr_loss = 1.0 - corr.clamp(-1.0, 1.0)
    total = float(value_weight) * value_loss + float(delta_weight) * delta_loss + float(corr_weight) * corr_loss
    return (total, {'sequence_curve_value_loss': float(value_loss.detach().cpu()), 'sequence_curve_delta_loss': float(delta_loss.detach().cpu()), 'sequence_curve_corr_loss': float(corr_loss.detach().cpu())})

def temporal_difference_loss(pred: torch.Tensor, pred_next: torch.Tensor, target: torch.Tensor, target_next: torch.Tensor, vessel_mask: torch.Tensor | None=None, valid: torch.Tensor | None=None, vessel_weight: float=0.0, curve_weight: float=0.0) -> tuple[torch.Tensor, dict[str, float]]:
    """Match predicted and target frame-to-frame changes for adjacent frames."""
    delta_pred = pred_next - pred
    delta_target = target_next - target
    abs_err = torch.abs(delta_pred - delta_target)
    if valid is not None:
        valid_map = valid.view(valid.shape[0], 1, 1, 1).to(abs_err.dtype)
        abs_err = abs_err * valid_map
        denom = valid_map.sum().clamp_min(1.0) * abs_err.shape[-1] * abs_err.shape[-2]
        temporal_l1 = abs_err.sum() / denom
    else:
        temporal_l1 = abs_err.mean()
    total = temporal_l1
    masked_temporal_l1 = pred.new_tensor(0.0)
    if vessel_mask is not None and vessel_weight > 0.0:
        mask = vessel_mask.float()
        if valid is not None:
            mask = mask * valid.view(valid.shape[0], 1, 1, 1).to(mask.dtype)
        masked_temporal_l1 = (abs_err * mask).sum() / mask.sum().clamp_min(1.0)
        total = total + float(vessel_weight) * masked_temporal_l1
    curve_loss = pred.new_tensor(0.0)
    if vessel_mask is not None and curve_weight > 0.0:
        mask = vessel_mask.float()
        if valid is not None:
            mask = mask * valid.view(valid.shape[0], 1, 1, 1).to(mask.dtype)
        pred_curve = (delta_pred * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        target_curve = (delta_target * mask).sum(dim=(1, 2, 3)) / mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
        if valid is not None:
            valid_flat = valid.flatten().to(pred_curve.dtype)
            curve_loss = (torch.abs(pred_curve - target_curve) * valid_flat).sum() / valid_flat.sum().clamp_min(1.0)
        else:
            curve_loss = torch.abs(pred_curve - target_curve).mean()
        total = total + float(curve_weight) * curve_loss
    return (total, {'temporal_l1': float(temporal_l1.detach().cpu()), 'temporal_masked_l1': float(masked_temporal_l1.detach().cpu()), 'temporal_curve_loss': float(curve_loss.detach().cpu())})
