from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from eval import build_fvd_metric, build_lpips_model, calculate_fvd_frames, calculate_lpips_frames, calculate_ssim_function, img_psnr
from wlicgv.data import (
    PATCH_RE,
    _find_lsci_input,
    _find_rgb_input,
    _find_mtt_map,
    _find_rgb2vessel_input,
    _find_vessel_mask,
    _list_target_frames,
    _load_gray,
    _load_rgb,
    _normalized_times,
)
from wlicgv.models import build_model
from wlicgv.time_templates import configure_time_template, validate_time_template
from wlicgv.utils import ensure_dir, save_preview_panel, subject_from_patch_dir, tensor_to_uint8


def split_generator_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if output.shape[1] == 1:
        return output, None
    return output[:, :1], output[:, 1:2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run autoregressive WhiteLight-to-ICG inference.")
    parser.add_argument("--experiment_name", default=None, help="Experiment directory name under --experiments_root.")
    parser.add_argument(
        "--experiments_root",
        default="E:/ClinicalICG/WhiteLight2ICGVideo/code/4-WhiteLight2ICGVideo-ConvGRU/experiments",
        help="Root directory containing experiment folders.",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to checkpoints/best.pt or latest.pt.")
    parser.add_argument("--checkpoint_type", default="best", choices=["best", "last", "latest"], help="Checkpoint to use inside an experiment.")
    parser.add_argument("--config", default=None, help="Optional config.yaml override. Defaults to the experiment config.")
    parser.add_argument("--time_template_file", default=None, help="Override model input times with a frozen mean vector from a normalized-time summary JSON.")
    parser.add_argument("--time_template_split", choices=["train", "all"], default=None, help="Mean-time split (default train). All is exploratory, using held-out statistics.")
    parser.add_argument("--use_saved_time_template", action="store_true", help="Require the frozen mean-time vector saved in this experiment; do not reload external statistics.")
    parser.add_argument("--patch_dir", default=None, help="Run one patch directory, e.g. S14-260203_01.")
    parser.add_argument("--rgb_path", default=None, help="Override the static RGB input for a single --patch_dir; vessel mask and inference times remain unchanged.")
    parser.add_argument("--data_root", default=None, help="Run all patch directories under this root.")
    parser.add_argument("--out_dir", default=None, help="Output directory. Defaults to <patch>/predicted_icg_sequence.")
    parser.add_argument("--subjects", nargs="*", default=None, help="Optional subject filter for --data_root.")
    parser.add_argument("--splits", nargs="*", default=["val", "test"], choices=["train", "val", "test"], help="Splits to infer when --experiment_name is used.")
    parser.add_argument("--max_patches", type=int, default=None, help="Optional limit for --data_root inference.")
    parser.add_argument("--frames", type=int, default=None, help="Override number of output frames.")
    parser.add_argument("--frame_step", type=int, default=None, help="Override data.frame_step from checkpoint config.")
    parser.add_argument("--sample_mode", default=None, help="Override data.sample_mode from checkpoint config.")
    parser.add_argument(
        "--require_targets",
        action="store_true",
        help="Fail before inference if any selected patch lacks the requested GT key-frame sequence.",
    )
    parser.add_argument(
        "--input_vessel_mask_source",
        choices=["global", "rgb2vessel_mask", "rgb2vessel_prob"],
        default=None,
        help="Override data.input_vessel_mask_source for inference.",
    )
    parser.add_argument(
        "--skip_missing_input_vessel_mask",
        action="store_true",
        help="Skip patches that do not contain the selected input vessel mask instead of failing.",
    )
    parser.add_argument(
        "--merge_bypass_vessel_mask",
        action="store_true",
        help="Merge the selected vessel mask with the patch bypass vessel mask as one input channel.",
    )
    parser.add_argument(
        "--bypass_dilate_pixels",
        type=int,
        default=4,
        help="Dilation radius applied to the bypass vessel mask before merging.",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save_targets", action="store_true", help="Save resized target frames when available.")
    parser.add_argument("--save_panels", action="store_true", help="Save RGB/Vessel/Pred/Target/AbsErr panels when targets are available.")
    parser.add_argument("--copy_config", action="store_true", help="Copy checkpoint config into each output directory.")
    parser.add_argument("--fvd_model", default="i3d", choices=["i3d", "videomae"], help="cd-fvd feature backbone.")
    parser.add_argument("--fvd_ckpt_path", default=None, help="Optional cd-fvd checkpoint/cache path.")
    parser.add_argument("--no_save_raw_mtt_npy", action="store_true", help="Do not save raw gt_mtt.npy and pred_mtt.npy.")
    parser.add_argument("--no_overwrite_raw_mtt_npy", action="store_true", help="Keep existing raw MTT .npy files when present.")
    parser.add_argument("--raw_mtt_baseline_frames", type=int, default=5, help="Number of first frames used as raw MTT baseline.")
    parser.add_argument("--raw_mtt_mask_threshold", type=float, default=0.5, help="Threshold for the vessel mask used by vessel-region MTT normalization.")
    parser.add_argument("--raw_mtt_keep_outside_mask", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def experiment_dir(args: argparse.Namespace) -> Path | None:
    if not args.experiment_name:
        return None
    exp_dir = Path(args.experiments_root) / args.experiment_name
    if not exp_dir.exists():
        raise FileNotFoundError(f"Experiment not found: {exp_dir}")
    return exp_dir


def checkpoint_path(args: argparse.Namespace, exp_dir: Path | None) -> Path:
    if args.checkpoint:
        return Path(args.checkpoint)
    if exp_dir is None:
        raise ValueError("Provide either --experiment_name or --checkpoint.")
    name = "latest.pt" if args.checkpoint_type in {"last", "latest"} else "best.pt"
    path = exp_dir / "checkpoints" / name
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def config_path(args: argparse.Namespace, exp_dir: Path | None) -> Path | None:
    if args.config:
        return Path(args.config)
    if exp_dir is not None:
        path = exp_dir / "config.yaml"
        if path.exists():
            return path
    return None


def load_checkpoint(path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    if "model" not in ckpt:
        raise ValueError(f"Checkpoint {path} must contain 'model'.")
    return ckpt


def choose_device(name: str) -> torch.device:
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_experiment_splits(exp_dir: Path | None) -> dict[str, list[str]]:
    if exp_dir is None:
        return {}
    split_path = exp_dir / "run_config_and_split.json"
    if not split_path.exists():
        split_path = exp_dir / "subject_split_summary.json"
    if not split_path.exists():
        return {}
    data = json.loads(split_path.read_text(encoding="utf-8"))
    if "splits" in data:
        return {key: list(value) for key, value in data["splits"].items() if isinstance(value, list)}
    splits = {}
    for key in ["train", "val", "test"]:
        value = data.get(key)
        if isinstance(value, dict) and isinstance(value.get("subjects"), dict):
            splits[key] = sorted(value["subjects"].keys())
        elif isinstance(value, list):
            splits[key] = list(value)
    return splits


def resolve_subject_filter(args: argparse.Namespace, exp_dir: Path | None) -> set[str]:
    if args.subjects:
        return set(args.subjects)
    if args.patch_dir or args.data_root:
        return set()
    splits = load_experiment_splits(exp_dir)
    subjects = set()
    for split_name in args.splits:
        subjects.update(splits.get(split_name, []))
    return subjects


def resolve_patch_dirs(args: argparse.Namespace, cfg: dict, exp_dir: Path | None) -> list[Path]:
    if args.patch_dir:
        return [Path(args.patch_dir)]

    data_root = Path(args.data_root or cfg.get("data", {}).get("root", ""))
    if not data_root.exists():
        raise FileNotFoundError("Provide --patch_dir or a valid --data_root.")

    subjects = resolve_subject_filter(args, exp_dir)
    patch_dirs = []
    for patch_dir in sorted(p for p in data_root.iterdir() if p.is_dir() and PATCH_RE.match(p.name)):
        subject = subject_from_patch_dir(patch_dir)
        if subjects and subject not in subjects and subject.split("-", 1)[0] not in subjects:
            continue
        patch_dirs.append(patch_dir)
        if args.max_patches is not None and len(patch_dirs) >= args.max_patches:
            break
    return patch_dirs


def sampled_target_paths(patch_dir: Path, cfg: dict, args: argparse.Namespace) -> tuple[list[Path], list[float]]:
    sequences = sampled_target_sequences(patch_dir, cfg, args)
    targets = [path for sequence in sequences for path in sequence["targets"]]
    times = [time for sequence in sequences for time in sequence["times"]]
    return targets, times


def normalize_inference_frame_step_offsets(frame_step: int, offsets) -> list[int]:
    if frame_step <= 1:
        return [0]
    if offsets is None:
        return [0]
    if isinstance(offsets, str):
        value = offsets.strip().lower()
        if value in {"all", "*"}:
            return list(range(frame_step))
        if value in {"", "0", "first", "default", "none"}:
            return [0]
        values = [int(part) for part in re.split(r"[,;\s]+", value) if part]
    elif isinstance(offsets, int):
        values = [int(offsets)]
    else:
        values = [int(offset) for offset in offsets]
    values = sorted(set(values))
    bad = [offset for offset in values if offset < 0 or offset >= frame_step]
    if bad:
        raise ValueError(f"inference_frame_step_offsets must be in [0, {frame_step - 1}], got {bad}.")
    return values or [0]


def sampled_target_sequences(patch_dir: Path, cfg: dict, args: argparse.Namespace) -> list[dict]:
    data_cfg = cfg.get("data", {})
    sample_mode = args.sample_mode or data_cfg.get("sample_mode", "Intensity_step_sample")
    frame_step = max(1, int(args.frame_step or data_cfg.get("frame_step", data_cfg.get("inference_frame_step", 1)) or 1))
    frame_offsets = normalize_inference_frame_step_offsets(
        frame_step,
        data_cfg.get("frame_step_offsets", data_cfg.get("inference_frame_step_offsets") if frame_step > 1 else None),
    )
    max_frames = (
        args.frames
        if args.frames is not None
        else data_cfg.get("inference_max_frames_per_patch", data_cfg.get("max_frames_per_patch"))
    )

    all_targets = _list_target_frames(patch_dir, sample_mode)
    if not all_targets:
        n_frames = int(max_frames or 32)
        template = None
        if data_cfg.get("time_reference") == "shared_template":
            template = validate_time_template(data_cfg.get("time_template"))
            if n_frames > len(template):
                raise ValueError("Requested more inference frames than the frozen time template contains.")
        sequences = []
        for sequence_index, offset in enumerate(frame_offsets):
            frame_indices = list(range(offset, n_frames, frame_step))
            times = [template[i] if template is not None else (0.0 if n_frames <= 1 else i / float(n_frames - 1)) for i in frame_indices]
            if times:
                sequences.append(
                    {
                        "sequence_index": sequence_index,
                        "frame_step": frame_step,
                        "frame_step_offset": offset,
                        "targets": [],
                        "times": times,
                    }
                )
        return sequences

    if max_frames is not None:
        all_targets = all_targets[: int(max_frames)]
    normalized_times = _normalized_times(
        all_targets,
        sample_mode,
        data_cfg.get("time_reference", "normalized"),
        data_cfg.get("time_template"),
    )
    sequences = []
    for sequence_index, offset in enumerate(frame_offsets):
        targets = all_targets[offset::frame_step]
        times = [normalized_times[path][0] for path in targets]
        if targets:
            sequences.append(
                {
                    "sequence_index": sequence_index,
                    "frame_step": frame_step,
                    "frame_step_offset": offset,
                    "targets": targets,
                    "times": times,
                }
            )
    return sequences


def inference_sampling_config(cfg: dict, args: argparse.Namespace) -> dict:
    data_cfg = cfg.get("data", {})
    frame_step = max(1, int(args.frame_step or data_cfg.get("frame_step", data_cfg.get("inference_frame_step", 1)) or 1))
    return {
        "sample_mode": args.sample_mode or data_cfg.get("sample_mode", "Intensity_step_sample"),
        "time_reference": data_cfg.get("time_reference", "normalized"),
        "time_template": data_cfg.get("time_template"),
        "time_template_provenance": data_cfg.get("time_template_provenance"),
        "frame_step": frame_step,
        "frame_step_offsets": normalize_inference_frame_step_offsets(
            frame_step,
            data_cfg.get("frame_step_offsets", data_cfg.get("inference_frame_step_offsets") if frame_step > 1 else None),
        ),
        "max_frames": (
            args.frames
            if args.frames is not None
            else data_cfg.get("inference_max_frames_per_patch", data_cfg.get("max_frames_per_patch"))
        ),
    }


def inference_output_frame_number(target_path: Path | None, sequence_frame: int, frame_step: int, offset: int) -> int:
    if target_path is not None:
        match = re.match(r"^(\d+)", target_path.stem)
        if match:
            return int(match.group(1))
    return int(offset) + (int(sequence_frame) - 1) * int(frame_step) + 1


def default_output_root(args: argparse.Namespace, exp_dir: Path | None, checkpoint: Path) -> Path | None:
    if args.out_dir:
        return Path(args.out_dir)
    if exp_dir is not None:
        return exp_dir / "inference" / checkpoint.stem
    return None


def default_output_dir(patch_dir: Path, args: argparse.Namespace, batch_mode: bool, output_root: Path | None) -> Path:
    if output_root is not None:
        return output_root / patch_dir.name if batch_mode else output_root
    return patch_dir / "predicted_icg_sequence"


def masked_l1_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[float, float]:
    mask = mask.to(dtype=pred.dtype, device=pred.device)
    abs_err = torch.abs(pred - target) * mask
    sq_err = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return float((abs_err.sum() / denom).detach().cpu()), float((sq_err.sum() / denom).detach().cpu())


def vessel_curve_metrics(
    pred_frames: list[torch.Tensor],
    target_frames: list[torch.Tensor],
    mask: torch.Tensor,
    eps: float = 1e-8,
) -> dict[str, float]:
    if len(pred_frames) != len(target_frames) or len(pred_frames) == 0:
        return {}
    pred_seq = torch.stack([frame.float() for frame in pred_frames], dim=0)
    target_seq = torch.stack([frame.float() for frame in target_frames], dim=0)
    vessel_mask = mask.float().to(pred_seq.device)
    if vessel_mask.ndim == 2:
        vessel_mask = vessel_mask.unsqueeze(0)
    if vessel_mask.shape[-2:] != pred_seq.shape[-2:]:
        vessel_mask = torch.nn.functional.interpolate(
            vessel_mask.unsqueeze(0), size=pred_seq.shape[-2:], mode="nearest"
        )[0]
    if vessel_mask.shape[0] != pred_seq.shape[1]:
        vessel_mask = vessel_mask.mean(dim=0, keepdim=True).expand(pred_seq.shape[1], -1, -1)
    denom = vessel_mask.sum().clamp_min(1.0)
    pred_curve = (pred_seq * vessel_mask).sum(dim=(1, 2, 3)) / denom
    target_curve = (target_seq * vessel_mask).sum(dim=(1, 2, 3)) / denom
    curve_mae = torch.abs(pred_curve - target_curve).mean()
    if pred_curve.numel() > 1:
        pred_delta = pred_curve[1:] - pred_curve[:-1]
        target_delta = target_curve[1:] - target_curve[:-1]
        delta_mae = torch.abs(pred_delta - target_delta).mean()
        pred_centered = pred_curve - pred_curve.mean()
        target_centered = target_curve - target_curve.mean()
        corr = (pred_centered * target_centered).sum() / (
            pred_centered.square().sum().sqrt() * target_centered.square().sum().sqrt()
        ).clamp_min(eps)
        corr = corr.clamp(-1.0, 1.0)
    else:
        delta_mae = pred_seq.new_tensor(0.0)
        corr = pred_seq.new_tensor(float("nan"))
    return {
        "vessel_curve_mae": float(curve_mae.detach().cpu()),
        "vessel_curve_delta_mae": float(delta_mae.detach().cpu()),
        "vessel_curve_correlation": float(corr.detach().cpu()),
    }


def raw_mtt_normalized_times(
    patch_dir: Path,
    sample_mode: str,
    target_paths: list[Path],
    n_frames: int,
    time_reference: str = "normalized",
) -> list[float]:
    all_targets = _list_target_frames(patch_dir, sample_mode)
    if not all_targets:
        return [0.0 if n_frames <= 1 else i / float(n_frames - 1) for i in range(n_frames)]

    normalized_times = _normalized_times(all_targets, sample_mode, time_reference)
    times: list[float] = []
    for idx, target_path in enumerate(target_paths[:n_frames]):
        value = normalized_times.get(target_path)
        if value is None:
            value = normalized_times.get(all_targets[min(idx, len(all_targets) - 1)])
        if value is None:
            t = 0.0 if n_frames <= 1 else idx / float(n_frames - 1)
        else:
            t = float(value[0])
        times.append(t)

    if len(times) < n_frames:
        start = len(times)
        times.extend(0.0 if n_frames <= 1 else i / float(n_frames - 1) for i in range(start, n_frames))
    return times[:n_frames]
def resize_mask_like(mask: torch.Tensor, shape: tuple[int, int], threshold: float = 0.5) -> torch.Tensor:
    vessel_mask = mask.float()
    if vessel_mask.ndim == 2:
        vessel_mask = vessel_mask.unsqueeze(0)
    if vessel_mask.ndim == 3 and vessel_mask.shape[0] != 1:
        vessel_mask = vessel_mask.mean(dim=0, keepdim=True)
    if vessel_mask.shape[-2:] != shape:
        vessel_mask = torch.nn.functional.interpolate(
            vessel_mask.unsqueeze(0), size=shape, mode="nearest"
        )[0]
    return vessel_mask.gt(float(threshold))


def normalize_stack_inside_mask(stack: torch.Tensor, vessel_mask: torch.Tensor) -> torch.Tensor:
    mask = vessel_mask.bool().to(stack.device)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    if int(mask.sum().item()) == 0:
        raise ValueError("Cannot normalize MTT input stack because vessel mask is empty.")
    values = stack[:, :, mask[0]]
    min_value = values.min()
    max_value = values.max()
    if float((max_value - min_value).detach().cpu()) <= 0.0:
        return torch.zeros_like(stack)
    return ((stack - min_value) / (max_value - min_value)).clamp(0.0, 1.0)


def raw_mtt_map_from_frames(
    frames: list[torch.Tensor],
    times: list[float],
    baseline_frames: int = 5,
    eps: float = 1e-8,
) -> torch.Tensor:
    if not frames:
        raise ValueError("Cannot compute MTT map without frames.")
    seq = torch.stack([frame.float() for frame in frames], dim=0)
    t = torch.tensor(times[: seq.shape[0]], dtype=seq.dtype, device=seq.device).view(-1, 1, 1, 1)
    baseline_count = max(1, min(int(baseline_frames), int(seq.shape[0])))
    baseline = seq[:baseline_count].mean(dim=0, keepdim=True)
    signal = (seq - baseline).clamp_min(0.0)
    signal_sum = signal.sum(dim=0)
    mtt = (signal * t).sum(dim=0) / signal_sum.clamp_min(eps)
    mtt = mtt.masked_fill(signal_sum.le(float(eps)), float("nan"))
    return mtt


def normalize_mtt_in_vessel_mask(
    mtt: torch.Tensor,
    vessel_mask: torch.Tensor,
    mask_threshold: float = 0.5,
    eps: float = 1e-8,
) -> torch.Tensor:
    mask = resize_mask_like(vessel_mask, mtt.shape[-2:], threshold=mask_threshold).to(mtt.device)
    out = torch.nan_to_num(mtt.float(), nan=0.0, posinf=0.0, neginf=0.0)
    finite_in_mask = torch.isfinite(mtt) & mask
    if not bool(finite_in_mask.any()):
        return out
    values = mtt[finite_in_mask]
    vmin = values.min()
    vmax = values.max()
    scale = (vmax - vmin).clamp_min(eps)
    out[finite_in_mask] = ((mtt[finite_in_mask] - vmin) / scale).clamp(0.0, 1.0)
    return out


def raw_mtt_metrics(pred_mtt: torch.Tensor, target_mtt: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    valid = torch.isfinite(pred_mtt) & torch.isfinite(target_mtt)
    if not bool(valid.any()):
        nan = float("nan")
        return {
            "raw_mtt_l1": nan,
            "raw_mtt_mse": nan,
            "raw_mtt_psnr": nan,
            "raw_mtt_ssim": nan,
        }
    pred_values = pred_mtt[valid]
    target_values = target_mtt[valid]
    diff = pred_values - target_values
    raw_l1 = torch.abs(diff).mean()
    raw_mse = diff.square().mean()
    raw_psnr = psnr_from_mse(float(raw_mse.detach().cpu()))
    pred_norm = torch.nan_to_num(pred_mtt.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    target_norm = torch.nan_to_num(target_mtt.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    raw_ssim = calculate_ssim_function(
        pred_norm.detach().cpu().numpy(),
        target_norm.detach().cpu().numpy(),
    )
    return {
        "raw_mtt_l1": float(raw_l1.detach().cpu()),
        "raw_mtt_mse": float(raw_mse.detach().cpu()),
        "raw_mtt_psnr": float(raw_psnr),
        "raw_mtt_ssim": float(raw_ssim),
    }


def mtt_metrics(
    pred_frames: list[torch.Tensor],
    target_frames: list[torch.Tensor],
    times: list[float],
    mask: torch.Tensor | None = None,
    baseline_frames: int = 5,
    keep_outside_mask: bool = False,
    mask_threshold: float = 0.5,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    if len(pred_frames) != len(target_frames) or len(pred_frames) == 0:
        return {}, torch.empty(0), torch.empty(0)
    if mask is None:
        mask = torch.ones_like(pred_frames[0])
    vessel_mask = resize_mask_like(mask, pred_frames[0].shape[-2:], threshold=mask_threshold)
    pred_mtt = raw_mtt_map_from_frames(
        pred_frames,
        times,
        baseline_frames=baseline_frames,
    )
    target_mtt = raw_mtt_map_from_frames(
        target_frames,
        times,
        baseline_frames=baseline_frames,
    )
    metrics = raw_mtt_metrics(pred_mtt, target_mtt)
    pred_vessel_display = normalize_mtt_in_vessel_mask(pred_mtt, vessel_mask, mask_threshold=mask_threshold)
    target_vessel_display = normalize_mtt_in_vessel_mask(target_mtt, vessel_mask, mask_threshold=mask_threshold)
    diff = pred_vessel_display - target_vessel_display
    if mask is not None:
        vessel_mask = vessel_mask.float().to(pred_vessel_display.device)
        if vessel_mask.shape[0] != pred_vessel_display.shape[0]:
            vessel_mask = vessel_mask.mean(dim=0, keepdim=True).expand(pred_vessel_display.shape[0], -1, -1)
        denom = vessel_mask.sum().clamp_min(1.0)
        global_vessel_mtt_l1 = (torch.abs(diff) * vessel_mask).sum().div(denom)
        global_vessel_mtt_mse = (diff.square() * vessel_mask).sum().div(denom)
        metrics["global_vessel_mtt_l1"] = float(global_vessel_mtt_l1.detach().cpu())
        metrics["global_vessel_mtt_mse"] = float(global_vessel_mtt_mse.detach().cpu())
        metrics["global_vessel_mtt_psnr"] = psnr_from_mse(float(global_vessel_mtt_mse.detach().cpu()))
        pred_np = (pred_vessel_display * vessel_mask).detach().cpu().numpy()
        target_np = (target_vessel_display * vessel_mask).detach().cpu().numpy()
        metrics["global_vessel_mtt_ssim"] = calculate_ssim_function(pred_np, target_np)
    return metrics, pred_mtt.detach().cpu(), target_mtt.detach().cpu()


def save_mtt_map_png(mtt: torch.Tensor, path: Path) -> None:
    if mtt.numel() == 0:
        return
    Image.fromarray(tensor_to_uint8(mtt.clamp(0.0, 1.0))).save(path)


def save_raw_mtt_npy(mtt: torch.Tensor, path: Path) -> None:
    if mtt.numel() == 0:
        return
    arr = mtt.detach().cpu().numpy().astype(np.float32)
    np.save(path, arr)


def psnr_from_mse(mse: float, eps: float = 1e-8) -> float:
    return 20.0 * math.log10(1.0 / math.sqrt(max(float(mse), eps)))


def static_input_panels(
    cfg: dict,
    rgb: torch.Tensor,
    mask: torch.Tensor,
    mtt: torch.Tensor,
    lsci: torch.Tensor,
) -> list[tuple[str, torch.Tensor]]:
    model_cfg = cfg.get("model", {})
    panels: list[tuple[str, torch.Tensor]] = []
    if bool(model_cfg.get("include_rgb", True)):
        panels.append(("RGB", rgb))
    if bool(model_cfg.get("include_vessel_mask", True)):
        panels.append(("Vessel", mask))
    if bool(model_cfg.get("include_mtt", False)):
        panels.append(("MTT", mtt))
    if bool(model_cfg.get("include_lsci", False)):
        panels.append(("LSCI", lsci))
    return panels


def normalize_input_vessel_mask_source(source: str) -> str:
    source = str(source or "global").lower()
    if source in {"original", "gt"}:
        return "global"
    if source in {"rgb2vessel", "rgb2vessel_masks", "rgb_mask", "mask"}:
        return "rgb2vessel_mask"
    if source in {"rgb2vessel_probs", "rgb_prob", "prob"}:
        return "rgb2vessel_prob"
    return source


def find_bypass_vessel_mask(patch_dir: Path) -> Path | None:
    for name in (
        f"{patch_dir.name}_bypass_vessel_mask.png",
        f"{patch_dir.name}_bypass_graft_mask.png",
    ):
        path = patch_dir / name
        if path.exists():
            return path
    candidates = sorted(patch_dir.glob("*bypass*vessel*mask*.png"))
    return candidates[0] if candidates else None


def average_numeric_metrics(metric_dicts: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    keys = sorted({key for metrics in metric_dicts for key in metrics})
    for key in keys:
        values = []
        for metrics in metric_dicts:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                values.append(float(value))
        if values:
            out[key] = sum(values) / len(values)
    return out


def compute_metrics_for_sequence(
    records: list[dict],
    items: list[dict],
    patch_dir: Path,
    sample_mode: str,
    cfg: dict,
    args: argparse.Namespace,
    device: torch.device,
    lpips_model: torch.nn.Module,
    fvd_metric,
    global_mask: torch.Tensor,
    out_dir: Path,
    sequence_label: str,
) -> dict:
    metric_records = [record for record in records if "l1" in record]
    if not metric_records or not items:
        return {}
    items = sorted(items, key=lambda item: item["sort_key"])
    pred_frames = [item["pred"] for item in items]
    target_frames = [item["target"] for item in items]
    pred_global_vessel_frames = [item["global_vessel_pred"] for item in items]
    target_global_vessel_frames = [item["global_vessel_target"] for item in items]
    target_paths = [item["target_path"] for item in items]

    metrics = {
        "mean_l1": sum(record["l1"] for record in metric_records) / len(metric_records),
        "mean_mse": sum(record["mse"] for record in metric_records) / len(metric_records),
        "mean_psnr": sum(record["psnr"] for record in metric_records) / len(metric_records),
        "mean_ssim": sum(record["ssim"] for record in metric_records) / len(metric_records),
        "global_vessel_mean_l1": sum(record["global_vessel_l1"] for record in metric_records) / len(metric_records),
        "global_vessel_mean_mse": sum(record["global_vessel_mse"] for record in metric_records) / len(metric_records),
        "global_vessel_mean_psnr": sum(record["global_vessel_psnr"] for record in metric_records) / len(metric_records),
        "global_vessel_mean_ssim": sum(record["global_vessel_ssim"] for record in metric_records) / len(metric_records),
    }
    metrics["mean_lpips"] = calculate_lpips_frames(pred_frames, target_frames, device, lpips_model)
    metrics["fvd"] = calculate_fvd_frames(pred_frames, target_frames, fvd_metric)
    metrics["global_vessel_mean_lpips"] = calculate_lpips_frames(
        pred_global_vessel_frames,
        target_global_vessel_frames,
        device,
        lpips_model,
    )
    metrics["global_vessel_fvd"] = calculate_fvd_frames(
        pred_global_vessel_frames,
        target_global_vessel_frames,
        fvd_metric,
    )
    metrics.update(vessel_curve_metrics(pred_frames, target_frames, global_mask.detach().cpu()))
    # A shared model-input template must not change the evaluation/GT time axis.
    metric_time_reference = cfg.get("data", {}).get("time_reference", "normalized")
    if metric_time_reference == "shared_template":
        metric_time_reference = "acquisition_normalized"
    mtt_times = raw_mtt_normalized_times(
        patch_dir,
        sample_mode,
        target_paths,
        len(metric_records),
        metric_time_reference,
    )
    mtt_metric_values, pred_mtt, target_mtt = mtt_metrics(
        pred_frames,
        target_frames,
        mtt_times,
        global_mask.detach().cpu(),
        baseline_frames=args.raw_mtt_baseline_frames,
        keep_outside_mask=args.raw_mtt_keep_outside_mask,
        mask_threshold=args.raw_mtt_mask_threshold,
    )
    metrics.update(mtt_metric_values)
    metrics["mtt_time_unit"] = "normalized"
    metrics["mtt_time_reference"] = str(metric_time_reference)
    metrics["mtt_times"] = [float(t) for t in mtt_times]
    if pred_mtt.numel() and target_mtt.numel():
        mtt_dir = ensure_dir(out_dir / "mtt_seq")
        pred_mtt_normalized = normalize_mtt_in_vessel_mask(
            pred_mtt,
            global_mask.detach().cpu(),
            mask_threshold=args.raw_mtt_mask_threshold,
        )
        target_mtt_normalized = normalize_mtt_in_vessel_mask(
            target_mtt,
            global_mask.detach().cpu(),
            mask_threshold=args.raw_mtt_mask_threshold,
        )
        pred_mtt_path = mtt_dir / f"{sequence_label}_pred_mtt.png"
        target_mtt_path = mtt_dir / f"{sequence_label}_target_mtt.png"
        save_mtt_map_png(pred_mtt_normalized, pred_mtt_path)
        save_mtt_map_png(target_mtt_normalized, target_mtt_path)
        metrics["pred_mtt_path"] = str(pred_mtt_path)
        metrics["target_mtt_path"] = str(target_mtt_path)
        pred_raw_mtt_path = mtt_dir / f"{sequence_label}_pred_mtt_raw.png"
        target_raw_mtt_path = mtt_dir / f"{sequence_label}_target_mtt_raw.png"
        save_mtt_map_png(torch.nan_to_num(pred_mtt.float(), nan=0.0, posinf=1.0, neginf=0.0), pred_raw_mtt_path)
        save_mtt_map_png(torch.nan_to_num(target_mtt.float(), nan=0.0, posinf=1.0, neginf=0.0), target_raw_mtt_path)
        metrics["pred_raw_mtt_path"] = str(pred_raw_mtt_path)
        metrics["target_raw_mtt_path"] = str(target_raw_mtt_path)
        if not args.no_save_raw_mtt_npy:
            pred_mtt_npy_path = mtt_dir / f"{sequence_label}_pred_mtt.npy"
            gt_mtt_npy_path = mtt_dir / f"{sequence_label}_gt_mtt.npy"
            should_write_npy = (
                not args.no_overwrite_raw_mtt_npy
                or not pred_mtt_npy_path.exists()
                or not gt_mtt_npy_path.exists()
            )
            if should_write_npy:
                save_raw_mtt_npy(pred_mtt, pred_mtt_npy_path)
                save_raw_mtt_npy(target_mtt, gt_mtt_npy_path)
            metrics["pred_mtt_npy_path"] = str(pred_mtt_npy_path)
            metrics["gt_mtt_npy_path"] = str(gt_mtt_npy_path)
    return metrics


def infer_one_patch(
    model: torch.nn.Module,
    cfg: dict,
    patch_dir: Path,
    out_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    lpips_model: torch.nn.Module,
    fvd_metric,
) -> dict:
    image_size = int(cfg.get("data", {}).get("image_size", 512))
    previous_frames = int(cfg.get("model", {}).get("previous_frames", 0))

    data_cfg = cfg.get("data", {})
    model_cfg = cfg.get("model", {})
    sample_mode = args.sample_mode or data_cfg.get("sample_mode", "Intensity_step_sample")
    rgb_path = Path(args.rgb_path) if getattr(args, "rgb_path", None) else _find_rgb_input(patch_dir)
    global_mask_path = _find_vessel_mask(patch_dir)
    bypass_vessel_mask_path = find_bypass_vessel_mask(patch_dir) if args.merge_bypass_vessel_mask else None
    guidance_vessel_mask_source = str(data_cfg.get("guidance_vessel_mask_source", "global")).lower()
    input_vessel_mask_source = normalize_input_vessel_mask_source(
        data_cfg.get("input_vessel_mask_source", "global")
    )
    rgb2vessel_root = Path(data_cfg.get("rgb2vessel_root", "E:/ClinicalICG/WhiteLight2ICGVideo/data/RGB2Vessel"))
    mtt_path = _find_mtt_map(patch_dir)
    lsci_path = _find_lsci_input(patch_dir)
    input_mask_path = (
        global_mask_path
        if input_vessel_mask_source == "global"
        else _find_rgb2vessel_input(patch_dir, rgb2vessel_root, input_vessel_mask_source)
    )
    if rgb_path is None:
        raise FileNotFoundError(f"No RGB baseline found in {patch_dir}")
    if not rgb_path.is_file():
        raise FileNotFoundError(f"RGB input does not exist: {rgb_path}")

    use_vessel_input = bool(data_cfg.get("include_vessel_mask", True) or model_cfg.get("include_vessel_mask", True))
    if use_vessel_input and input_mask_path is None:
        raise MissingInputVesselMaskError(
            f"Vessel mask input source '{input_vessel_mask_source}' is enabled but no matching mask was found for {patch_dir}"
        )

    target_sequences = sampled_target_sequences(patch_dir, cfg, args)
    targets = [path for sequence in target_sequences for path in sequence["targets"]]
    times = [time for sequence in target_sequences for time in sequence["times"]]
    if not times:
        raise RuntimeError(f"No inference times generated for {patch_dir}")
    multi_sequence = len(target_sequences) > 1

    pred_dir = ensure_dir(out_dir / "pred")
    aux_mask_dir = ensure_dir(out_dir / "aux_mask") if int(cfg.get("model", {}).get("output_channels", 1)) > 1 else None
    target_dir = ensure_dir(out_dir / "target") if args.save_targets and targets else None
    panel_dir = ensure_dir(out_dir / "panel") if args.save_panels and targets else None

    rgb = _load_rgb(rgb_path, image_size)[None].to(device)
    if global_mask_path is not None:
        global_mask = _load_gray(global_mask_path, image_size, "mask")[None].to(device)
    else:
        global_mask = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32, device=device)
    if use_vessel_input and input_mask_path is not None:
        input_mask_kind = "image" if input_vessel_mask_source == "rgb2vessel_prob" else "mask"
        input_mask = _load_gray(input_mask_path, image_size, input_mask_kind)[None].to(device)
    else:
        input_mask = torch.zeros_like(global_mask)
    if args.merge_bypass_vessel_mask:
        if bypass_vessel_mask_path is None:
            raise MissingInputVesselMaskError(f"No bypass vessel mask was found for {patch_dir}")
        bypass_mask = _load_gray(bypass_vessel_mask_path, image_size, "mask")[None].to(device)
        radius = max(0, int(args.bypass_dilate_pixels))
        if radius > 0:
            bypass_mask = torch.nn.functional.max_pool2d(
                bypass_mask,
                kernel_size=radius * 2 + 1,
                stride=1,
                padding=radius,
            )
        input_mask = torch.maximum(input_mask, bypass_mask.to(dtype=input_mask.dtype))
    use_mtt = bool(cfg.get("data", {}).get("include_mtt", False) or cfg.get("model", {}).get("include_mtt", False))
    if use_mtt and mtt_path is None:
        raise FileNotFoundError(f"MTT input is enabled but no *_roi_mtt image was found in {patch_dir}")
    if use_mtt:
        mtt_base = _load_gray(mtt_path, image_size, "image")[None].to(device)
    else:
        mtt_base = torch.zeros((1, 1, image_size, image_size), dtype=torch.float32, device=device)
    use_lsci = bool(cfg.get("data", {}).get("include_lsci", False) or cfg.get("model", {}).get("include_lsci", False))
    if use_lsci and lsci_path is None:
        raise FileNotFoundError(f"LSCI input is enabled but no *_LSCI_Reg_1FRigid_baseline_frame image was found in {patch_dir}")
    if use_lsci:
        lsci = _load_gray(lsci_path, image_size, "image")[None].to(device)
    else:
        lsci = torch.zeros_like(global_mask)

    frame_records = []
    pred_metric_frames: list[torch.Tensor] = []
    target_metric_frames: list[torch.Tensor] = []
    pred_global_vessel_metric_frames: list[torch.Tensor] = []
    target_global_vessel_metric_frames: list[torch.Tensor] = []
    target_metric_paths: list[Path] = []
    metric_items: list[dict] = []

    model.eval()
    with torch.no_grad():
        for sequence in target_sequences:
            offset = int(sequence["frame_step_offset"])
            sequence_targets = sequence["targets"]
            sequence_times = sequence["times"]
            sequence_label = f"offset{offset}"
            sequence_pred_dir = pred_dir
            sequence_aux_mask_dir = aux_mask_dir
            sequence_target_dir = target_dir
            sequence_panel_dir = panel_dir
            prev_buffer = torch.zeros((1, previous_frames, image_size, image_size), dtype=torch.float32, device=device)
            prev_time_buffer = torch.zeros((1, previous_frames), dtype=torch.float32, device=device)
            convgru_hidden = None
            for sequence_frame, t in enumerate(sequence_times, start=1):
                time = torch.tensor([[t]], dtype=torch.float32, device=device)
                target_path = sequence_targets[sequence_frame - 1] if sequence_frame - 1 < len(sequence_targets) else None
                output_frame = inference_output_frame_number(target_path, sequence_frame, int(sequence["frame_step"]), offset)
                output_name = f"{output_frame:03d}.png"
                mask = input_mask
                mtt = mtt_base * global_mask if use_mtt and data_cfg.get("mask_mtt_with_vessel", False) else mtt_base
                if hasattr(model, "forward_step"):
                    output, convgru_hidden = model(
                        rgb,
                        time,
                        mask,
                        prev_buffer,
                        mtt,
                        prev_time_buffer,
                        lsci,
                        hidden=convgru_hidden,
                        return_hidden=True,
                    )
                else:
                    output = model(rgb, time, mask, prev_buffer, mtt, prev_time_buffer, lsci)
                pred, pred_aux_mask = split_generator_output(output)
                pred_path = sequence_pred_dir / output_name
                Image.fromarray(tensor_to_uint8(pred[0])).save(pred_path)
                aux_mask_path = None
                if pred_aux_mask is not None:
                    if sequence_aux_mask_dir is None:
                        sequence_aux_mask_dir = ensure_dir(out_dir / "aux_mask")
                    aux_mask_path = sequence_aux_mask_dir / output_name
                    Image.fromarray(tensor_to_uint8(pred_aux_mask[0])).save(aux_mask_path)

                record = {
                    "frame": output_frame,
                    "sequence_index": int(sequence["sequence_index"]),
                    "sequence_frame": sequence_frame,
                    "frame_step": int(sequence["frame_step"]),
                    "frame_step_offset": offset,
                    "time": float(t),
                    "pred_path": str(pred_path),
                }
                if aux_mask_path is not None:
                    record["aux_mask_path"] = str(aux_mask_path)

                if target_path is not None:
                    target = _load_gray(target_path, image_size, "image").to(device)
                    l1 = torch.mean(torch.abs(pred[0] - target)).item()
                    mse = torch.mean((pred[0] - target) ** 2).item()
                    pred_np = pred[0].detach().cpu().numpy()
                    target_np = target.detach().cpu().numpy()
                    global_vessel_pred = pred[0] * global_mask[0]
                    global_vessel_target = target * global_mask[0]
                    global_vessel_l1, global_vessel_mse = masked_l1_mse(pred[0], target, global_mask[0])
                    global_vessel_pred_np = global_vessel_pred.detach().cpu().numpy()
                    global_vessel_target_np = global_vessel_target.detach().cpu().numpy()
                    record.update(
                        {
                            "target_path": str(target_path),
                            "l1": l1,
                            "mse": mse,
                            "psnr": img_psnr(pred_np, target_np),
                            "ssim": calculate_ssim_function(pred_np, target_np),
                            "global_vessel_l1": global_vessel_l1,
                            "global_vessel_mse": global_vessel_mse,
                            "global_vessel_psnr": psnr_from_mse(global_vessel_mse),
                            "global_vessel_ssim": calculate_ssim_function(global_vessel_pred_np, global_vessel_target_np),
                        }
                    )
                    metric_items.append(
                        {
                            "sort_key": target_path.name,
                            "sequence_index": int(sequence["sequence_index"]),
                            "sequence_label": sequence_label,
                            "pred": pred[0].detach().cpu(),
                            "target": target.detach().cpu(),
                            "global_vessel_pred": global_vessel_pred.detach().cpu(),
                            "global_vessel_target": global_vessel_target.detach().cpu(),
                            "target_path": target_path,
                        }
                    )
                    if sequence_target_dir is not None:
                        Image.fromarray(tensor_to_uint8(target)).save(sequence_target_dir / output_name)
                    if sequence_panel_dir is not None:
                        save_preview_panel(
                            rgb[0].detach().cpu(),
                            mask[0].detach().cpu(),
                            pred[0].detach().cpu(),
                            target.detach().cpu(),
                            sequence_panel_dir / output_name,
                            title=(
                                f"patch {patch_dir.name} | {sequence_label} | frame "
                                f"{output_frame:03d} | seq {sequence_frame:03d}/{len(sequence_times):03d} | t={t:.4f}"
                            ),
                            input_panels=[
                                (label, tensor.detach().cpu())
                                for label, tensor in static_input_panels(cfg, rgb[0], mask[0], mtt[0], lsci[0])
                            ] + ([("AuxMask", pred_aux_mask[0].detach().cpu())] if pred_aux_mask is not None else []),
                        )

                frame_records.append(record)
                if previous_frames > 0:
                    prev_buffer = torch.cat([prev_buffer[:, 1:], pred.detach()], dim=1)
                    prev_time_buffer = torch.cat([prev_time_buffer[:, 1:], time.detach()], dim=1)

    frame_records = sorted(frame_records, key=lambda record: int(record.get("frame", 0)))

    metrics = {}
    sequence_metric_summaries = []
    if metric_items:
        for sequence in target_sequences:
            sequence_index = int(sequence["sequence_index"])
            sequence_label = f"offset{int(sequence['frame_step_offset'])}"
            sequence_records = [
                record for record in frame_records
                if int(record.get("sequence_index", -1)) == sequence_index and "l1" in record
            ]
            sequence_items = [
                item for item in metric_items
                if int(item.get("sequence_index", -1)) == sequence_index
            ]
            sequence_metrics = compute_metrics_for_sequence(
                sequence_records,
                sequence_items,
                patch_dir,
                sample_mode,
                cfg,
                args,
                device,
                lpips_model,
                fvd_metric,
                global_mask[0],
                out_dir,
                sequence_label,
            )
            if sequence_metrics:
                sequence_metric_summaries.append(
                    {
                        "sequence_index": sequence_index,
                        "frame_step": int(sequence["frame_step"]),
                        "frame_step_offset": int(sequence["frame_step_offset"]),
                        "num_frames": len(sequence_records),
                        "metrics": sequence_metrics,
                    }
                )
        metrics = average_numeric_metrics([summary["metrics"] for summary in sequence_metric_summaries])
        if sequence_metric_summaries:
            metrics["metric_mode"] = "sequence_average"
            metrics["metric_sequence_count"] = len(sequence_metric_summaries)

    summary = {
        "patch_id": patch_dir.name,
        "subject": subject_from_patch_dir(patch_dir),
        "rgb_path": str(rgb_path),
        "mask_path": str(global_mask_path) if global_mask_path is not None else None,
        "input_vessel_mask_source": input_vessel_mask_source,
        "input_mask_path": str(input_mask_path) if input_mask_path is not None else None,
        "merge_bypass_vessel_mask": bool(args.merge_bypass_vessel_mask),
        "bypass_vessel_mask_path": str(bypass_vessel_mask_path) if bypass_vessel_mask_path is not None else None,
        "bypass_dilate_pixels": int(args.bypass_dilate_pixels),
        "guidance_vessel_mask_source": guidance_vessel_mask_source,
        "mtt_path": str(mtt_path) if mtt_path is not None else None,
        "lsci_path": str(lsci_path) if lsci_path is not None else None,
        "out_dir": str(out_dir),
        "aux_mask_dir": str(aux_mask_dir) if aux_mask_dir is not None else None,
        "num_frames": len(times),
        "num_sequences": len(target_sequences),
        "sequences": [
            {
                "sequence_index": int(sequence["sequence_index"]),
                "frame_step": int(sequence["frame_step"]),
                "frame_step_offset": int(sequence["frame_step_offset"]),
                "num_frames": len(sequence["times"]),
            }
            for sequence in target_sequences
        ],
        "inference_sampling": inference_sampling_config(cfg, args),
        "previous_frames": previous_frames,
        "has_targets": bool(targets),
        "metrics": metrics,
        "sequence_metrics": sequence_metric_summaries,
        "frames": frame_records,
    }
    ensure_dir(out_dir)
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.copy_config:
        (out_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


class MissingInputVesselMaskError(FileNotFoundError):
    pass


def main() -> None:
    args = parse_args()
    if args.rgb_path and not args.patch_dir:
        raise ValueError("--rgb_path requires --patch_dir to avoid reusing one RGB image across unrelated cases")
    exp_dir = experiment_dir(args)
    ckpt_path = checkpoint_path(args, exp_dir)
    ckpt = load_checkpoint(ckpt_path)
    cfg_file = config_path(args, exp_dir)
    if cfg_file is not None:
        cfg = load_yaml(cfg_file)
    elif "config" in ckpt:
        cfg = ckpt["config"]
    else:
        raise ValueError("No config found. Provide --config or use a checkpoint that contains config.")
    if args.input_vessel_mask_source is not None:
        cfg.setdefault("data", {})["input_vessel_mask_source"] = args.input_vessel_mask_source
    if args.use_saved_time_template and (args.time_template_file or args.time_template_split):
        raise ValueError("--use_saved_time_template cannot be combined with a template file/split override.")
    # args.sample_mode selects the GT frame set. It may intentionally differ
    # from the sample mode used to compute the frozen mean-time statistics.
    configure_time_template(cfg, args.time_template_file, args.time_template_split)
    if args.use_saved_time_template and cfg.get("data", {}).get("time_reference") != "shared_template":
        raise ValueError("This experiment does not contain a saved shared time template.")
    device = choose_device(args.device)

    model = build_model(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model"])

    patch_dirs = resolve_patch_dirs(args, cfg, exp_dir)
    if not patch_dirs:
        raise RuntimeError("No patch directories found for inference.")
    if args.require_targets:
        data_cfg = cfg.get("data", {})
        sample_mode = args.sample_mode or data_cfg.get("sample_mode", "Intensity_step_sample")
        expected_frames = int(
            args.frames
            or data_cfg.get("inference_max_frames_per_patch")
            or data_cfg.get("max_frames_per_patch")
            or 32
        )
        bad_targets = []
        for patch_dir in patch_dirs:
            target_count = len(_list_target_frames(patch_dir, sample_mode))
            if target_count != expected_frames:
                bad_targets.append(f"{patch_dir.name} ({target_count}/{expected_frames})")
        if bad_targets:
            preview = ", ".join(bad_targets[:10])
            suffix = " ..." if len(bad_targets) > 10 else ""
            raise RuntimeError(
                f"Requested GT mode {sample_mode!r} is incomplete for "
                f"{len(bad_targets)} patch(es): {preview}{suffix}"
            )

    batch_mode = len(patch_dirs) > 1
    output_root = default_output_root(args, exp_dir, ckpt_path)
    if cfg.get("data", {}).get("time_reference") == "shared_template":
        # Never let a CLI time override silently overwrite the original inference.
        if args.time_template_file and not args.out_dir:
            raise ValueError("Mean-time inference overrides require a distinct --out_dir under the experiment/inference directory.")
        if output_root is not None:
            root = ensure_dir(output_root)
            (root / "time_template.json").write_text(json.dumps({
                "time_reference": "shared_template",
                "values": cfg["data"]["time_template"],
                "provenance": cfg["data"].get("time_template_provenance"),
                "metric_time_reference": "acquisition_normalized",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
    lpips_model = build_lpips_model(device)
    fvd_metric = build_fvd_metric(device, args.fvd_model, args.fvd_ckpt_path)
    summaries = []
    skipped_patches = []
    for patch_dir in tqdm(patch_dirs, desc="Inference"):
        out_dir = default_output_dir(patch_dir, args, batch_mode, output_root)
        try:
            summaries.append(infer_one_patch(model, cfg, patch_dir, out_dir, device, args, lpips_model, fvd_metric))
        except MissingInputVesselMaskError as exc:
            if not args.skip_missing_input_vessel_mask:
                raise
            skipped = {
                "patch_id": patch_dir.name,
                "subject": subject_from_patch_dir(patch_dir),
                "reason": str(exc),
            }
            skipped_patches.append(skipped)
            tqdm.write(f"Skip {patch_dir.name}: selected input vessel mask is missing")

    if output_root is not None:
        root = ensure_dir(output_root)
        metric_keys = [
            "mean_l1",
            "mean_mse",
            "mean_psnr",
            "mean_ssim",
            "mean_lpips",
            "fvd",
            "global_vessel_mean_l1",
            "global_vessel_mean_mse",
            "global_vessel_mean_psnr",
            "global_vessel_mean_ssim",
            "global_vessel_mean_lpips",
            "global_vessel_fvd",
            "vessel_curve_mae",
            "vessel_curve_delta_mae",
            "vessel_curve_correlation",
            "raw_mtt_l1",
            "raw_mtt_mse",
            "raw_mtt_psnr",
            "raw_mtt_ssim",
            "global_vessel_mtt_l1",
            "global_vessel_mtt_mse",
            "global_vessel_mtt_psnr",
            "global_vessel_mtt_ssim",
        ]
        split_subjects = load_experiment_splits(exp_dir)

        def write_batch_summary(
            filename: str,
            selected_summaries: list[dict],
            selected_skipped: list[dict],
            selected_splits: list[str],
        ) -> None:
            aggregate_metrics = {}
            for key in metric_keys:
                values = [
                    summary["metrics"][key]
                    for summary in selected_summaries
                    if key in summary.get("metrics", {})
                ]
                if values:
                    aggregate_metrics[key] = sum(values) / len(values)
            batch_summary = {
                "experiment_name": args.experiment_name,
                "checkpoint": str(ckpt_path),
                "checkpoint_type": args.checkpoint_type,
                "config": str(cfg_file) if cfg_file is not None else None,
                "inference_sampling": inference_sampling_config(cfg, args),
                "device": str(device),
                "num_patches": len(selected_summaries),
                "num_skipped_patches": len(selected_skipped),
                "splits": selected_splits,
                "metrics": aggregate_metrics,
                "patches": selected_summaries,
                "skipped_patches": selected_skipped,
            }
            (root / filename).write_text(
                json.dumps(batch_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        write_batch_summary("inference_summary.json", summaries, skipped_patches, list(args.splits))
        for split_name in args.splits:
            subjects = set(split_subjects.get(split_name, []))
            split_summaries = [summary for summary in summaries if summary["subject"] in subjects]
            split_skipped = [item for item in skipped_patches if item["subject"] in subjects]
            write_batch_summary(
                f"inference_summary_{split_name}.json",
                split_summaries,
                split_skipped,
                [split_name],
            )


if __name__ == "__main__":
    main()

