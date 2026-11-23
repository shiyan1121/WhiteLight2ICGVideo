from __future__ import annotations

import csv
import random
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .utils import subject_from_patch_dir
from .time_templates import validate_time_template

PATCH_RE = re.compile(r"^S\d+.*_\d+$")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
FRAME_INDEX_RE = re.compile(r"(?:^|_)f(\d+)(?=\D|$)", re.IGNORECASE)


def _resize_mode(kind: str) -> int:
    return Image.Resampling.NEAREST if kind == "mask" else Image.Resampling.BILINEAR


def _load_rgb(path: Path, image_size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB").resize((image_size, image_size), _resize_mode("image"))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def _load_gray(path: Path, image_size: int, kind: str = "image") -> torch.Tensor:
    img = Image.open(path).convert("L").resize((image_size, image_size), _resize_mode(kind))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    if kind == "mask":
        arr = (arr > 0.5).astype(np.float32)
    return torch.from_numpy(arr)[None]


def _find_rgb_input(patch_dir: Path) -> Path | None:
    preferred = patch_dir / "rgb_keyframe_patches" / "baseline_frame.png"
    if preferred.exists():
        return preferred
    fallback = patch_dir / f"{patch_dir.name}_baseline_frame.png"
    if fallback.exists():
        return fallback
    candidates = sorted(patch_dir.glob("*_baseline_frame.png"))
    return candidates[0] if candidates else None


def _find_vessel_mask(patch_dir: Path) -> Path | None:
    preferred = patch_dir / f"{patch_dir.name}_vessel_mask.png"
    if preferred.exists():
        return preferred
    candidates = sorted(patch_dir.glob("*vessel_mask*.png"))
    return candidates[0] if candidates else None


def _find_rgb2vessel_input(patch_dir: Path, rgb2vessel_root: Path | None, source: str) -> Path | None:
    if rgb2vessel_root is None:
        return None
    subdir = {"rgb2vessel_mask": "masks", "rgb2vessel_prob": "prob"}.get(source)
    if subdir is None:
        return None
    mask_dir = rgb2vessel_root / subdir
    preferred = mask_dir / f"{patch_dir.name}_baseline_frame.png"
    if preferred.exists():
        return preferred
    candidates = sorted(p for p in mask_dir.glob(f"{patch_dir.name}*.*") if p.suffix.lower() in IMAGE_EXTS)
    return candidates[0] if candidates else None


def _normalize_input_vessel_mask_source(source: str) -> str:
    source = str(source or "global").lower()
    if source in {"original", "gt"}:
        source = "global"
    if source in {"rgb2vessel", "rgb2vessel_masks", "rgb_mask", "mask"}:
        source = "rgb2vessel_mask"
    if source in {"rgb2vessel_probs", "rgb_prob", "prob"}:
        source = "rgb2vessel_prob"
    if source in {"mixed", "mix", "mixed_rgb2vessel", "mixed_global_rgb2vessel"}:
        source = "mixed_global_rgb2vessel_mask"
    if source in {"mixed_mask", "mix_mask", "mixed_global_rgb2vessel_mask"}:
        source = "mixed_global_rgb2vessel_mask"
    if source in {"mixed_prob", "mix_prob", "mixed_global_rgb2vessel_prob"}:
        source = "mixed_global_rgb2vessel_prob"
    return source


def _mixed_rgb_source(source: str) -> str | None:
    if source == "mixed_global_rgb2vessel_mask":
        return "rgb2vessel_mask"
    if source == "mixed_global_rgb2vessel_prob":
        return "rgb2vessel_prob"
    return None


def _list_keyframe_vessel_masks(patch_dir: Path, sample_mode: str) -> dict[str, Path]:
    mask_dir = patch_dir / "vessel_mask_keyframe_patches" / sample_mode
    if not mask_dir.exists():
        return {}
    return {
        p.name: p
        for p in sorted(mask_dir.iterdir())
        if p.suffix.lower() in IMAGE_EXTS
    }


def _find_mtt_map(patch_dir: Path) -> Path | None:
    preferred = patch_dir / f"{patch_dir.name}_roi_mtt.png"
    if preferred.exists():
        return preferred
    candidates = sorted(p for p in patch_dir.glob("*roi_mtt*.*") if p.suffix.lower() in IMAGE_EXTS)
    return candidates[0] if candidates else None


def _find_lsci_input(patch_dir: Path) -> Path | None:
    preferred = patch_dir / f"{patch_dir.name}_LSCI_Reg_1FRigid_baseline_frame.png"
    if preferred.exists():
        return preferred
    candidates = sorted(p for p in patch_dir.glob("*LSCI_Reg_1FRigid_baseline_frame.*") if p.suffix.lower() in IMAGE_EXTS)
    return candidates[0] if candidates else None


def _list_target_frames(patch_dir: Path, sample_mode: str) -> list[Path]:
    target_dir = patch_dir / "icg_keyframe_patches" / sample_mode
    if not target_dir.exists():
        return []
    paths = sorted([p for p in target_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])

    # Re-extraction can leave stale PNGs with the same order prefix but an old
    # source-frame suffix (for example both 004_f00476 and 004_f00477).  The
    # CSV is the authoritative 32-frame sequence, so use its exact order/frame
    # identities whenever every row can be resolved.  Falling back preserves
    # compatibility with datasets that have no CSV.
    rows = _read_keyframe_csv_rows(patch_dir, sample_mode)
    if rows:
        by_stem = {path.stem.lower(): path for path in paths}
        selected = []
        try:
            rows = sorted(rows, key=lambda row: int(round(float(row["order"]))))
            for row in rows:
                order = int(round(float(row["order"])))
                frame = int(round(float(row["frame"])))
                path = by_stem.get(f"{order:03d}_f{frame:05d}".lower())
                if path is None:
                    selected = []
                    break
                selected.append(path)
        except (KeyError, TypeError, ValueError):
            selected = []
        if selected:
            return selected
    return paths


def _find_keyframe_csv(patch_dir: Path, sample_mode: str) -> Path | None:
    preferred = patch_dir / f"{patch_dir.name}_keyframes_{sample_mode}.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(patch_dir.glob(f"*keyframes*{sample_mode}*.csv"))
    return candidates[0] if candidates else None


def _parse_frame_index(path: Path) -> int | None:
    match = FRAME_INDEX_RE.search(path.stem)
    return int(match.group(1)) if match else None


def _read_keyframe_csv_rows(patch_dir: Path, sample_mode: str) -> list[dict]:
    csv_path = _find_keyframe_csv(patch_dir, sample_mode)
    if csv_path is None:
        return []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _peak_relative_times(target_paths: list[Path], sample_mode: str, reference: str) -> dict[Path, tuple[float, int | None]] | None:
    if not target_paths:
        return {}
    patch_dir = target_paths[0].parents[2]
    rows = _read_keyframe_csv_rows(patch_dir, sample_mode)
    if not rows:
        return None

    peak_rows = [row for row in rows if str(row.get("name", "")).lower() == "rise_100peak"]
    if not peak_rows:
        peak_rows = [row for row in rows if "100peak" in str(row.get("name", "")).lower()]
    if not peak_rows:
        return None

    use_frame = reference in {"peak_relative_frame", "peak_frame", "frame_peak"}

    def as_float(row: dict, key: str) -> float | None:
        try:
            return float(row.get(key, ""))
        except (TypeError, ValueError):
            return None

    peak_value = as_float(peak_rows[0], "frame" if use_frame else "time_sec")
    if peak_value is None or peak_value <= 0:
        peak_value = as_float(peak_rows[0], "frame")
        use_frame = True
    if peak_value is None or peak_value <= 0:
        return None

    by_frame = {}
    for row in rows:
        frame = as_float(row, "frame")
        if frame is not None:
            by_frame[int(round(frame))] = row

    out = {}
    for row, path in zip(rows, target_paths):
        frame_index = _parse_frame_index(path)
        matched = by_frame.get(frame_index, row) if frame_index is not None else row
        value = as_float(matched, "frame" if use_frame else "time_sec")
        if value is None:
            value = as_float(matched, "frame")
        if value is None:
            return None
        out[path] = (float(value) / float(peak_value), frame_index)
    return out


def _normalized_times(
    target_paths: list[Path],
    sample_mode: str = "Intensity_step_sample",
    time_reference: str = "normalized",
    time_template: Sequence[float] | None = None,
) -> dict[Path, tuple[float, int | None]]:
    reference = str(time_reference or "normalized").lower()
    if not target_paths:
        return {}
    if reference == "shared_template":
        times = validate_time_template(time_template, len(target_paths))
        return {path: (t, _parse_frame_index(path)) for path, t in zip(target_paths, times)}
    if reference == "acquisition_normalized":
        # Match the original acquisition CSV, never substitute uniform frame order.
        patch_dir = target_paths[0].parents[2]
        rows = _read_keyframe_csv_rows(patch_dir, sample_mode)
        by_frame = {}
        for row in rows:
            try:
                index = int(float(row["frame"]))
                seconds = float(row["time_sec"])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(seconds):
                by_frame[index] = seconds
        indices = [_parse_frame_index(path) for path in target_paths]
        if any(index is None or index not in by_frame for index in indices):
            raise ValueError(f"Missing original time_sec in keyframe CSV for {patch_dir}; no frame-index fallback.")
        seconds = [by_frame[index] for index in indices]
        duration = seconds[-1] - seconds[0]
        if duration <= 0 or any(b < a for a, b in zip(seconds, seconds[1:])):
            raise ValueError(f"Invalid original acquisition timestamps for {patch_dir}.")
        return {path: ((t - seconds[0]) / duration, index)
                for path, t, index in zip(target_paths, seconds, indices)}
    if reference in {
        "sequence_index_1based",
        "sequence_index",
        "frame_idx_1based",
        "frame_index_1based",
    }:
        n_frames = len(target_paths)
        return {
            path: ((frame_order + 1) / float(n_frames), _parse_frame_index(path))
            for frame_order, path in enumerate(target_paths)
        }
    if reference in {"peak_relative", "peak", "peak_relative_time", "time_peak", "peak_relative_frame", "peak_frame", "frame_peak"}:
        peak_times = _peak_relative_times(target_paths, sample_mode, reference)
        if peak_times is not None:
            return peak_times

    frame_indices = [_parse_frame_index(path) for path in target_paths]
    if target_paths and all(index is not None for index in frame_indices):
        first = min(frame_indices)
        last = max(frame_indices)
        denom = float(last - first)
        if denom > 0:
            return {
                path: ((index - first) / denom, index)
                for path, index in zip(target_paths, frame_indices)
            }
        return {path: (0.0, index) for path, index in zip(target_paths, frame_indices)}

    n_frames = len(target_paths)
    return {
        path: (0.0 if n_frames <= 1 else frame_order / float(n_frames - 1), None)
        for frame_order, path in enumerate(target_paths)
    }


def _normalize_frame_step_offsets(frame_step: int, frame_step_offsets: str | int | Sequence[int] | None) -> list[int]:
    if frame_step <= 1:
        return [0]
    if frame_step_offsets is None:
        return [0]
    if isinstance(frame_step_offsets, str):
        value = frame_step_offsets.strip().lower()
        if value in {"all", "*"}:
            return list(range(frame_step))
        if value in {"", "0", "first", "default", "none"}:
            return [0]
        offsets = [int(part) for part in re.split(r"[,;\s]+", value) if part]
    elif isinstance(frame_step_offsets, int):
        offsets = [int(frame_step_offsets)]
    else:
        offsets = [int(offset) for offset in frame_step_offsets]
    offsets = sorted(set(offsets))
    bad_offsets = [offset for offset in offsets if offset < 0 or offset >= frame_step]
    if bad_offsets:
        raise ValueError(
            f"frame_step_offsets must be in [0, {frame_step - 1}] when frame_step={frame_step}; "
            f"got {bad_offsets}."
        )
    return offsets or [0]


class PatchFrameDataset(Dataset):
    """Frame-wise dataset for RGB/mask/time conditioned ICG prediction."""

    def __init__(
        self,
        data_root: str | Path,
        sample_mode: str = "Intensity_step_sample",
        subjects: Sequence[str] | None = None,
        image_size: int = 512,
        include_vessel_mask: bool = True,
        input_vessel_mask_source: str = "global",
        rgb2vessel_root: str | Path | None = None,
        guidance_vessel_mask_source: str | None = None,
        include_mtt: bool = False,
        mask_mtt_with_vessel: bool = False,
        include_lsci: bool = False,
        max_frames_per_patch: int | None = None,
        frame_step: int = 1,
        frame_step_offsets: str | int | Sequence[int] | None = None,
        previous_frames: int = 0,
        time_reference: str = "normalized",
        require_complete_patch: bool = True,
        time_template: Sequence[float] | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.sample_mode = sample_mode
        self.subjects = set(subjects) if subjects else None
        self.image_size = int(image_size)
        self.include_vessel_mask = include_vessel_mask
        self.input_vessel_mask_source = _normalize_input_vessel_mask_source(input_vessel_mask_source or "global")
        if self.input_vessel_mask_source not in {
            "global",
            "rgb2vessel_mask",
            "rgb2vessel_prob",
            "mixed_global_rgb2vessel_mask",
            "mixed_global_rgb2vessel_prob",
        }:
            raise ValueError(f"Unknown input_vessel_mask_source: {self.input_vessel_mask_source}")
        self.rgb2vessel_root = Path(rgb2vessel_root) if rgb2vessel_root else None
        self.guidance_vessel_mask_source = str(guidance_vessel_mask_source or "global").lower()
        if self.guidance_vessel_mask_source not in {"global", "keyframe"}:
            raise ValueError(f"Unknown guidance_vessel_mask_source: {self.guidance_vessel_mask_source}")
        self.include_mtt = include_mtt
        self.mask_mtt_with_vessel = mask_mtt_with_vessel
        self.include_lsci = bool(include_lsci)
        self.frame_step = max(1, int(frame_step))
        self.frame_step_offsets = _normalize_frame_step_offsets(self.frame_step, frame_step_offsets)
        self.previous_frames = int(previous_frames)
        self.time_reference = str(time_reference or "normalized").lower()
        self.time_template = time_template
        self.require_complete_patch = require_complete_patch
        self.samples: list[dict] = []

        patch_dirs = [p for p in self.data_root.iterdir() if p.is_dir() and PATCH_RE.match(p.name)]
        for patch_dir in sorted(patch_dirs):
            subject = subject_from_patch_dir(patch_dir)
            if self.subjects is not None and subject not in self.subjects:
                continue

            rgb_path = _find_rgb_input(patch_dir)
            global_mask_path = _find_vessel_mask(patch_dir)
            mixed_rgb_source = _mixed_rgb_source(self.input_vessel_mask_source)
            rgb2vessel_input_path = (
                None
                if self.input_vessel_mask_source == "global"
                else _find_rgb2vessel_input(
                    patch_dir,
                    self.rgb2vessel_root,
                    mixed_rgb_source or self.input_vessel_mask_source,
                )
            )
            input_mask_path = global_mask_path if self.input_vessel_mask_source == "global" else rgb2vessel_input_path
            keyframe_mask_paths = _list_keyframe_vessel_masks(patch_dir, sample_mode)
            mtt_path = _find_mtt_map(patch_dir)
            lsci_path = _find_lsci_input(patch_dir)
            all_target_paths = _list_target_frames(patch_dir, sample_mode)
            source_target_paths = (
                all_target_paths[: int(max_frames_per_patch)]
                if max_frames_per_patch is not None
                else all_target_paths
            )
            time_target_paths = source_target_paths if self.time_reference == "shared_template" else all_target_paths
            normalized_times = _normalized_times(
                time_target_paths, sample_mode, self.time_reference, self.time_template
            )
            offset_sequences: list[tuple[int, list[Path]]] = []
            for step_offset in self.frame_step_offsets:
                offset_target_paths = source_target_paths[step_offset:: self.frame_step]
                if offset_target_paths:
                    offset_sequences.append((step_offset, offset_target_paths))
            sampled_target_paths = [path for _, paths in offset_sequences for path in paths]

            complete = (
                rgb_path is not None
                and sampled_target_paths
                and (
                    not include_vessel_mask
                    or (
                        global_mask_path is not None
                        and input_mask_path is not None
                        and (
                            self.guidance_vessel_mask_source == "global"
                            or (self.guidance_vessel_mask_source == "keyframe" and bool(keyframe_mask_paths))
                        )
                    )
                )
                and (mtt_path is not None or not include_mtt)
                and (lsci_path is not None or not self.include_lsci)
            )
            if require_complete_patch and not complete:
                continue
            if rgb_path is None or not offset_sequences:
                continue

            multi_offset = len(self.frame_step_offsets) > 1
            for step_offset, target_paths in offset_sequences:
                sequence_id = (
                    patch_dir.name
                    if not multi_offset and step_offset == 0
                    else f"{patch_dir.name}_step{self.frame_step}_offset{step_offset}"
                )
                n_frames = len(target_paths)
                for frame_order, target_path in enumerate(target_paths):
                    keyframe_mask_path = keyframe_mask_paths.get(target_path.name)
                    guidance_mask_path = (
                        keyframe_mask_path
                        if self.guidance_vessel_mask_source == "keyframe"
                        else global_mask_path
                    )
                    if include_vessel_mask and guidance_mask_path is None:
                        if require_complete_patch:
                            continue
                        guidance_mask_path = global_mask_path
                    t, frame_index = normalized_times[target_path]
                    prev_target_paths = []
                    prev_times = []
                    for offset in range(self.previous_frames, 0, -1):
                        prev_index = frame_order - offset
                        prev_path = target_paths[prev_index] if prev_index >= 0 else None
                        prev_target_paths.append(prev_path)
                        prev_times.append(normalized_times[prev_path][0] if prev_path is not None else 0.0)
                    next_target_path = target_paths[frame_order + 1] if frame_order + 1 < n_frames else None
                    next_time = normalized_times[next_target_path][0] if next_target_path is not None else t
                    self.samples.append(
                        {
                            "patch_id": patch_dir.name,
                            "sequence_id": sequence_id,
                            "subject": subject,
                            "rgb_path": rgb_path,
                            "mask_path": global_mask_path,
                            "input_mask_path": input_mask_path,
                            "rgb2vessel_input_mask_path": rgb2vessel_input_path,
                            "input_vessel_mask_source": self.input_vessel_mask_source,
                            "guidance_mask_path": guidance_mask_path,
                            "keyframe_mask_path": keyframe_mask_path,
                            "mtt_path": mtt_path,
                            "lsci_path": lsci_path,
                            "target_path": target_path,
                            "next_target_path": next_target_path,
                            "prev_target_paths": prev_target_paths,
                            "prev_times": prev_times,
                            "next_time": next_time,
                            "frame_order": frame_order,
                            "frame_step": self.frame_step,
                            "frame_step_offset": step_offset,
                            "frame_index": frame_index,
                            "n_frames": n_frames,
                            "time": t,
                        }
                    )

        self.patch_to_indices: dict[str, list[int]] = {}
        for idx, sample in enumerate(self.samples):
            self.patch_to_indices.setdefault(sample.get("sequence_id", sample["patch_id"]), []).append(idx)

    def _load_input_mask(self, item: dict, global_mask: torch.Tensor) -> torch.Tensor:
        if not self.include_vessel_mask:
            return torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        source = str(item.get("input_vessel_mask_source", "global")).lower()
        rgb_source = _mixed_rgb_source(source)
        if rgb_source is not None:
            use_rgb_mask = random.random() < 0.5
            if use_rgb_mask and item.get("rgb2vessel_input_mask_path") is not None:
                kind = "image" if rgb_source == "rgb2vessel_prob" else "mask"
                return _load_gray(item["rgb2vessel_input_mask_path"], self.image_size, kind)
            return global_mask.clone()

        input_mask_path = item.get("input_mask_path")
        if input_mask_path is not None:
            kind = "image" if source == "rgb2vessel_prob" else "mask"
            return _load_gray(input_mask_path, self.image_size, kind)
        return torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

    def get_patch_sequence(self, patch_id: str) -> dict:
        """Load all sampled frames for one patch/offset sequence."""
        sequence_id = str(patch_id)
        indices = self.patch_to_indices.get(sequence_id, [])
        if not indices:
            raise KeyError(f"Unknown patch_id/sequence_id for sequence loading: {patch_id}")
        items = [self.samples[idx] for idx in indices]
        items = sorted(items, key=lambda item: int(item.get("frame_order", 0)))
        first = items[0]
        rgb = _load_rgb(first["rgb_path"], self.image_size)
        if self.include_vessel_mask and first["mask_path"] is not None:
            global_mask = _load_gray(first["mask_path"], self.image_size, "mask")
        else:
            global_mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        mask = self._load_input_mask(first, global_mask)
        if self.include_mtt and first["mtt_path"] is not None:
            mtt = _load_gray(first["mtt_path"], self.image_size, "image")
            if self.mask_mtt_with_vessel:
                mtt = mtt * global_mask
        else:
            mtt = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        if self.include_lsci and first["lsci_path"] is not None:
            lsci = _load_gray(first["lsci_path"], self.image_size, "image")
        else:
            lsci = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        targets = [_load_gray(item["target_path"], self.image_size, "image") for item in items]
        guidance_masks = []
        keyframe_masks = []
        keyframe_available = []
        for item in items:
            if self.include_vessel_mask and item["guidance_mask_path"] is not None:
                guidance_masks.append(_load_gray(item["guidance_mask_path"], self.image_size, "mask"))
            else:
                guidance_masks.append(global_mask)
            if self.include_vessel_mask and item["keyframe_mask_path"] is not None:
                keyframe_masks.append(_load_gray(item["keyframe_mask_path"], self.image_size, "mask"))
                keyframe_available.append(1.0)
            else:
                keyframe_masks.append(global_mask)
                keyframe_available.append(0.0)
        if self.previous_frames > 0:
            prev_frames = []
            for item in items:
                frames = [
                    _load_gray(path, self.image_size, "image")
                    if path is not None
                    else torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
                    for path in item["prev_target_paths"]
                ]
                prev_frames.append(torch.cat(frames, dim=0))
            prev_frames_tensor = torch.stack(prev_frames, dim=0)
        else:
            prev_frames_tensor = torch.zeros((len(items), 0, self.image_size, self.image_size), dtype=torch.float32)

        return {
            "rgb": rgb.unsqueeze(0).expand(len(items), -1, -1, -1).clone(),
            "mask": mask.unsqueeze(0).expand(len(items), -1, -1, -1).clone(),
            "global_mask": global_mask,
            "guidance_mask": torch.stack(guidance_masks, dim=0),
            "keyframe_mask": torch.stack(keyframe_masks, dim=0),
            "keyframe_mask_available": torch.tensor(keyframe_available, dtype=torch.float32).view(len(items), 1),
            "mtt": mtt.unsqueeze(0).expand(len(items), -1, -1, -1).clone(),
            "lsci": lsci.unsqueeze(0).expand(len(items), -1, -1, -1).clone(),
            "target": torch.stack(targets, dim=0),
            "time": torch.tensor([[item["time"]] for item in items], dtype=torch.float32),
            "prev_frames": prev_frames_tensor,
            "prev_times": torch.tensor([item["prev_times"] for item in items], dtype=torch.float32),
            "patch_id": str(first["patch_id"]),
            "sequence_id": sequence_id,
            "frame_step": int(first.get("frame_step", self.frame_step)),
            "frame_step_offset": int(first.get("frame_step_offset", 0)),
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        item = self.samples[index]
        rgb = _load_rgb(item["rgb_path"], self.image_size)
        target = _load_gray(item["target_path"], self.image_size, "image")
        if item["next_target_path"] is not None:
            next_target = _load_gray(item["next_target_path"], self.image_size, "image")
            has_next = torch.tensor([1.0], dtype=torch.float32)
        else:
            next_target = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
            has_next = torch.tensor([0.0], dtype=torch.float32)
        if self.previous_frames > 0:
            prev_frames = [
                _load_gray(path, self.image_size, "image")
                if path is not None
                else torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
                for path in item["prev_target_paths"]
            ]
            prev_frames = torch.cat(prev_frames, dim=0)
        else:
            prev_frames = torch.zeros((0, self.image_size, self.image_size), dtype=torch.float32)
        if self.include_vessel_mask and item["mask_path"] is not None:
            global_mask = _load_gray(item["mask_path"], self.image_size, "mask")
        else:
            global_mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        mask = self._load_input_mask(item, global_mask)
        if self.include_vessel_mask and item["guidance_mask_path"] is not None:
            guidance_mask = _load_gray(item["guidance_mask_path"], self.image_size, "mask")
        else:
            guidance_mask = global_mask
        if self.include_vessel_mask and item["keyframe_mask_path"] is not None:
            keyframe_mask = _load_gray(item["keyframe_mask_path"], self.image_size, "mask")
            keyframe_mask_available = torch.tensor([1.0], dtype=torch.float32)
        else:
            keyframe_mask = global_mask
            keyframe_mask_available = torch.tensor([0.0], dtype=torch.float32)
        if self.include_mtt and item["mtt_path"] is not None:
            mtt = _load_gray(item["mtt_path"], self.image_size, "image")
            if self.mask_mtt_with_vessel:
                mtt = mtt * global_mask
        else:
            mtt = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        if self.include_lsci and item["lsci_path"] is not None:
            lsci = _load_gray(item["lsci_path"], self.image_size, "image")
        else:
            lsci = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)
        return {
            "rgb": rgb,
            "mask": mask,
            "global_mask": global_mask,
            "guidance_mask": guidance_mask,
            "keyframe_mask": keyframe_mask,
            "keyframe_mask_available": keyframe_mask_available,
            "mtt": mtt,
            "lsci": lsci,
            "target": target,
            "next_target": next_target,
            "has_next": has_next,
            "prev_frames": prev_frames,
            "prev_times": torch.tensor(item["prev_times"], dtype=torch.float32),
            "time": torch.tensor([item["time"]], dtype=torch.float32),
            "next_time": torch.tensor([item["next_time"]], dtype=torch.float32),
            "patch_id": item["patch_id"],
            "sequence_id": item.get("sequence_id", item["patch_id"]),
            "subject": item["subject"],
            "frame_order": item["frame_order"],
            "frame_step": item.get("frame_step", self.frame_step),
            "frame_step_offset": item.get("frame_step_offset", 0),
            "frame_index": item["frame_index"],
        }
