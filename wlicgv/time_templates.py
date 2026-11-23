"""Frozen, split-level normalized time templates (no image/runtime dependencies)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def validate_time_template(values, expected_frames: int | None = None) -> list[float]:
    if not isinstance(values, (list, tuple)) or len(values) < 2:
        raise ValueError("A time template must contain at least two normalized timestamps.")
    times = [float(value) for value in values]
    if expected_frames is not None and len(times) != expected_frames:
        raise ValueError(f"Time template has {len(times)} frames; expected {expected_frames}. No interpolation is performed.")
    if not all(math.isfinite(t) and 0 <= t <= 1 for t in times):
        raise ValueError("Template timestamps must be finite and within [0, 1].")
    if abs(times[0]) > 1e-8 or abs(times[-1] - 1) > 1e-8:
        raise ValueError("Normalized time template must start at 0 and end at 1.")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise ValueError("Template timestamps must be strictly increasing.")
    return times


def configure_time_template(cfg: dict, template_file=None, template_split=None, sample_mode=None) -> None:
    """Apply CLI overrides, or validate the embedded snapshot without rereading its source."""
    data = cfg.setdefault("data", {})
    if template_split is not None and template_file is None:
        raise ValueError("--time_template_split requires --time_template_file.")
    if template_file is not None:
        path = Path(template_file).expanduser().resolve()
        raw = path.read_bytes()
        summary = json.loads(raw.decode("utf-8-sig"))
        split = template_split or "train"
        if split not in {"train", "all"}:
            raise ValueError("Only train or all mean-time templates are supported.")
        record = summary.get("splits", {}).get(split)
        if not isinstance(record, dict) or not record.get("num_patches", 0):
            raise ValueError(f"No usable mean timestamps for split {split!r} in {path}.")
        values = validate_time_template(record.get("mean_normalized_time"), int(summary["num_frames"]))
        data["time_reference"] = "shared_template"
        data["time_template"] = values
        data["time_template_provenance"] = {
            "file": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "split": split, "num_patches": record["num_patches"],
            "num_frames": len(values), "sample_mode": summary.get("sample_mode"),
            "normalization": summary.get("normalization"),
            "data_root": summary.get("data_root"), "split_config": summary.get("split_config"),
            "uses_held_out_statistics": split == "all",
        }
    if data.get("time_reference") == "shared_template":
        data["time_template"] = validate_time_template(data.get("time_template"))
        provenance = data.get("time_template_provenance", {})
        expected_mode = provenance.get("sample_mode")
        actual_mode = sample_mode or data.get("sample_mode", "Intensity_step_sample")
        if expected_mode and expected_mode != actual_mode:
            raise ValueError(f"Template sample_mode={expected_mode!r}, but data use {actual_mode!r}.")
