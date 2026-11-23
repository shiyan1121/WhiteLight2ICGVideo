"""
Summarize recomputed image metrics for every vessel-mask inference mode.

For one experiment, this script automatically discovers inference modes such as
``best_global``, ``best_rgb2vessel_mask`` and ``best_rgb2vessel_prob``. It then
recomputes the following metrics from saved PNG files:

    PSNR, SSIM, LPIPS, FVD,
    Vessel PSNR, Vessel SSIM, Vessel LPIPS, Vessel FVD

The aggregation follows ``summarize_recomputed_image_metrics_table.py``:
metrics are first averaged over frames within each case, and then reported as
mean +/- sample standard deviation over cases. Three tables are written:

    validation.csv/.json/.md
    test.csv/.json/.md
    overall.csv/.json/.md

The split is read from ``subject_split_summary.json`` in the experiment
directory. Prediction frames are always read from the current case directory
(``pred/*.png``), rather than trusting possibly stale prediction paths in
``summary.json``. The script does not perform time alignment; it evaluates
prediction frames against same-index GT frames, matching the original
script's non-aligned rows.

Examples (run from the project root):

    python notebook/summarize_experiment_metrics_by_split.py \
        --experiment 261010-ablation_261009_no_convgru

    python notebook/summarize_experiment_metrics_by_split.py \
        --experiment E:/.../experiments/261010-ablation_261009_no_convgru \
        --list-only

To supplement previously evaluated inference modes without loading LPIPS/FVD:

    python notebook/summarize_experiment_metrics_by_split.py \
        --experiment 261012-ablation_261009_raw_time --supplement-only

This discovers all inference modes, computes whole-image L1/MSE from local PNGs,
reuses full-sequence PSNR/SSIM/LPIPS/FVD from saved case tables (or the case
summary when absent), and writes validation/test/overall and subject tables to
recomputed_full_image_metrics_by_split. metric_sources.json records provenance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
NOTEBOOK_DIR = PROJECT_ROOT / "notebook"
if str(NOTEBOOK_DIR) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_DIR))

# These heavy imports are deliberately lazy. In particular, --help and
# --list-only should work without initializing the PyTorch/OpenMP runtime.
torch = None
build_fvd_metric = None
build_lpips_model = None
calculate_fvd_frames = None
calculate_lpips_frames = None
calculate_ssim_function = None
img_psnr = None
DEVICE = None
FVD_MODEL = "i3d"
FVD_CKPT_PATH = PROJECT_ROOT / "i3d_pretrained_400.pt"

SEQUENCE_METRICS = [
    ("mean_psnr", "PSNR"),
    ("mean_ssim", "SSIM"),
    ("mean_lpips", "LPIPS"),
    ("fvd", "FVD"),
    ("global_vessel_mean_psnr", "Vessel PSNR"),
    ("global_vessel_mean_ssim", "Vessel SSIM"),
    ("global_vessel_mean_lpips", "Vessel LPIPS"),
    ("global_vessel_fvd", "Vessel FVD"),
]

FULL_IMAGE_METRICS = [
    ("mean_l1", "L1"),
    ("mean_mse", "MSE"),
    ("mean_psnr", "PSNR"),
    ("mean_ssim", "SSIM"),
    ("mean_lpips", "LPIPS"),
    ("fvd", "FVD"),
]


def finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    return float(arr.mean()), std


def format_mean_std(value: tuple[float, float], decimals: int) -> str:
    mean, std = value
    if math.isnan(mean):
        return "-"
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    # Some older inference summaries contain a relative path copied from a
    # different experiment. Prefer the files physically present in the current
    # case directory when the path identifies a standard inference subfolder.
    for folder_name in ("pred", "aux_mask"):
        if folder_name in path.parts:
            local_candidate = base_dir / folder_name / path.name
            if local_candidate.exists():
                return local_candidate
    # Inference summaries usually store paths relative to the project root.
    project_candidate = PROJECT_ROOT / path
    if project_candidate.exists():
        return project_candidate
    case_candidate = base_dir / path
    return case_candidate


def find_case_dirs(mode_dir: Path) -> list[Path]:
    return sorted(
        p.parent
        for p in mode_dir.rglob("summary.json")
        if p.parent.is_dir()
    )


def local_frame_paths(case_dir: Path, folder_name: str) -> list[Path]:
    folder = case_dir / folder_name
    if not folder.is_dir():
        raise FileNotFoundError(f"Missing required folder: {folder}")
    paths = list(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No PNG frames found in {folder}")

    def frame_number(path: Path) -> tuple[int, str]:
        try:
            return int(path.stem), path.name
        except ValueError:
            return 10**9, path.name

    return sorted(paths, key=frame_number)


def discover_modes(inference_dir: Path, excluded: set[str]) -> list[Path]:
    modes = []
    for child in sorted(inference_dir.iterdir()):
        if not child.is_dir() or child.name in excluded:
            continue
        if any(child.rglob("summary.json")):
            modes.append(child)
    return modes


def load_gray01(
    path: str | Path,
    size: tuple[int, int] | None = None,
    nearest: bool = False,
) -> np.ndarray:
    image = Image.open(path).convert("L")
    if size is not None and image.size != size:
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        image = image.resize(size, resample)
    return np.asarray(image, dtype=np.float32) / 255.0


def as_tensor_chw(frame: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(frame.astype(np.float32))[None]


def psnr_value(pred: np.ndarray, gt: np.ndarray) -> float:
    if img_psnr is not None:
        return float(img_psnr(pred, gt))
    mse = float(np.mean((pred - gt) ** 2))
    if mse < 1e-10:
        return 100.0
    return float(20.0 * math.log10(1.0 / math.sqrt(mse)))


def ssim_value(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(calculate_ssim_function(pred[None], gt[None]))


def pair_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray | None,
) -> dict[str, float]:
    diff = pred - gt
    abs_err = np.abs(diff)
    sq_err = diff**2
    if mask is None:
        return {
            "l1": float(abs_err.mean()),
            "mse": float(sq_err.mean()),
            "psnr": psnr_value(pred, gt),
            "ssim": ssim_value(pred, gt),
        }
    mask_bool = mask > 0.5
    denom = max(1, int(mask_bool.sum()))
    vessel_pred = pred * mask_bool
    vessel_gt = gt * mask_bool
    return {
        "l1": float(abs_err.mean()),
        "mse": float(sq_err.mean()),
        "psnr": psnr_value(pred, gt),
        "ssim": ssim_value(pred, gt),
        "global_vessel_l1": float(abs_err[mask_bool].sum() / denom),
        "global_vessel_mse": float(sq_err[mask_bool].sum() / denom),
        "global_vessel_psnr": psnr_value(vessel_pred, vessel_gt),
        "global_vessel_ssim": ssim_value(vessel_pred, vessel_gt),
    }


def load_case_data(case_dir: Path, include_vessel_metrics: bool = True) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray | None]:
    summary_path = case_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    frames = [
        frame
        for frame in summary.get("frames", [])
        if frame.get("pred_path") and frame.get("target_path")
    ]
    if not frames:
        raise ValueError(f"No usable frames in {summary_path}")

    # Do not use summary['pred_path'] here. Older inference runs may contain
    # paths copied from another experiment. The current case directory is the
    # source of truth for prediction frames.
    pred_paths = local_frame_paths(case_dir, "pred")

    frames = sorted(frames, key=lambda item: int(item.get("frame", 0)))
    target_paths = [resolve_path(frame["target_path"], case_dir) for frame in frames]
    if len(pred_paths) != len(target_paths):
        raise ValueError(
            f"pred/GT frame count mismatch in {case_dir}: "
            f"{len(pred_paths)} vs {len(target_paths)}"
        )
    if not all(path.exists() for path in target_paths):
        missing = next(path for path in target_paths if not path.exists())
        raise FileNotFoundError(f"Missing frame file: {missing}")

    first_pred = Image.open(pred_paths[0]).convert("L")
    pred_frames = [load_gray01(path) for path in pred_paths]
    gt_frames = [load_gray01(path, size=first_pred.size) for path in target_paths]

    if not include_vessel_metrics:
        return pred_frames, gt_frames, None

    mask_value = summary.get("mask_path")
    if not mask_value:
        raise ValueError(f"No evaluation mask_path in {summary_path}")
    mask_path = resolve_path(mask_value, case_dir)
    if not mask_path.exists():
        raise FileNotFoundError(f"Missing evaluation mask: {mask_path}")
    mask = load_gray01(mask_path, size=first_pred.size, nearest=True)
    return pred_frames, gt_frames, mask


def calculate_sequence_metrics(
    pred_frames: list[np.ndarray],
    gt_frames: list[np.ndarray],
    mask: np.ndarray | None,
    lpips_model,
    fvd_metric,
    include_vessel_metrics: bool = True,
) -> dict[str, float | None]:
    frame_metrics = [pair_metrics(pred, gt, mask if include_vessel_metrics else None) for pred, gt in zip(pred_frames, gt_frames)]
    out: dict[str, float | None] = {
        "mean_l1": float(np.mean([item["l1"] for item in frame_metrics])),
        "mean_mse": float(np.mean([item["mse"] for item in frame_metrics])),
        "mean_psnr": float(np.mean([item["psnr"] for item in frame_metrics])),
        "mean_ssim": float(np.mean([item["ssim"] for item in frame_metrics])),
    }

    if include_vessel_metrics:
        out["global_vessel_mean_psnr"] = float(
            np.mean([item["global_vessel_psnr"] for item in frame_metrics])
        )
        out["global_vessel_mean_ssim"] = float(
            np.mean([item["global_vessel_ssim"] for item in frame_metrics])
        )

    pred_tensors = [as_tensor_chw(frame) for frame in pred_frames]
    gt_tensors = [as_tensor_chw(frame) for frame in gt_frames]
    if include_vessel_metrics:
        mask_tensor = as_tensor_chw((mask > 0.5).astype(np.float32))
        pred_vessel_tensors = [frame * mask_tensor for frame in pred_tensors]
        gt_vessel_tensors = [frame * mask_tensor for frame in gt_tensors]

    if lpips_model is not None:
        out["mean_lpips"] = float(
            calculate_lpips_frames(pred_tensors, gt_tensors, DEVICE, lpips_model)
        )
        if include_vessel_metrics:
            out["global_vessel_mean_lpips"] = float(
                calculate_lpips_frames(
                    pred_vessel_tensors, gt_vessel_tensors, DEVICE, lpips_model
                )
            )
    else:
        out["mean_lpips"] = None
        if include_vessel_metrics:
            out["global_vessel_mean_lpips"] = None

    if fvd_metric is not None:
        out["fvd"] = float(calculate_fvd_frames(pred_tensors, gt_tensors, fvd_metric))
        if include_vessel_metrics:
            out["global_vessel_fvd"] = float(
                calculate_fvd_frames(pred_vessel_tensors, gt_vessel_tensors, fvd_metric)
            )
    else:
        out["fvd"] = None
        if include_vessel_metrics:
            out["global_vessel_fvd"] = None
    return out


def build_metric_models():
    load_metric_backend()
    lpips_model = build_lpips_model(DEVICE)
    ckpt_path = str(FVD_CKPT_PATH) if FVD_CKPT_PATH.exists() else None
    fvd_metric = build_fvd_metric(DEVICE, FVD_MODEL, ckpt_path)
    return lpips_model, fvd_metric


def load_metric_backend() -> None:
    global torch
    global build_fvd_metric, build_lpips_model
    global calculate_fvd_frames, calculate_lpips_frames
    global calculate_ssim_function, img_psnr, DEVICE
    if torch is not None:
        return
    import torch as torch_module

    from eval import (
        build_fvd_metric as _build_fvd_metric,
        build_lpips_model as _build_lpips_model,
        calculate_fvd_frames as _calculate_fvd_frames,
        calculate_lpips_frames as _calculate_lpips_frames,
        calculate_ssim_function as _calculate_ssim_function,
        img_psnr as _img_psnr,
    )

    torch = torch_module
    build_fvd_metric = _build_fvd_metric
    build_lpips_model = _build_lpips_model
    calculate_fvd_frames = _calculate_fvd_frames
    calculate_lpips_frames = _calculate_lpips_frames
    calculate_ssim_function = _calculate_ssim_function
    img_psnr = _img_psnr
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def subject_from_case_id(case_id: str) -> str:
    return case_id.rsplit("_", 1)[0] if "_" in case_id else case_id


def load_split_subjects(experiment_dir: Path, split_file: Path | None) -> dict[str, set[str]]:
    path = split_file or (experiment_dir / "subject_split_summary.json")
    if not path.is_absolute():
        path = experiment_dir / path
    if not path.exists():
        raise FileNotFoundError(
            f"Cannot find split file: {path}. Use --split-file to specify one."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    val_subjects = set(data.get("val", {}).get("subjects", {}).keys())
    test_value = data.get("test_subjects", data.get("test", {}).get("subjects", {}))
    if isinstance(test_value, dict):
        test_subjects = set(test_value.keys())
    else:
        test_subjects = set(test_value or [])
    if not val_subjects or not test_subjects:
        raise ValueError(
            f"Split file must contain val subjects and test subjects: {path}"
        )
    return {"validation": val_subjects, "test": test_subjects}


def split_for_case(case_id: str, split_subjects: dict[str, set[str]]) -> str | None:
    subject = subject_from_case_id(case_id)
    for split_name, subjects in split_subjects.items():
        if subject in subjects:
            return split_name
    return None


def metric_row(
    mode_name: str,
    mode_dir: Path,
    split_name: str,
    split_subjects: dict[str, set[str]],
    lpips_model,
    fvd_metric,
    case_records: list[dict[str, Any]],
    max_frames: int | None = None,
    include_vessel_metrics: bool = True,
) -> dict[str, Any]:
    per_case: list[dict[str, Any]] = []
    cached_records = {(str(item.get("vessel_mask")), str(item.get("case_id"))): item for item in case_records}
    for case_dir in find_case_dirs(mode_dir):
        case_id = case_dir.name
        case_split = split_for_case(case_id, split_subjects)
        if split_name != "overall" and case_split != split_name:
            continue
        if split_name == "overall" and case_split is None:
            continue
        cached = cached_records.get((mode_name, case_id))
        if cached is not None:
            if cached.get("status") == "ok":
                per_case.append(cached)
            continue
        try:
            print(f"    {case_id}", flush=True)
            pred_frames, gt_frames, mask = load_case_data(case_dir, include_vessel_metrics=include_vessel_metrics)
            if max_frames is not None:
                pred_frames = pred_frames[:max_frames]
                gt_frames = gt_frames[:max_frames]
                if not pred_frames:
                    raise ValueError(f"No frames remain after applying --max-frames {max_frames}")
            metrics = calculate_sequence_metrics(
                pred_frames,
                gt_frames,
                mask,
                lpips_model,
                fvd_metric,
                include_vessel_metrics=include_vessel_metrics,
            )
        except Exception as exc:  # keep other cases running and record the reason
            case_records.append(
                {
                    "split": split_name,
                    "vessel_mask": mode_name,
                    "case_id": case_id,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue

        record = {
            "split": case_split if split_name == "overall" else split_name,
            "vessel_mask": mode_name,
            "case_id": case_id,
            "status": "ok",
            **{key: finite_float(value) for key, value in metrics.items()},
        }
        case_records.append(record)
        per_case.append(record)

    row: dict[str, Any] = {
        "Input vessel mask": mode_name,
        "n": len(per_case),
    }
    for key, display_name in SEQUENCE_METRICS:
        values = [item.get(key) for item in per_case]
        clean_values = [value for value in values if value is not None]
        row[display_name] = mean_std(clean_values)
    return row


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["Input vessel mask", "n", *[name for _key, name in SEQUENCE_METRICS]]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow(
                [
                    row["Input vessel mask"],
                    row["n"],
                    *[format_mean_std(row[name], 6) for _key, name in SEQUENCE_METRICS],
                ]
            )


def write_json(rows: list[dict[str, Any]], path: Path) -> None:
    serializable = []
    for row in rows:
        output = {"Input vessel mask": row["Input vessel mask"], "n": row["n"]}
        for _key, name in SEQUENCE_METRICS:
            mean, std = row[name]
            output[name] = {
                "mean": None if math.isnan(mean) else mean,
                "std": None if math.isnan(std) else std,
            }
        serializable.append(output)
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = ["Input vessel mask", "n", *[name for _key, name in SEQUENCE_METRICS]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|" + "---:|" * len(SEQUENCE_METRICS),
    ]
    decimals = {
        "L1": 4,
        "MSE": 4,
        "PSNR": 2,
        "SSIM": 3,
        "LPIPS": 3,
        "FVD": 2,
        "Vessel PSNR": 2,
        "Vessel SSIM": 3,
        "Vessel LPIPS": 3,
        "Vessel FVD": 2,
    }
    for row in rows:
        cells = [row["Input vessel mask"], str(row["n"])]
        for _key, name in SEQUENCE_METRICS:
            cells.append(format_mean_std(row[name], decimals[name]))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_case_records(records: list[dict[str, Any]], path: Path) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = ["split", "vessel_mask", "case_id", "status", "error"] + [
        key for key, _name in SEQUENCE_METRICS
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def deduplicate_case_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove validation/test rows duplicated by an additional overall pass."""
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            str(record.get("split", "")),
            str(record.get("vessel_mask", "")),
            str(record.get("case_id", "")),
        )
        unique[key] = record
    return list(unique.values())


def build_subject_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for record in deduplicate_case_records(records):
        if record.get("status") != "ok":
            continue
        split = str(record.get("split", ""))
        mode = str(record.get("vessel_mask", ""))
        subject = subject_from_case_id(str(record.get("case_id", "")))
        groups.setdefault((split, subject, mode), []).append(record)

    rows: list[dict[str, Any]] = []
    for (split, subject, mode), cases in sorted(groups.items()):
        row: dict[str, Any] = {
            "split": split,
            "subject": subject,
            "Input vessel mask": mode,
            "n": len(cases),
        }
        for key, display_name in SEQUENCE_METRICS:
            values = [finite_float(case.get(key)) for case in cases]
            row[display_name] = mean_std([value for value in values if value is not None])
        rows.append(row)
    return rows


def write_subject_tables(records: list[dict[str, Any]], out_dir: Path) -> None:
    rows = build_subject_rows(records)
    headers = ["split", "subject", "Input vessel mask", "n", *[name for _, name in SEQUENCE_METRICS]]

    with (out_dir / "subject_metrics.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([
                row["split"],
                row["subject"],
                row["Input vessel mask"],
                row["n"],
                *[format_mean_std(row[name], 6) for _, name in SEQUENCE_METRICS],
            ])

    serializable = []
    for row in rows:
        item: dict[str, Any] = {
            "split": row["split"],
            "subject": row["subject"],
            "Input vessel mask": row["Input vessel mask"],
            "n": row["n"],
        }
        for _, name in SEQUENCE_METRICS:
            mean, std = row[name]
            item[name] = {"mean": mean, "std": std}
        serializable.append(item)
    (out_dir / "subject_metrics.json").write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    decimals = {"L1": 4, "MSE": 4, "PSNR": 2, "SSIM": 3, "LPIPS": 3, "FVD": 2,
                "Vessel PSNR": 2, "Vessel SSIM": 3, "Vessel LPIPS": 3, "Vessel FVD": 2}
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---|---|---:|" + "---:|" * len(SEQUENCE_METRICS),
    ]
    for row in rows:
        cells = [row["split"], row["subject"], row["Input vessel mask"], str(row["n"])]
        cells.extend(format_mean_std(row[name], decimals.get(name, 6)) for _, name in SEQUENCE_METRICS)
        lines.append("| " + " | ".join(cells) + " |")
    (out_dir / "subject_metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment",
        required=True,
        type=Path,
        help="Experiment name under experiments/ or an explicit experiment directory.",
    )
    parser.add_argument(
        "--experiments-root",
        type=Path,
        default=PROJECT_ROOT / "experiments",
        help="Root directory used when --experiment is a name.",
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=None,
        help="Override the inference directory; default is <experiment>/inference.",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=None,
        help="Split JSON; default is <experiment>/subject_split_summary.json.",
    )
    parser.add_argument(
        "--mode",
        action="append",
        default=None,
        help="Only process this inference mode. Repeat for multiple modes.",
    )
    parser.add_argument(
        "--exclude-mode",
        action="append",
        default=[],
        help="Exclude an automatically discovered inference mode.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; default is <experiment>/recomputed_image_metrics_by_split.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Only evaluate the first N frames of each case (for example, 26 for wash-in).",
    )
    parser.add_argument(
        "--summary-splits",
        nargs="+",
        choices=["validation", "test", "overall"],
        default=["validation", "test", "overall"],
        help="Metric tables to compute. Defaults to validation, test, and overall.",
    )
    parser.add_argument(
        "--supplement-only",
        action="store_true",
        help="Reuse saved whole-image PSNR/SSIM/LPIPS/FVD, compute only L1/MSE, and write split/subject tables without loading metric models.",
    )
    parser.add_argument(
        "--full-image-only",
        action="store_true",
        help="Only report whole-image L1, MSE, PSNR, SSIM, LPIPS, and FVD.",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="List discovered modes and split counts without loading LPIPS/FVD models.",
    )
    return parser.parse_args()


def resolve_experiment(value: Path, experiments_root: Path) -> Path:
    if value.exists():
        return value.resolve()
    candidate = experiments_root / value
    if candidate.exists():
        return candidate.resolve()
    raise FileNotFoundError(f"Experiment directory not found: {value} or {candidate}")


def supplement_full_image_metrics(experiment_dir, modes, split_subjects, out_dir, summary_splits):
    """Reuse full-sequence saved metrics and calculate only pixel errors."""
    cache = {}
    sources = {}
    primary = experiment_dir / "recomputed_image_metrics_by_split" / "case_metrics.csv"
    other_paths = sorted(experiment_dir.rglob("case_metrics.csv"))
    paths = ([primary] if primary.exists() else []) + [p for p in other_paths if p != primary]
    for path in paths:
        relative = str(path.relative_to(experiment_dir)).lower()
        if any(token in relative for token in ("wash", "align", "vessel_pixel", "full_image", "match")):
            continue
        with path.open(encoding="utf-8-sig", newline="") as handle:
            for saved in csv.DictReader(handle):
                if saved.get("status") != "ok":
                    continue
                key = (saved.get("vessel_mask"), saved.get("case_id"))
                if key not in cache:
                    cache[key] = saved
                    sources[key] = str(path)

    records = []
    audit = []
    for mode_dir in modes:
        cases = find_case_dirs(mode_dir)
        print(f"Supplementing {mode_dir.name}: {len(cases)} cases", flush=True)
        for case_dir in cases:
            split = split_for_case(case_dir.name, split_subjects)
            if split is None:
                continue
            record = {"split": split, "vessel_mask": mode_dir.name, "case_id": case_dir.name,
                      "status": "ok", "error": ""}
            try:
                summary = json.loads((case_dir / "summary.json").read_text(encoding="utf-8"))
                saved = cache.get((mode_dir.name, case_dir.name), {})
                saved_summary = summary.get("metrics", {})
                metric_sources = {}
                for key in ("mean_psnr", "mean_ssim", "mean_lpips", "fvd"):
                    value = finite_float(saved.get(key))
                    if value is not None:
                        source = sources[(mode_dir.name, case_dir.name)]
                    else:
                        value = finite_float(saved_summary.get(key))
                        source = str(case_dir / "summary.json")
                    record[key] = value
                    metric_sources[key] = source if value is not None else "unavailable"
                pred_frames, gt_frames, _ = load_case_data(case_dir, include_vessel_metrics=False)
                record["mean_l1"] = float(np.mean([np.abs(p - g).mean() for p, g in zip(pred_frames, gt_frames)]))
                record["mean_mse"] = float(np.mean([((p - g) ** 2).mean() for p, g in zip(pred_frames, gt_frames)]))
                audit.append({"mode": mode_dir.name, "case_id": case_dir.name,
                              "n_frames": len(pred_frames), "metric_sources": metric_sources})
            except Exception as exc:
                record.update(status="failed", error=str(exc))
                print(f"  FAILED {case_dir.name}: {exc}", flush=True)
            records.append(record)
        write_case_records(records, out_dir / "case_metrics.csv")
        print(f"Completed {mode_dir.name}", flush=True)

    for split_name in dict.fromkeys(summary_splits):
        rows = []
        for mode_dir in modes:
            cases = [r for r in records if r["status"] == "ok" and r["vessel_mask"] == mode_dir.name
                     and (split_name == "overall" or r["split"] == split_name)]
            row = {"Input vessel mask": mode_dir.name, "n": len(cases)}
            for key, name in SEQUENCE_METRICS:
                values = [finite_float(r.get(key)) for r in cases]
                row[name] = mean_std([value for value in values if value is not None])
            rows.append(row)
        write_csv(rows, out_dir / f"{split_name}.csv")
        write_json(rows, out_dir / f"{split_name}.json")
        (out_dir / f"{split_name}.md").write_text(markdown_table(rows) + "\n", encoding="utf-8")
    write_subject_tables(records, out_dir)
    (out_dir / "metric_sources.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_dir}; cases={len(records)}; failed={sum(r['status'] != 'ok' for r in records)}", flush=True)


def main() -> None:
    global SEQUENCE_METRICS
    args = parse_args()
    if args.supplement_only:
        args.full_image_only = True
        if args.max_frames is not None:
            raise ValueError("--supplement-only reuses full-sequence metrics and cannot be combined with --max-frames")
    if args.full_image_only:
        SEQUENCE_METRICS = FULL_IMAGE_METRICS
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be a positive integer")
    experiment_dir = resolve_experiment(args.experiment, args.experiments_root)
    inference_dir = (args.inference_dir or experiment_dir / "inference").resolve()
    if not inference_dir.exists():
        raise FileNotFoundError(f"Inference directory not found: {inference_dir}")

    split_subjects = load_split_subjects(experiment_dir, args.split_file)
    excluded = set(args.exclude_mode)
    if args.mode:
        modes = [inference_dir / name for name in args.mode]
        missing = [path for path in modes if not path.is_dir()]
        if missing:
            raise FileNotFoundError(f"Inference mode directory not found: {missing}")
    else:
        modes = discover_modes(inference_dir, excluded)
    if not modes:
        raise RuntimeError(f"No inference mode containing summary.json found in {inference_dir}")

    print(f"Experiment: {experiment_dir}")
    print(f"Inference:  {inference_dir}")
    print(f"Validation subjects: {len(split_subjects['validation'])}")
    print(f"Test subjects:       {len(split_subjects['test'])}")
    for mode_dir in modes:
        case_count = len(find_case_dirs(mode_dir))
        print(f"  {mode_dir.name}: {case_count} cases")

    if args.list_only:
        return

    if args.supplement_only:
        out_dir = (args.out_dir or experiment_dir / "recomputed_full_image_metrics_by_split").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        supplement_full_image_metrics(experiment_dir, modes, split_subjects, out_dir, args.summary_splits)
        return

    lpips_model, fvd_metric = build_metric_models()
    print(f"Using device: {DEVICE}")
    default_out_name = (
        "recomputed_full_image_metrics_by_split"
        if args.full_image_only
        else "recomputed_image_metrics_by_split"
    )
    out_dir = (args.out_dir or experiment_dir / default_out_name).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    case_records: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, Any]]] = {
        split_name: [] for split_name in dict.fromkeys(args.summary_splits)
    }
    for split_name in split_rows:
        print(f"\nComputing {split_name} metrics...")
        for mode_dir in modes:
            print(f"  {mode_dir.name}")
            split_rows[split_name].append(
                metric_row(
                    mode_dir.name,
                    mode_dir,
                    split_name,
                    split_subjects,
                    lpips_model,
                    fvd_metric,
                    case_records,
                    max_frames=args.max_frames,
                    include_vessel_metrics=not args.full_image_only,
                )
            )

    for split_name, rows in split_rows.items():
        md = markdown_table(rows)
        (out_dir / f"{split_name}.md").write_text(md + "\n", encoding="utf-8")
        write_csv(rows, out_dir / f"{split_name}.csv")
        write_json(rows, out_dir / f"{split_name}.json")
        print(f"\n{split_name.capitalize()} table:\n{md}")

    case_records = deduplicate_case_records(case_records)
    write_case_records(case_records, out_dir / "case_metrics.csv")
    write_subject_tables(case_records, out_dir)
    failed = [record for record in case_records if record.get("status") == "failed"]
    print(f"\nSaved tables to: {out_dir}")
    print(f"Case-level audit: {out_dir / 'case_metrics.csv'}")
    print(f"Subject-level table: {out_dir / 'subject_metrics.csv'}")
    if failed:
        print(f"Warning: {len(failed)} case evaluations failed; see case_metrics.csv.")


if __name__ == "__main__":
    main()
