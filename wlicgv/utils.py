from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def subject_from_patch_dir(path: str | Path) -> str:
    name = Path(path).name
    if "_" not in name:
        return name
    return name.rsplit("_", 1)[0]


def list_subjects_from_patch_dirs(data_root: str | Path) -> list[str]:
    subjects = set()
    for patch_dir in Path(data_root).iterdir():
        if patch_dir.is_dir() and "_" in patch_dir.name:
            subjects.add(subject_from_patch_dir(patch_dir))
    return sorted(subjects)


def make_subject_split(
    subjects: Sequence[str],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> dict[str, list[str]]:
    subjects = list(subjects)
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(round(n * test_fraction))) if n >= 3 and test_fraction > 0 else 0
    n_val = max(1, int(round(n * val_fraction))) if n >= 3 and val_fraction > 0 else 0
    if n_val + n_test >= n:
        n_val = max(1, n // 5)
        n_test = max(0, n // 5)
    test = sorted(subjects[:n_test])
    val = sorted(subjects[n_test:n_test + n_val])
    train = sorted(subjects[n_test + n_val:])
    return {"train": train, "val": val, "test": test}


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = torch.mean((pred - target) ** 2).item()
    return 20.0 * math.log10(1.0 / math.sqrt(max(mse, eps)))


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    x = x.detach().float().cpu().clamp(0, 1)
    if x.ndim == 3:
        if x.shape[0] == 1:
            arr = x[0].numpy()
            return (arr * 255).astype(np.uint8)
        arr = x.permute(1, 2, 0).numpy()
        return (arr * 255).astype(np.uint8)
    raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")


def _load_font(size: int) -> ImageFont.ImageFont:
    for font_path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def save_preview_panel(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    path: str | Path,
    title: str = "",
    font_size: int = 24,
    input_panels: Sequence[tuple[str, torch.Tensor]] | None = None,
) -> None:
    path = Path(path)
    if input_panels is None:
        input_panels = [("RGB", rgb), ("Vessel", mask)]
    labels = [label for label, _ in input_panels] + ["Pred", "Target", "AbsErr"]
    images = [Image.fromarray(tensor_to_uint8(tensor)).convert("RGB") for _, tensor in input_panels]
    images.extend(
        [
            Image.fromarray(tensor_to_uint8(pred)).convert("RGB"),
            Image.fromarray(tensor_to_uint8(target)).convert("RGB"),
            Image.fromarray(tensor_to_uint8(torch.abs(pred - target))).convert("RGB"),
        ]
    )

    w, h = images[0].size
    label_h = max(40, font_size + 18)
    title_h = label_h if title else 0
    canvas = Image.new("RGB", (w * len(images), h + label_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = _load_font(font_size)

    if title:
        draw.text((10, 8), title, fill=(0, 0, 0), font=font)
    y_label = title_h
    y_img = title_h + label_h
    for c, img in enumerate(images):
        x0 = c * w
        draw.text((x0 + 10, y_label + 8), labels[c], fill=(0, 0, 0), font=font)
        canvas.paste(img, (x0, y_img))

    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def save_preview_grid(
    rgb: torch.Tensor,
    mask: torch.Tensor,
    pred: torch.Tensor,
    target: torch.Tensor,
    path: str | Path,
    max_items: int = 4,
) -> None:
    path = Path(path)
    n = min(max_items, rgb.shape[0])
    panels = []
    labels = ["RGB", "Vessel", "Pred", "Target", "AbsErr"]
    for i in range(n):
        row = []
        rgb_img = Image.fromarray(tensor_to_uint8(rgb[i]))
        mask_img = Image.fromarray(tensor_to_uint8(mask[i])).convert("RGB")
        pred_img = Image.fromarray(tensor_to_uint8(pred[i])).convert("RGB")
        target_img = Image.fromarray(tensor_to_uint8(target[i])).convert("RGB")
        err_img = Image.fromarray(tensor_to_uint8(torch.abs(pred[i] - target[i]))).convert("RGB")
        row.extend([rgb_img, mask_img, pred_img, target_img, err_img])
        panels.append(row)

    w, h = panels[0][0].size
    label_h = 24
    canvas = Image.new("RGB", (w * len(labels), (h + label_h) * n), "white")
    draw = ImageDraw.Draw(canvas)
    for r, row in enumerate(panels):
        y0 = r * (h + label_h)
        for c, img in enumerate(row):
            x0 = c * w
            draw.text((x0 + 6, y0 + 4), labels[c], fill=(0, 0, 0))
            canvas.paste(img, (x0, y0 + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)
