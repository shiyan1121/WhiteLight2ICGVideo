"""Generate ICG frames from RGB and an RGB-derived vessel mask; no GT is required."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import yaml
from wlicgv.data import _load_gray, _load_rgb
from wlicgv.models import build_model


def load_checkpoint(path, device="cpu"):
    # Only load trusted files: historical checkpoints embed Python configuration objects.
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False
    return torch.load(path, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--mask-kind", choices=["binary", "probability"], default="binary")
    parser.add_argument("--config", help="Optional matching config; otherwise read from checkpoint.")
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if min(args.frames, args.threads) < 1:
        parser.error("--frames and --threads must be positive.")
    for filename in (args.checkpoint, args.rgb, args.mask):
        if not Path(filename).is_file():
            parser.error(f"File not found: {filename}")
    output = Path(args.out_dir)
    if output.exists() and any(output.iterdir()):
        parser.error("Choose a new empty output directory to preserve previous results.")
    torch.set_num_threads(args.threads)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    checkpoint = load_checkpoint(args.checkpoint)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")) if args.config else checkpoint.get("config")
    if not cfg or "model" not in cfg:
        parser.error("Checkpoint has no config; supply --config.")
    m = cfg["model"]
    if m.get("include_time", False) or int(m.get("previous_frames", 0)):
        parser.error("Use infer_patch.py for time-conditioned or previous-frame-feedback checkpoints.")
    if m.get("include_mtt", False) or m.get("include_lsci", False):
        parser.error("This entry point accepts RGB and vessel masks only.")
    model = build_model(m)
    model.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint
    model.to(device).eval()
    size = int(cfg.get("data", {}).get("image_size", 256))
    rgb = _load_rgb(Path(args.rgb), size)[None].to(device)
    mask = _load_gray(Path(args.mask), size, kind="mask" if args.mask_kind == "binary" else "image")[None].to(device)
    output.mkdir(parents=True, exist_ok=True)
    hidden = None
    with torch.no_grad():
        for i in range(args.frames):
            # The no-time network ignores this placeholder; it is not a physical timestamp.
            prediction, hidden = model.forward_step(rgb, torch.zeros(1, 1, device=device), mask, hidden)
            for channel, folder in enumerate(("pred", "aux_mask")[:prediction.shape[1]]):
                destination = output / folder
                destination.mkdir(exist_ok=True)
                pixels = (prediction[0, channel].cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
                Image.fromarray(pixels).save(destination / f"frame_{i + 1:03d}.png")
            print(f"Generated {i + 1}/{args.frames}", flush=True)
    record = {"checkpoint": str(Path(args.checkpoint).resolve()), "rgb": str(Path(args.rgb).resolve()),
              "mask": str(Path(args.mask).resolve()), "mask_kind": args.mask_kind, "frames": args.frames,
              "image_size": size, "device": str(device), "include_time": False, "uses_target_icg": False,
              "note": "8-bit preview PNGs; output positions are rollout steps, not physical timestamps."}
    (output / "prediction.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved to: {output.resolve()}")


if __name__ == "__main__":
    main()
