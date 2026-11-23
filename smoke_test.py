"""Check forward, real training loss, backward, AdamW and checkpoint reload without clinical data."""
from __future__ import annotations
import argparse
import copy
from pathlib import Path
import tempfile
import torch
import yaml
from predict import load_checkpoint
from train import compute_sequence_loss, validate_training_scope
from wlicgv.models import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs/convgru_resunet.yaml"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(2026)
    cfg = copy.deepcopy(yaml.safe_load(Path(args.config).read_text(encoding="utf-8-sig")))
    cfg["model"]["base_channels"] = 4  # Reduced width only for this test, not the experiment.
    device = torch.device(args.device)
    validate_training_scope(cfg)
    model = build_model(cfg["model"]).to(device).train()
    batch = {"rgb": torch.rand(1, 3, 3, 32, 32, device=device),
             "mask": (torch.rand(1, 3, 1, 32, 32, device=device) > .5).float(),
             "target": torch.rand(1, 3, 1, 32, 32, device=device),
             "time": torch.tensor([[[0.0], [.5], [1.0]]], device=device),
             "keyframe_mask": (torch.rand(1, 3, 1, 32, 32, device=device) > .5).float(),
             "keyframe_mask_available": torch.ones(1, 3, device=device)}
    batch["guidance_mask"] = batch["mask"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(.5, .999), weight_decay=0)
    optimizer.zero_grad()
    loss, parts, pred, aux = compute_sequence_loss(model, batch, cfg, device)
    assert torch.isfinite(loss)
    assert pred.shape == aux.shape == (3, 1, 32, 32)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
    assert model.out.frame_gate.weight.grad is not None, "Auxiliary gate was disconnected"
    optimizer.step()
    model.eval()
    with torch.no_grad():
        before = model.forward_sequence(batch["rgb"], batch["time"], batch["mask"])
        rgb = batch["rgb"][:, :1].expand(-1, 32, -1, -1, -1)
        mask = batch["mask"][:, :1].expand(-1, 32, -1, -1, -1)
        rollout = model.forward_sequence(rgb, torch.zeros(1, 32, 1, device=device), mask)
        assert rollout.shape == (1, 32, 2, 32, 32) and torch.isfinite(rollout).all()
    with tempfile.TemporaryDirectory(prefix="convgru_resunet_smoke_") as temporary:
        path = Path(temporary) / "test.pt"
        torch.save({"model": model.state_dict(), "config": cfg}, path)
        restored = build_model(cfg["model"]).to(device).eval()
        restored.load_state_dict(load_checkpoint(path)["model"], strict=True)
        with torch.no_grad():
            after = restored.forward_sequence(batch["rgb"], batch["time"], batch["mask"])
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    print(f"PASS: loss={loss.item():.6f}; forward/backward/AdamW/checkpoint reload passed.")
    print(f"PASS: static-input 32-frame rollout shape={tuple(rollout.shape)}.")
    print(f"Auxiliary loss={parts.get('aux_vessel_loss', 0):.6f}; gate gradients finite.")


if __name__ == "__main__":
    main()
