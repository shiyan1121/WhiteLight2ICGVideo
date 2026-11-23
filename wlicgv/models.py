"""ConvGRU-ResUNet and its input, residual, recurrent and output modules."""
from __future__ import annotations

import math

import torch

from torch import nn

import torch.nn.functional as F

class TemporalHistoryAttention(nn.Module):
    """Per-pixel temporal attention over previous ICG frames.

    The module keeps the history tensor shape unchanged: B x T x H x W.
    Each spatial location attends only across its T previous-frame tokens.
    """

    def __init__(self, previous_frames: int, hidden_dim: int = 16, num_heads: int = 4, chunk_pixels: int = 65536) -> None:
        super().__init__()
        self.previous_frames = int(previous_frames)
        self.hidden_dim = int(hidden_dim)
        self.chunk_pixels = int(chunk_pixels)
        if self.previous_frames < 1:
            raise ValueError("TemporalHistoryAttention requires previous_frames >= 1")
        if self.hidden_dim % int(num_heads) != 0:
            raise ValueError("temporal attention hidden_dim must be divisible by num_heads")
        self.token = nn.Sequential(
            nn.Linear(3, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.attn = nn.MultiheadAttention(self.hidden_dim, int(num_heads), batch_first=True)
        self.out = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, 1),
        )
        self.gate = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        prev_frames: torch.Tensor,
        target_time: torch.Tensor,
        prev_times: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if prev_frames.shape[1] != self.previous_frames:
            return prev_frames
        b, t, h, w = prev_frames.shape
        if prev_times is None or prev_times.shape[1] != t:
            prev_times = torch.zeros(b, t, dtype=prev_frames.dtype, device=prev_frames.device)
        target_time = target_time.view(b, 1).to(dtype=prev_frames.dtype, device=prev_frames.device)
        prev_times = prev_times.to(dtype=prev_frames.dtype, device=prev_frames.device)
        delta_times = (target_time - prev_times).clamp_min(0.0)

        intensity = prev_frames.permute(0, 2, 3, 1).reshape(b * h * w, t, 1)
        prev_time_tokens = prev_times[:, None, None, :, None].expand(b, h, w, t, 1).reshape(b * h * w, t, 1)
        delta_tokens = delta_times[:, None, None, :, None].expand(b, h, w, t, 1).reshape(b * h * w, t, 1)
        token_input = torch.cat([intensity, prev_time_tokens, delta_tokens], dim=-1)

        outputs = []
        chunk_pixels = max(1, self.chunk_pixels)
        for start in range(0, token_input.shape[0], chunk_pixels):
            chunk = token_input[start : start + chunk_pixels]
            tokens = self.token(chunk)
            attended, _ = self.attn(tokens, tokens, tokens, need_weights=False)
            outputs.append(self.out(attended))
        enhanced = torch.cat(outputs, dim=0).reshape(b, h, w, t).permute(0, 3, 1, 2)
        return prev_frames + torch.sigmoid(self.gate) * enhanced

def _maybe_prepare_prev_frames(
    rgb: torch.Tensor,
    time: torch.Tensor,
    previous_frames: int,
    prev_frames: torch.Tensor | None,
    prev_times: torch.Tensor | None,
    temporal_attention: TemporalHistoryAttention | None,
) -> torch.Tensor:
    if prev_frames is None or prev_frames.shape[1] != previous_frames:
        prev_frames = torch.zeros(
            rgb.shape[0],
            previous_frames,
            rgb.shape[-2],
            rgb.shape[-1],
            dtype=rgb.dtype,
            device=rgb.device,
        )
    if temporal_attention is not None and previous_frames > 0:
        prev_frames = temporal_attention(prev_frames, time, prev_times)
    return prev_frames

def _time_modulation_mode(time_embedding: dict | None = None) -> str:
    cfg = time_embedding or {}
    return str(cfg.get("modulation", cfg.get("mode", "concat"))).lower()

def _uses_time_concat(include_time: bool, time_embedding: dict | None = None) -> bool:
    if not include_time:
        return False
    return _time_modulation_mode(time_embedding) in {"concat", "input", "channel", "channels", "raw_concat"}

def _uses_time_film(include_time: bool, time_embedding: dict | None = None) -> bool:
    if not include_time:
        return False
    return _time_modulation_mode(time_embedding) in {"film", "film_only", "fourier_film", "raw_film", "modulate"}

def _time_embedding_vector_channels(time_embedding: dict | None = None) -> int:
    cfg = time_embedding or {}
    embedding_type = str(cfg.get("type", "raw")).lower()
    if embedding_type in {"raw", "scalar", "none"}:
        return 1
    if embedding_type in {"fourier", "sinusoidal"}:
        num_frequencies = max(1, int(cfg.get("num_frequencies", 8)))
        include_raw = bool(cfg.get("include_raw", True))
        return (1 if include_raw else 0) + 2 * num_frequencies
    raise ValueError(f"Unknown time_embedding.type: {cfg.get('type')}")

def _time_embedding_channels(include_time: bool, time_embedding: dict | None = None) -> int:
    if not include_time:
        return 0
    if not _uses_time_concat(include_time, time_embedding):
        return 0
    return _time_embedding_vector_channels(time_embedding)

def _build_time_embedding_vector(time: torch.Tensor, time_embedding: dict | None = None) -> torch.Tensor:
    cfg = time_embedding or {}
    embedding_type = str(cfg.get("type", "raw")).lower()
    t = time.view(time.shape[0], 1).float()
    if embedding_type in {"raw", "scalar", "none"}:
        features = t
    elif embedding_type in {"fourier", "sinusoidal"}:
        num_frequencies = max(1, int(cfg.get("num_frequencies", 8)))
        max_frequency = float(cfg.get("max_frequency", 2 ** (num_frequencies - 1)))
        if num_frequencies == 1:
            frequencies = torch.ones(1, dtype=t.dtype, device=t.device) * max_frequency
        else:
            frequencies = torch.logspace(
                0.0,
                math.log2(max_frequency),
                steps=num_frequencies,
                base=2.0,
                dtype=t.dtype,
                device=t.device,
            )
        angles = t * frequencies[None] * math.pi
        pieces = []
        if bool(cfg.get("include_raw", True)):
            pieces.append(t)
        pieces.extend([torch.sin(angles), torch.cos(angles)])
        features = torch.cat(pieces, dim=1)
    else:
        raise ValueError(f"Unknown time_embedding.type: {cfg.get('type')}")
    return features.to(dtype=time.dtype, device=time.device)

def _time_film_hidden_dim(channels: int, time_embedding: dict | None = None) -> int:
    cfg = time_embedding or {}
    return int(cfg.get("film_hidden_dim", cfg.get("hidden_dim", max(32, min(256, int(channels) * 2)))))

class TimeFiLM(nn.Module):
    """Feature-wise linear modulation from normalized time."""

    def __init__(self, channels: int, time_embedding: dict | None = None) -> None:
        super().__init__()
        channels = int(channels)
        hidden_dim = _time_film_hidden_dim(channels, time_embedding)
        self.time_embedding = time_embedding or {}
        self.net = nn.Sequential(
            nn.Linear(_time_embedding_vector_channels(self.time_embedding), hidden_dim),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim, channels * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        params = self.net(_build_time_embedding_vector(time, self.time_embedding).to(dtype=x.dtype, device=x.device))
        gamma, beta = params.chunk(2, dim=1)
        gamma = gamma.view(gamma.shape[0], gamma.shape[1], 1, 1)
        beta = beta.view(beta.shape[0], beta.shape[1], 1, 1)
        return x * (1.0 + gamma) + beta

def _make_time_films(include_time: bool, time_embedding: dict | None, channels: dict[str, int]) -> nn.ModuleDict:
    if not _uses_time_film(include_time, time_embedding):
        return nn.ModuleDict()
    return nn.ModuleDict({name: TimeFiLM(ch, time_embedding) for name, ch in channels.items()})

def _apply_time_film(films: nn.ModuleDict, name: str, x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
    return films[name](x, time) if name in films else x

def _build_time_embedding_maps(
    time: torch.Tensor,
    height: int,
    width: int,
    time_embedding: dict | None = None,
) -> torch.Tensor:
    features = _build_time_embedding_vector(time, time_embedding)
    return features.view(time.shape[0], -1, 1, 1).expand(-1, -1, height, width)

class OutputHeads(nn.Module):
    """Configurable frame / auxiliary vessel output heads.

    With one output channel this is a plain frame head. With two channels the
    second head predicts an auxiliary key-frame vessel mask. The aux mask feature
    gates the frame feature only when guide_frame_with_aux is enabled.
    """

    def __init__(
        self,
        in_channels: int,
        output_channels: int = 1,
        kernel_size: int = 1,
        padding: int = 0,
        reflection_pad: int = 0,
        guide_frame_with_aux: bool = False,
    ) -> None:
        super().__init__()
        self.output_channels = int(output_channels)
        self.guide_frame_with_aux = bool(guide_frame_with_aux)
        mask_channels = max(0, self.output_channels - 1)
        self.mask_channels = mask_channels

        def make_head(channels: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            if reflection_pad > 0:
                layers.append(nn.ReflectionPad2d(reflection_pad))
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=kernel_size, padding=padding))
            return nn.Sequential(*layers)

        self.frame_head = make_head(1)
        if mask_channels > 0:
            guide_channels = max(8, min(int(in_channels), 64))
            self.mask_decoder = nn.Sequential(
                nn.Conv2d(in_channels, guide_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(guide_channels, affine=True),
                nn.SiLU(inplace=True),
                nn.Conv2d(guide_channels, guide_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(guide_channels, affine=True),
                nn.SiLU(inplace=True),
            )
            self.mask_head = nn.Conv2d(guide_channels, mask_channels, kernel_size=1)
            self.frame_gate = nn.Conv2d(guide_channels, in_channels, kernel_size=1)
            nn.init.zeros_(self.frame_gate.weight)
            nn.init.zeros_(self.frame_gate.bias)
        else:
            self.mask_decoder = None
            self.mask_head = None
            self.frame_gate = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if self.mask_head is None or self.mask_decoder is None:
            frame = torch.sigmoid(self.frame_head(features))
            return frame

        mask_features = self.mask_decoder(features)
        mask = torch.sigmoid(self.mask_head(mask_features))
        if self.frame_gate is not None:
            gate = torch.tanh(self.frame_gate(mask_features))
            features = features * (1.0 + gate)
        frame = torch.sigmoid(self.frame_head(features))
        return torch.cat([frame, mask], dim=1)

class ConvGRUCell2d(nn.Module):
    """ConvGRU cell for spatial bottleneck feature maps."""

    def __init__(self, input_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = int(kernel_size) // 2
        self.hidden_channels = int(hidden_channels)
        self.gates = nn.Conv2d(
            int(input_channels) + self.hidden_channels,
            self.hidden_channels * 2,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.candidate = nn.Conv2d(
            int(input_channels) + self.hidden_channels,
            self.hidden_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

    def init_hidden(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((x.shape[0], self.hidden_channels, x.shape[-2], x.shape[-1]))

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if hidden is None:
            hidden = self.init_hidden(x)
        if hidden.shape[-2:] != x.shape[-2:]:
            hidden = F.interpolate(hidden, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gates = torch.sigmoid(self.gates(torch.cat([x, hidden], dim=1)))
        reset, update = gates.chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1.0 - update) * hidden + update * candidate

class ConvGRUResUNet(nn.Module):
    """Residual U-Net encoder-decoder with a ConvGRU bottleneck state."""

    def __init__(
        self,
        base_channels: int = 32,
        previous_frames: int = 0,
        include_rgb: bool = True,
        include_time: bool = True,
        include_vessel_mask: bool = True,
        include_mtt: bool = False,
        include_lsci: bool = False,
        include_prev_time_delta: bool = False,
        temporal_attention: bool = False,
        temporal_attention_dim: int = 16,
        temporal_attention_heads: int = 4,
        temporal_attention_chunk_pixels: int = 65536,
        time_embedding: dict | None = None,
        output_channels: int = 1,
        detach_interval: int = 0,
    ) -> None:
        super().__init__()
        b = int(base_channels)
        self.previous_frames = int(previous_frames)
        self.include_rgb = bool(include_rgb)
        self.include_time = bool(include_time)
        self.include_vessel_mask = bool(include_vessel_mask)
        self.include_mtt = bool(include_mtt)
        self.include_lsci = bool(include_lsci)
        self.include_prev_time_delta = bool(include_prev_time_delta)
        self.time_embedding = time_embedding or {}
        self.detach_interval = int(detach_interval)
        self.temporal_attention = (
            TemporalHistoryAttention(
                self.previous_frames,
                hidden_dim=temporal_attention_dim,
                num_heads=temporal_attention_heads,
                chunk_pixels=temporal_attention_chunk_pixels,
            )
            if bool(temporal_attention) and self.previous_frames > 0
            else None
        )
        in_channels = _input_channels(
            self.previous_frames,
            self.include_rgb,
            self.include_time,
            self.include_vessel_mask,
            self.include_mtt,
            self.include_lsci,
            self.include_prev_time_delta,
            self.time_embedding,
        )
        self.enc1 = ResidualBlock(in_channels, b)
        self.down1 = nn.Conv2d(b, b * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ResidualBlock(b * 2, b * 2)
        self.down2 = nn.Conv2d(b * 2, b * 4, kernel_size=3, stride=2, padding=1)
        self.enc3 = ResidualBlock(b * 4, b * 4)
        self.down3 = nn.Conv2d(b * 4, b * 8, kernel_size=3, stride=2, padding=1)
        self.enc4 = ResidualBlock(b * 8, b * 8)
        self.down4 = nn.Conv2d(b * 8, b * 16, kernel_size=3, stride=2, padding=1)
        self.bottleneck = ResidualBlock(b * 16, b * 16)
        self.convgru = ConvGRUCell2d(b * 16, b * 16, kernel_size=3)

        self.up4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.dec4 = ResidualBlock(b * 16, b * 8)
        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(b * 4, b * 2)
        self.up1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(b * 2, b)
        self.time_films = _make_time_films(
            self.include_time,
            self.time_embedding,
            {"e1": b, "e2": b * 2, "e3": b * 4, "e4": b * 8, "btm": b * 16, "gru": b * 16, "d4": b * 8, "d3": b * 4, "d2": b * 2, "d1": b},
        )
        self.out = OutputHeads(b, int(output_channels))

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] == ref.shape[-2:]:
            return x
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def _encode(
        self,
        rgb: torch.Tensor,
        time: torch.Tensor,
        vessel_mask: torch.Tensor,
        prev_frames: torch.Tensor | None = None,
        mtt: torch.Tensor | None = None,
        prev_times: torch.Tensor | None = None,
        lsci: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        x = _build_autoregressive_inputs(
            rgb,
            time,
            vessel_mask,
            self.previous_frames,
            self.include_rgb,
            self.include_time,
            self.include_vessel_mask,
            self.include_mtt,
            self.include_lsci,
            self.include_prev_time_delta,
            prev_frames=prev_frames,
            mtt=mtt,
            prev_times=prev_times,
            lsci=lsci,
            temporal_attention=self.temporal_attention,
            time_embedding=self.time_embedding,
        )
        e1 = _apply_time_film(self.time_films, "e1", self.enc1(x), time)
        e2 = _apply_time_film(self.time_films, "e2", self.enc2(self.down1(e1)), time)
        e3 = _apply_time_film(self.time_films, "e3", self.enc3(self.down2(e2)), time)
        e4 = _apply_time_film(self.time_films, "e4", self.enc4(self.down3(e3)), time)
        btm = _apply_time_film(self.time_films, "btm", self.bottleneck(self.down4(e4)), time)
        return btm, (e1, e2, e3, e4)

    def _decode(self, bottleneck: torch.Tensor, skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], time: torch.Tensor) -> torch.Tensor:
        e1, e2, e3, e4 = skips
        d4 = _apply_time_film(self.time_films, "d4", self.dec4(torch.cat([self._match(self.up4(bottleneck), e4), e4], dim=1)), time)
        d3 = _apply_time_film(self.time_films, "d3", self.dec3(torch.cat([self._match(self.up3(d4), e3), e3], dim=1)), time)
        d2 = _apply_time_film(self.time_films, "d2", self.dec2(torch.cat([self._match(self.up2(d3), e2), e2], dim=1)), time)
        d1 = _apply_time_film(self.time_films, "d1", self.dec1(torch.cat([self._match(self.up1(d2), e1), e1], dim=1)), time)
        return self.out(d1)

    def forward_step(
        self,
        rgb: torch.Tensor,
        time: torch.Tensor,
        vessel_mask: torch.Tensor,
        hidden: torch.Tensor | None = None,
        prev_frames: torch.Tensor | None = None,
        mtt: torch.Tensor | None = None,
        prev_times: torch.Tensor | None = None,
        lsci: torch.Tensor | None = None,
        detach_hidden: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if detach_hidden and hidden is not None:
            hidden = hidden.detach()
        bottleneck, skips = self._encode(rgb, time, vessel_mask, prev_frames, mtt, prev_times, lsci)
        hidden = self.convgru(bottleneck, hidden)
        hidden = _apply_time_film(self.time_films, "gru", hidden, time)
        return self._decode(hidden, skips, time), hidden

    def forward(
        self,
        rgb: torch.Tensor,
        time: torch.Tensor,
        vessel_mask: torch.Tensor,
        prev_frames: torch.Tensor | None = None,
        mtt: torch.Tensor | None = None,
        prev_times: torch.Tensor | None = None,
        lsci: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output, hidden = self.forward_step(rgb, time, vessel_mask, hidden, prev_frames, mtt, prev_times, lsci)
        if return_hidden:
            return output, hidden
        return output

    def forward_sequence(
        self,
        rgb: torch.Tensor,
        time: torch.Tensor,
        vessel_mask: torch.Tensor,
        prev_frames: torch.Tensor | None = None,
        mtt: torch.Tensor | None = None,
        prev_times: torch.Tensor | None = None,
        lsci: torch.Tensor | None = None,
        detach_interval: int | None = None,
    ) -> torch.Tensor:
        if rgb.ndim != 5:
            raise ValueError("forward_sequence expects rgb shaped [B, T, C, H, W].")
        hidden = None
        outputs = []
        detach_every = self.detach_interval if detach_interval is None else int(detach_interval)
        for idx in range(rgb.shape[1]):
            step_prev = prev_frames[:, idx] if prev_frames is not None and prev_frames.ndim == 5 else None
            step_prev_times = prev_times[:, idx] if prev_times is not None and prev_times.ndim == 3 else None
            step_mtt = mtt[:, idx] if mtt is not None and mtt.ndim == 5 else mtt
            step_lsci = lsci[:, idx] if lsci is not None and lsci.ndim == 5 else lsci
            output, hidden = self.forward_step(
                rgb[:, idx],
                time[:, idx],
                vessel_mask[:, idx],
                hidden,
                step_prev,
                step_mtt,
                step_prev_times,
                step_lsci,
                detach_hidden=detach_every > 0 and idx > 0 and idx % detach_every == 0,
            )
            outputs.append(output)
        return torch.stack(outputs, dim=1)

def _input_channels(
    previous_frames: int,
    include_rgb: bool,
    include_time: bool,
    include_vessel_mask: bool,
    include_mtt: bool,
    include_lsci: bool,
    include_prev_time_delta: bool,
    time_embedding: dict | None = None,
) -> int:
    input_channels = (3 if include_rgb else 0) + int(previous_frames)
    if include_prev_time_delta:
        input_channels += int(previous_frames)
    if include_lsci:
        input_channels += 1
    input_channels += _time_embedding_channels(include_time, time_embedding)
    if include_vessel_mask:
        input_channels += 1
    if include_mtt:
        input_channels += 1
    return input_channels

def _build_autoregressive_inputs(
    rgb: torch.Tensor,
    time: torch.Tensor,
    vessel_mask: torch.Tensor,
    previous_frames: int,
    include_rgb: bool,
    include_time: bool,
    include_vessel_mask: bool,
    include_mtt: bool,
    include_lsci: bool,
    include_prev_time_delta: bool,
    prev_frames: torch.Tensor | None = None,
    mtt: torch.Tensor | None = None,
    prev_times: torch.Tensor | None = None,
    lsci: torch.Tensor | None = None,
    temporal_attention: TemporalHistoryAttention | None = None,
    time_embedding: dict | None = None,
) -> torch.Tensor:
    inputs = [rgb] if include_rgb else []
    if previous_frames > 0:
        prev_frames = _maybe_prepare_prev_frames(
            rgb,
            time,
            previous_frames,
            prev_frames,
            prev_times,
            temporal_attention,
        )
        if include_prev_time_delta:
            if prev_times is None or prev_times.shape[1] != previous_frames:
                prev_times = torch.zeros(
                    rgb.shape[0],
                    previous_frames,
                    dtype=rgb.dtype,
                    device=rgb.device,
                )
            delta_times = (time.view(time.shape[0], 1) - prev_times.to(dtype=rgb.dtype, device=rgb.device)).clamp_min(0.0)
            for idx in range(previous_frames):
                inputs.append(prev_frames[:, idx : idx + 1])
                delta_map = delta_times[:, idx].view(time.shape[0], 1, 1, 1).expand(-1, 1, rgb.shape[-2], rgb.shape[-1])
                inputs.append(delta_map)
        else:
            inputs.append(prev_frames)
    if include_lsci:
        inputs.append(lsci if lsci is not None else torch.zeros_like(vessel_mask))
    if _uses_time_concat(include_time, time_embedding):
        inputs.append(_build_time_embedding_maps(time, rgb.shape[-2], rgb.shape[-1], time_embedding))
    if include_vessel_mask:
        inputs.append(vessel_mask)
    if include_mtt:
        inputs.append(mtt if mtt is not None else torch.zeros_like(vessel_mask))
    return torch.cat(inputs, dim=1)

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_ch, affine=True),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False) if in_ch != out_ch else nn.Identity()
        self.act = nn.LeakyReLU(0.01, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.net(x) + self.skip(x))

# Historical class names retained for checkpoint/import compatibility.
ConvGRUNNUNet = ConvGRUResUNet
NNUNetResidualBlock = ResidualBlock


def build_model(cfg: dict) -> nn.Module:
    """Build ConvGRU-ResUNet; accept historical names used by saved checkpoints."""
    name = str(cfg.get("name", "convgru_resunet")).lower()
    backbone = str(cfg.get("backbone", "resunet") or "resunet").lower()
    if name not in {"convgru_resunet", "convgru_unet", "convgru_nnunet", "convgru_time_vessel_unet", "time_vessel_convgru_unet"}:
        raise ValueError(f"This code package contains only ConvGRU-ResUNet, not {name!r}.")
    if backbone not in {"resunet", "residual_unet", "nnunet", "nn_unet", "nnunet_residual"}:
        raise ValueError(f"This code package contains only the residual U-Net backbone, not {backbone!r}.")
    if not bool(cfg.get("convgru", {}).get("enabled", True)):
        raise ValueError("ConvGRU must be enabled in this code package.")
    output_mode = str(cfg.get("output_mode", cfg.get("head_mode", "")) or "").lower()
    requested = int(cfg.get("output_channels", 1))
    if output_mode in {"single", "single_head", "frame", "frame_only", "icg", "icg_only"}:
        channels = 1
    elif output_mode in {"dual", "dual_head", "frame_keyframe_mask", "frame_mask", "aux_mask", "keyframe_mask", "vessel_guided", "guided", "guided_mask", "gated_guided"}:
        channels = max(2, requested)
    else:
        channels = requested
    attention = cfg.get("temporal_attention", {})
    model = ConvGRUResUNet(
        base_channels=cfg.get("base_channels", 32),
        previous_frames=cfg.get("previous_frames", 0),
        include_rgb=cfg.get("include_rgb", True),
        include_time=cfg.get("include_time", False),
        include_vessel_mask=cfg.get("include_vessel_mask", True),
        include_mtt=cfg.get("include_mtt", False),
        include_lsci=cfg.get("include_lsci", False),
        include_prev_time_delta=cfg.get("include_prev_time_delta", False),
        temporal_attention=bool(attention.get("enabled", cfg.get("use_temporal_attention", False))),
        temporal_attention_dim=attention.get("hidden_dim", 16),
        temporal_attention_heads=attention.get("heads", 4),
        temporal_attention_chunk_pixels=attention.get("chunk_pixels", 65536),
        time_embedding=cfg.get("time_embedding", {}),
        output_channels=channels,
        detach_interval=int(cfg.get("convgru", {}).get("detach_interval", 0)),
    )
    guided = output_mode in {"vessel_guided", "guided", "guided_mask", "gated_guided"}
    if not output_mode:
        guided = bool(cfg.get("vessel_guided_decoder", {}).get("enabled", False))
    for module in model.modules():
        if isinstance(module, OutputHeads):
            module.guide_frame_with_aux = bool(guided and module.mask_channels > 0)
    return model

