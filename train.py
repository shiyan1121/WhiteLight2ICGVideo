"""Train ConvGRU-ResUNet with reconstruction and auxiliary vessel supervision."""
from __future__ import annotations
import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm
import yaml
from wlicgv.data import PatchFrameDataset, _load_gray, _load_rgb
from wlicgv.time_templates import configure_time_template
from wlicgv.losses import aux_keyframe_vessel_loss, reconstruction_loss, ssim_index, sequence_vessel_curve_loss, temporal_difference_loss
from wlicgv.models import build_model
from wlicgv.utils import ensure_dir, list_subjects_from_patch_dirs, make_subject_split, psnr, save_json, save_preview_panel, seed_everything

def expand_split_file(cfg: dict, config_path: str | Path) -> dict:
    config_path = Path(config_path)
    split_cfg = cfg.get('split', {})
    split_file = split_cfg.get('file') or split_cfg.get('path') or split_cfg.get('split_file')
    if split_file:
        split_path = Path(split_file)
        if not split_path.is_absolute():
            split_path = config_path.parent / split_path
        with split_path.open('r', encoding='utf-8') as f:
            loaded_split = yaml.safe_load(f)
        if not isinstance(loaded_split, dict):
            raise ValueError(f'Split file must contain a mapping: {split_path}')
        cfg['split'] = {**split_cfg, **loaded_split, 'file': str(split_path)}
    return cfg

def load_config(path: str | Path) -> dict:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return expand_split_file(cfg, path)

def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    (hours, rem) = divmod(seconds, 3600)
    (minutes, secs) = divmod(rem, 60)
    if hours:
        return f'{hours:d}h {minutes:02d}m {secs:02d}s'
    return f'{minutes:d}m {secs:02d}s'

def setup_logger(log_path: Path, append: bool=False) -> logging.Logger:
    logger = logging.getLogger('wlicgv.train')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_path, mode='a' if append else 'w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

class JsonlLogger:

    def __init__(self, path: Path, append: bool=False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a' if append else 'w', encoding='utf-8')

    def write(self, event: str, **payload) -> None:
        record = {'time': datetime.now().isoformat(timespec='seconds'), 'event': event, **payload}
        self.file.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.file.flush()

    def close(self) -> None:
        self.file.close()

def build_splits(cfg: dict) -> dict[str, list[str]]:
    split_cfg = cfg['split']
    subjects = list_subjects_from_patch_dirs(cfg['data']['root'])
    if split_cfg.get('train_subjects') or split_cfg.get('val_subjects') or split_cfg.get('test_subjects'):
        return {'train': resolve_subject_names(split_cfg.get('train_subjects', []), subjects, 'train_subjects'), 'val': resolve_subject_names(split_cfg.get('val_subjects', []), subjects, 'val_subjects'), 'test': resolve_subject_names(split_cfg.get('test_subjects', []), subjects, 'test_subjects')}
    return make_subject_split(subjects, split_cfg.get('val_fraction', 0.15), split_cfg.get('test_fraction', 0.15), cfg.get('seed', 2026))

def resolve_subject_names(requested: list[str], available_subjects: list[str], field_name: str) -> list[str]:
    resolved: list[str] = []
    seen: set[str] = set()
    for raw_name in requested:
        name = str(raw_name).strip()
        matches = [subject for subject in available_subjects if subject == name or subject.startswith(f'{name}-')]
        if not matches:
            raise ValueError(f"Unknown subject '{name}' in split.{field_name}. Available subjects include: {available_subjects[:10]}")
        if len(matches) > 1:
            raise ValueError(f"Ambiguous subject shorthand '{name}' in split.{field_name}: {matches}. Use the full subject name instead.")
        subject = matches[0]
        if subject not in seen:
            resolved.append(subject)
            seen.add(subject)
    return resolved

def validate_subject_splits(splits: dict[str, list[str]]) -> None:
    split_sets = {name: set(subjects) for (name, subjects) in splits.items()}
    names = list(split_sets)
    for (i, left) in enumerate(names):
        for right in names[i + 1:]:
            overlap = sorted(split_sets[left] & split_sets[right])
            if overlap:
                raise ValueError(f'Subject-level split violation: {left} and {right} share subjects {overlap}. Move each subject into exactly one split.')

def summarize_dataset(dataset: PatchFrameDataset) -> dict:
    subject_to_patches: dict[str, set[str]] = {}
    subject_to_sequences: dict[str, set[str]] = {}
    for sample in dataset.samples:
        subject_to_patches.setdefault(sample['subject'], set()).add(sample['patch_id'])
        subject_to_sequences.setdefault(sample['subject'], set()).add(sample.get('sequence_id', sample['patch_id']))
    return {'num_samples': len(dataset), 'num_subjects': len(subject_to_patches), 'num_patches': sum((len(patches) for patches in subject_to_patches.values())), 'num_sequences': sum((len(sequences) for sequences in subject_to_sequences.values())), 'subjects': {subject: sorted(patches) for (subject, patches) in sorted(subject_to_patches.items())}, 'sequences': {subject: sorted(sequences) for (subject, sequences) in sorted(subject_to_sequences.items())}}

def make_dataset(cfg: dict, subjects: list[str]) -> PatchFrameDataset:
    data_cfg = cfg['data']
    model_cfg = cfg.get('model', {})
    return PatchFrameDataset(data_root=data_cfg['root'], sample_mode=data_cfg.get('sample_mode', 'Intensity_step_sample'), subjects=subjects, image_size=data_cfg.get('image_size', 512), include_vessel_mask=data_cfg.get('include_vessel_mask', True), input_vessel_mask_source=data_cfg.get('input_vessel_mask_source', 'global'), rgb2vessel_root=data_cfg.get('rgb2vessel_root'), guidance_vessel_mask_source=data_cfg.get('guidance_vessel_mask_source'), include_mtt=data_cfg.get('include_mtt', False), mask_mtt_with_vessel=data_cfg.get('mask_mtt_with_vessel', False), include_lsci=data_cfg.get('include_lsci', False), max_frames_per_patch=data_cfg.get('max_frames_per_patch'), frame_step=data_cfg.get('frame_step', 1), frame_step_offsets=data_cfg.get('frame_step_offsets'), previous_frames=model_cfg.get('previous_frames', 0), time_reference=data_cfg.get('time_reference', 'normalized'), require_complete_patch=data_cfg.get('require_complete_patch', True), time_template=data_cfg.get('time_template'))

class PatchSequenceDataset(Dataset):
    """Patch-level sequence wrapper over PatchFrameDataset."""

    def __init__(self, frame_dataset: PatchFrameDataset) -> None:
        self.frame_dataset = frame_dataset
        self.patch_ids = sorted(frame_dataset.patch_to_indices)

    def __len__(self) -> int:
        return len(self.patch_ids)

    def __getitem__(self, index: int) -> dict:
        return self.frame_dataset.get_patch_sequence(self.patch_ids[index])

def move_batch(batch: dict, device: torch.device) -> dict:
    out = dict(batch)
    for key in ['rgb', 'mask', 'global_mask', 'guidance_mask', 'keyframe_mask', 'keyframe_mask_available', 'mtt', 'lsci', 'target', 'next_target', 'has_next', 'time', 'next_time', 'prev_frames', 'prev_times']:
        out[key] = batch[key].to(device, non_blocking=True)
    return out

def move_sequence_batch(sequence: dict, device: torch.device) -> dict:
    out = dict(sequence)
    for key in ['rgb', 'mask', 'global_mask', 'guidance_mask', 'keyframe_mask', 'keyframe_mask_available', 'mtt', 'lsci', 'target', 'time', 'prev_frames', 'prev_times']:
        if key in out:
            out[key] = out[key].to(device, non_blocking=True)
    return out

def split_generator_output(output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    if output.shape[1] == 1:
        return (output, None)
    return (output[:, :1], output[:, 1:2])

def add_disabled_aux_vessel_parts(parts: dict) -> None:
    parts['aux_vessel_loss'] = 0.0
    parts['aux_vessel_frame_l1'] = 0.0
    parts['aux_vessel_frame_l1_weighted'] = 0.0
    parts['aux_vessel_frame_time_weight_mean'] = 1.0
    parts['aux_vessel_frame_early_fraction'] = 0.0
    parts['aux_vessel_mask_bce'] = 0.0
    parts['aux_vessel_mask_bce_raw'] = 0.0
    parts['aux_vessel_mask_plain_bce'] = 0.0
    parts['aux_vessel_mask_plain_bce_raw'] = 0.0
    parts['aux_vessel_mask_dice'] = 0.0
    parts['aux_vessel_mask_dice_raw'] = 0.0
    parts['aux_vessel_pred_mask_frame_l1'] = 0.0
    parts['aux_vessel_pred_mask_frame_l1_weighted'] = 0.0
    parts['aux_vessel_pred_mean'] = 0.0
    parts['aux_vessel_target_mean'] = 0.0
    parts['aux_vessel_available'] = 0.0

def add_aux_keyframe_vessel_loss(loss: torch.Tensor, parts: dict, pred_frame: torch.Tensor, pred_aux_mask: torch.Tensor | None, batch: dict, loss_cfg: dict) -> tuple[torch.Tensor, dict]:
    aux_cfg = loss_cfg.get('aux_keyframe_vessel', {})
    aux_weight = float(aux_cfg.get('weight', 0.0))
    enabled = bool(aux_cfg.get('enabled', aux_weight > 0.0)) and aux_weight > 0.0 and (pred_aux_mask is not None)
    if not enabled:
        add_disabled_aux_vessel_parts(parts)
        return (loss, parts)
    time_weight_cfg = aux_cfg.get('frame_time_weighting', {})
    (aux_loss, aux_parts) = aux_keyframe_vessel_loss(pred_frame, batch['target'], pred_aux_mask, batch['keyframe_mask'], batch.get('keyframe_mask_available'), frame_weight=aux_cfg.get('frame_weight', 2.0), mask_weight=aux_cfg.get('mask_weight', 0.5), bce_weight=aux_cfg.get('bce_weight', 1.0), dice_weight=aux_cfg.get('dice_weight', 1.0), focal_gamma=aux_cfg.get('focal_gamma', 2.0), focal_alpha=aux_cfg.get('focal_alpha', 0.75), frame_time=batch.get('time'), frame_time_weighting_enabled=bool(time_weight_cfg.get('enabled', False)), early_time_threshold=time_weight_cfg.get('early_threshold', aux_cfg.get('early_time_threshold', 0.35)), early_frame_l1_weight=time_weight_cfg.get('early_weight', aux_cfg.get('early_frame_l1_weight', 2.0)), late_frame_l1_weight=time_weight_cfg.get('late_weight', aux_cfg.get('late_frame_l1_weight', 1.0)), pred_mask_frame_weight=aux_cfg.get('pred_mask_frame_weight', 0.0), target_threshold=aux_cfg.get('target_threshold', 0.5), smooth=aux_cfg.get('smooth', 1.0))
    loss = loss + aux_weight * aux_loss
    parts.update(aux_parts)
    return (loss, parts)

def compute_sequence_vessel_curve_loss(model: torch.nn.Module, dataset: PatchFrameDataset, patch_id: str, cfg: dict, device: torch.device) -> tuple[torch.Tensor, dict[str, float]]:
    curve_cfg = cfg.get('loss', {}).get('sequence_vessel_curve', {})
    sequence = move_sequence_batch(dataset.get_patch_sequence(patch_id), device)
    chunk_size = max(1, int(curve_cfg.get('chunk_size', 4)))
    preds = []
    for start in range(0, sequence['target'].shape[0], chunk_size):
        stop = min(start + chunk_size, sequence['target'].shape[0])
        output = model(sequence['rgb'][start:stop], sequence['time'][start:stop], sequence['mask'][start:stop], sequence['prev_frames'][start:stop], sequence.get('mtt', None)[start:stop] if sequence.get('mtt', None) is not None else None, sequence['prev_times'][start:stop], sequence.get('lsci', None)[start:stop] if sequence.get('lsci', None) is not None else None)
        preds.append(split_generator_output(output)[0])
    pred_seq = torch.cat(preds, dim=0)
    return sequence_vessel_curve_loss(pred_seq, sequence['target'], sequence['guidance_mask'], value_weight=curve_cfg.get('value_weight', 1.0), delta_weight=curve_cfg.get('delta_weight', 1.0), corr_weight=curve_cfg.get('corr_weight', 0.0))

def early_frame_loss_weight(batch: dict, cfg: dict, device: torch.device) -> torch.Tensor:
    early_cfg = cfg.get('loss', {}).get('early_frame_weight', {})
    if not bool(early_cfg.get('enabled', False)):
        return torch.tensor(1.0, dtype=torch.float32, device=device)
    num_frames = int(early_cfg.get('num_frames', 3))
    weight = float(early_cfg.get('weight', 1.0))
    if num_frames <= 0 or weight <= 0:
        return torch.tensor(1.0, dtype=torch.float32, device=device)
    frame_order = batch['frame_order'].to(device).view(-1).float()
    per_item = torch.where(frame_order < float(num_frames), torch.full_like(frame_order, weight), torch.ones_like(frame_order))
    return per_item.mean()

def reconstruction_loss_kwargs(loss_cfg: dict) -> dict:
    return {'l1_weight': loss_cfg.get('l1_weight', 1.0), 'ssim_weight': loss_cfg.get('ssim_weight', 0.2), 'vessel_weight': loss_cfg.get('vessel_weight', 2.0), 'ms_ssim_weight': loss_cfg.get('ms_ssim_weight', 0.0), 'vessel_ssim_weight': loss_cfg.get('vessel_ssim_weight', 0.0), 'vessel_ms_ssim_weight': loss_cfg.get('vessel_ms_ssim_weight', 0.0), 'gradient_weight': loss_cfg.get('gradient_weight', 0.0), 'vessel_gradient_weight': loss_cfg.get('vessel_gradient_weight', 0.0)}

def build_early_frame_sampler(dataset: PatchFrameDataset, cfg: dict, sequential_training: bool) -> WeightedRandomSampler | None:
    sampling_cfg = cfg.get('train', {}).get('early_frame_sampling', {})
    if not bool(sampling_cfg.get('enabled', False)):
        return None
    if sequential_training:
        raise ValueError('train.early_frame_sampling cannot be used with teacher_forcing scheduled/never because frame order must stay sequential.')
    num_frames = int(sampling_cfg.get('num_frames', 3))
    multiplier = float(sampling_cfg.get('multiplier', 3.0))
    if num_frames <= 0 or multiplier <= 0:
        return None
    weights = [multiplier if int(sample.get('frame_order', 0)) < num_frames else 1.0 for sample in dataset.samples]
    num_samples = int(sampling_cfg.get('num_samples', len(weights)) or len(weights))
    return WeightedRandomSampler(weights=torch.tensor(weights, dtype=torch.double), num_samples=num_samples, replacement=bool(sampling_cfg.get('replacement', True)))

def _first_collated_value(value) -> str:
    if isinstance(value, (list, tuple)):
        return str(value[0])
    return str(value)

def _first_int(value) -> int:
    if torch.is_tensor(value):
        return int(value.flatten()[0].item())
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)

def ensure_history_state(batch: dict, state: dict, previous_frames: int) -> None:
    if previous_frames <= 0 or batch['rgb'].shape[0] != 1:
        return
    patch_id = _first_collated_value(batch.get('sequence_id', batch['patch_id']))
    frame_order = _first_int(batch['frame_order'])
    if frame_order == 0 or state.get('patch_id') != patch_id or 'prev_frames' not in state:
        state['patch_id'] = patch_id
        state['prev_frames'] = torch.zeros_like(batch['prev_frames'])
        state['prev_times'] = torch.zeros_like(batch['prev_times'])

def get_history_frames(batch: dict, state: dict, previous_frames: int, teacher_forcing: bool) -> torch.Tensor:
    if teacher_forcing or previous_frames <= 0:
        return batch['prev_frames']
    if batch['rgb'].shape[0] != 1:
        raise ValueError('teacher_forcing=false requires train.batch_size=1 so frames can be processed sequentially.')
    ensure_history_state(batch, state, previous_frames)
    return state['prev_frames']

def get_history_times(batch: dict, state: dict, previous_frames: int, teacher_forcing: bool) -> torch.Tensor:
    if teacher_forcing or previous_frames <= 0:
        return batch['prev_times']
    if batch['rgb'].shape[0] != 1:
        raise ValueError('teacher_forcing=false requires train.batch_size=1 so frames can be processed sequentially.')
    ensure_history_state(batch, state, previous_frames)
    return state['prev_times']

def update_history_frames(state: dict, pred: torch.Tensor, current_time: torch.Tensor, previous_frames: int) -> None:
    if previous_frames <= 0:
        return
    state['prev_frames'] = torch.cat([state['prev_frames'][:, 1:], pred.detach()], dim=1)
    state['prev_times'] = torch.cat([state['prev_times'][:, 1:], current_time.detach().view(current_time.shape[0], 1)], dim=1)

def teacher_forcing_ratio_for_epoch(train_cfg: dict, epoch: int) -> float:
    mode = str(train_cfg.get('teacher_forcing', 'always')).lower()
    if mode in {'true', '1', 'yes', 'always'}:
        return 1.0
    if mode in {'false', '0', 'no', 'never'}:
        return 0.0
    if mode != 'scheduled':
        raise ValueError(f"Unknown teacher_forcing mode: {train_cfg.get('teacher_forcing')}")
    schedule = train_cfg.get('teacher_forcing_schedule', {})
    warmup_epochs = int(schedule.get('warmup_epochs', 5))
    decay_end_epoch = int(schedule.get('decay_end_epoch', 20))
    start_ratio = float(schedule.get('start_ratio', 1.0))
    end_ratio = float(schedule.get('end_ratio', 0.5))
    if epoch <= warmup_epochs:
        return start_ratio
    if epoch >= decay_end_epoch:
        return end_ratio
    progress = (epoch - warmup_epochs) / max(1, decay_end_epoch - warmup_epochs)
    return start_ratio + progress * (end_ratio - start_ratio)

def should_use_teacher_forcing(tf_ratio: float) -> bool:
    if tf_ratio >= 1.0:
        return True
    if tf_ratio <= 0.0:
        return False
    return random.random() < tf_ratio

def apply_finetune_freeze_policy(model: torch.nn.Module, policy: str) -> tuple[int, int]:
    """Apply a coarse fine-tuning freeze policy and return trainable/total parameters."""
    policy = str(policy or 'none').lower()
    if policy in {'none', 'false', 'disabled', 'off'}:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    else:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        if policy == 'encoder_only':
            trainable_prefixes = ('enc', 'down', 'mid', 'bottleneck', 'time_films', 'temporal_attention')
        elif policy == 'input_encoder':
            trainable_prefixes = ('enc1', 'enc2', 'down1', 'time_films', 'temporal_attention')
        else:
            raise ValueError(f'Unknown fine-tune freeze policy: {policy}')
        for (name, parameter) in model.named_parameters():
            if name.startswith(trainable_prefixes):
                parameter.requires_grad_(True)
    trainable = sum((parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    total = sum((parameter.numel() for parameter in model.parameters()))
    return (trainable, total)

def evaluate(model, loader, cfg, device, teacher_forcing: bool=True):
    model.eval()
    losses = []
    metric_sums = {'l1': 0.0, 'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'vessel_l1': 0.0, 'vessel_mse': 0.0, 'vessel_psnr': 0.0, 'vessel_ssim': 0.0, 'aux_mask_precision': 0.0, 'aux_mask_recall': 0.0, 'aux_mask_dice': 0.0}
    metric_counts = {key: 0 for key in metric_sums}
    first_batch = None
    previous_frames = int(cfg.get('model', {}).get('previous_frames', 0))
    autoreg_state: dict = {}
    with torch.no_grad():
        for batch in loader:
            batch = move_batch(batch, device)
            history = get_history_frames(batch, autoreg_state, previous_frames, teacher_forcing)
            history_times = get_history_times(batch, autoreg_state, previous_frames, teacher_forcing)
            output = model(batch['rgb'], batch['time'], batch['mask'], history, batch.get('mtt'), history_times, batch.get('lsci'))
            (pred, pred_aux_mask) = split_generator_output(output)
            guidance_mask = batch.get('guidance_mask', batch['mask'])
            loss_cfg = cfg.get('loss', {})
            (loss, _) = reconstruction_loss(pred, batch['target'], guidance_mask, **reconstruction_loss_kwargs(loss_cfg))
            (loss, _) = add_aux_keyframe_vessel_loss(loss, {}, pred, pred_aux_mask, batch, loss_cfg)
            losses.append(loss.item())
            diff = pred - batch['target']
            l1 = torch.abs(diff).mean()
            mse = diff.square().mean()
            metric_sums['l1'] += float(l1.detach().cpu())
            metric_sums['mse'] += float(mse.detach().cpu())
            metric_sums['psnr'] += psnr(pred, batch['target'])
            metric_sums['ssim'] += float(ssim_index(pred, batch['target']).detach().cpu())
            for key in ['l1', 'mse', 'psnr', 'ssim']:
                metric_counts[key] += 1
            vessel_mask = guidance_mask.float()
            vessel_denom = vessel_mask.sum().clamp_min(1.0)
            vessel_l1 = (torch.abs(diff) * vessel_mask).sum() / vessel_denom
            vessel_mse = (diff.square() * vessel_mask).sum() / vessel_denom
            vessel_pred = pred * vessel_mask
            vessel_target = batch['target'] * vessel_mask
            metric_sums['vessel_l1'] += float(vessel_l1.detach().cpu())
            metric_sums['vessel_mse'] += float(vessel_mse.detach().cpu())
            vessel_psnr = 20.0 * torch.log10(1.0 / torch.sqrt(vessel_mse.clamp_min(1e-08)))
            metric_sums['vessel_psnr'] += float(vessel_psnr.detach().cpu())
            metric_sums['vessel_ssim'] += float(ssim_index(vessel_pred, vessel_target).detach().cpu())
            for key in ['vessel_l1', 'vessel_mse', 'vessel_psnr', 'vessel_ssim']:
                metric_counts[key] += 1
            if pred_aux_mask is not None and 'keyframe_mask' in batch:
                target_mask = batch['keyframe_mask'].float()
                if target_mask.shape[-2:] != pred_aux_mask.shape[-2:]:
                    target_mask = torch.nn.functional.interpolate(target_mask, size=pred_aux_mask.shape[-2:], mode='nearest')
                target_bin = target_mask.ge(0.5)
                pred_bin = pred_aux_mask.ge(0.5)
                if 'keyframe_mask_available' in batch:
                    valid = batch['keyframe_mask_available'].view(-1, 1, 1, 1).bool()
                    pred_bin = pred_bin & valid
                    target_bin = target_bin & valid
                tp = (pred_bin & target_bin).sum().float()
                fp = (pred_bin & ~target_bin).sum().float()
                fn = (~pred_bin & target_bin).sum().float()
                valid_masks = int(batch['keyframe_mask_available'].sum().detach().cpu().item()) if 'keyframe_mask_available' in batch else pred.shape[0]
                if valid_masks > 0:
                    metric_sums['aux_mask_precision'] += float((tp / (tp + fp).clamp_min(1.0)).detach().cpu())
                    metric_sums['aux_mask_recall'] += float((tp / (tp + fn).clamp_min(1.0)).detach().cpu())
                    metric_sums['aux_mask_dice'] += float((2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)).detach().cpu())
                    for key in ['aux_mask_precision', 'aux_mask_recall', 'aux_mask_dice']:
                        metric_counts[key] += 1
            if first_batch is None:
                first_batch = (batch, pred.detach())
            if not teacher_forcing:
                update_history_frames(autoreg_state, pred, batch['time'], previous_frames)
    if not losses:
        return ({'loss': float('nan'), 'psnr': float('nan'), 'ssim': float('nan')}, None)
    metrics = {'loss': sum(losses) / len(losses)}
    for (key, value) in metric_sums.items():
        metrics[key] = value / metric_counts[key] if metric_counts[key] else float('nan')
    return (metrics, first_batch)

def flatten_sequence_tensor(x: torch.Tensor | None) -> torch.Tensor | None:
    if x is None:
        return None
    if x.ndim <= 2:
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
    return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])

def sequence_batch_for_loss(batch: dict) -> dict:
    out = {}
    (bsz, steps) = batch['target'].shape[:2]
    for (key, value) in batch.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        if key == 'global_mask' and value.ndim == 4:
            out[key] = value[:, None].expand(-1, steps, -1, -1, -1).reshape(bsz * steps, *value.shape[1:])
        elif value.ndim >= 3:
            out[key] = value.reshape(bsz * steps, *value.shape[2:])
        elif value.ndim == 2:
            out[key] = value.reshape(bsz * steps, *value.shape[2:])
        else:
            out[key] = value
    return out

def slice_sequence_batch(batch: dict, start: int, end: int) -> dict:
    """Slice time-varying tensors while preserving per-sequence values."""
    steps = int(batch['target'].shape[1])
    out = {}
    for (key, value) in batch.items():
        if key != 'global_mask' and torch.is_tensor(value) and (value.ndim >= 2) and (int(value.shape[1]) == steps):
            out[key] = value[:, start:end]
        else:
            out[key] = value
    return out

def compute_sequence_chunk_loss(model, batch: dict, cfg: dict, device: torch.device, hidden):
    """Run and score one temporal chunk while carrying recurrent state."""
    if not hasattr(model, 'forward_step'):
        raise ValueError('Chunked sequence training requires a model with forward_step().')
    outputs = []
    for idx in range(int(batch['target'].shape[1])):
        step_prev = batch.get('prev_frames')
        step_prev = step_prev[:, idx] if step_prev is not None and step_prev.ndim == 5 else None
        step_prev_times = batch.get('prev_times')
        step_prev_times = step_prev_times[:, idx] if step_prev_times is not None and step_prev_times.ndim == 3 else None
        step_mtt = batch.get('mtt')
        step_mtt = step_mtt[:, idx] if step_mtt is not None and step_mtt.ndim == 5 else step_mtt
        step_lsci = batch.get('lsci')
        step_lsci = step_lsci[:, idx] if step_lsci is not None and step_lsci.ndim == 5 else step_lsci
        (output, hidden) = model.forward_step(batch['rgb'][:, idx], batch['time'][:, idx], batch['mask'][:, idx], hidden, step_prev, step_mtt, step_prev_times, step_lsci)
        outputs.append(output)
    output = torch.stack(outputs, dim=1)
    (pred, pred_aux_mask) = split_generator_output(flatten_sequence_tensor(output))
    flat_batch = sequence_batch_for_loss(batch)
    guidance_mask = flat_batch.get('guidance_mask', flat_batch['mask'])
    loss_cfg = cfg.get('loss', {})
    (loss, parts) = reconstruction_loss(pred, flat_batch['target'], guidance_mask, **reconstruction_loss_kwargs(loss_cfg))
    early_weight = early_frame_loss_weight(flat_batch, cfg, device)
    parts['unweighted_reconstruction_loss'] = parts['loss']
    parts['early_frame_weight'] = float(early_weight.detach().cpu())
    if parts['early_frame_weight'] != 1.0:
        loss = loss * early_weight
        parts['loss'] = float(loss.detach().cpu())
    (loss, parts) = add_aux_keyframe_vessel_loss(loss, parts, pred, pred_aux_mask, flat_batch, loss_cfg)
    return (loss, parts, hidden)

def compute_sequence_loss(model: torch.nn.Module, batch: dict, cfg: dict, device: torch.device, detach_interval: int | None=None) -> tuple[torch.Tensor, dict, torch.Tensor, torch.Tensor | None]:
    if hasattr(model, 'forward_sequence'):
        output = model.forward_sequence(batch['rgb'], batch['time'], batch['mask'], batch.get('prev_frames'), batch.get('mtt'), batch.get('prev_times'), batch.get('lsci'), detach_interval=detach_interval)
    else:
        (bsz, steps) = batch['target'].shape[:2]
        flat_output = model(flatten_sequence_tensor(batch['rgb']), flatten_sequence_tensor(batch['time']), flatten_sequence_tensor(batch['mask']), flatten_sequence_tensor(batch.get('prev_frames')), flatten_sequence_tensor(batch.get('mtt')), flatten_sequence_tensor(batch.get('prev_times')), flatten_sequence_tensor(batch.get('lsci')))
        output = flat_output.reshape(bsz, steps, -1, batch['target'].shape[-2], batch['target'].shape[-1])
    (pred, pred_aux_mask) = split_generator_output(flatten_sequence_tensor(output))
    flat_batch = sequence_batch_for_loss(batch)
    guidance_mask = flat_batch.get('guidance_mask', flat_batch['mask'])
    loss_cfg = cfg.get('loss', {})
    (loss, parts) = reconstruction_loss(pred, flat_batch['target'], guidance_mask, **reconstruction_loss_kwargs(loss_cfg))
    early_weight = early_frame_loss_weight(flat_batch, cfg, device)
    parts['unweighted_reconstruction_loss'] = parts['loss']
    parts['early_frame_weight'] = float(early_weight.detach().cpu())
    if parts['early_frame_weight'] != 1.0:
        loss = loss * early_weight
        parts['loss'] = float(loss.detach().cpu())
    (loss, parts) = add_aux_keyframe_vessel_loss(loss, parts, pred, pred_aux_mask, flat_batch, loss_cfg)
    return (loss, parts, pred, pred_aux_mask)

def evaluate_sequence(model, loader, cfg, device):
    model.eval()
    losses = []
    metric_sums = {'l1': 0.0, 'mse': 0.0, 'psnr': 0.0, 'ssim': 0.0, 'vessel_l1': 0.0, 'vessel_mse': 0.0, 'vessel_psnr': 0.0, 'vessel_ssim': 0.0, 'aux_mask_precision': 0.0, 'aux_mask_recall': 0.0, 'aux_mask_dice': 0.0}
    metric_counts = {key: 0 for key in metric_sums}
    first_batch = None
    with torch.no_grad():
        for batch in loader:
            batch = move_sequence_batch(batch, device)
            (loss, _, pred, pred_aux_mask) = compute_sequence_loss(model, batch, cfg, device)
            flat_batch = sequence_batch_for_loss(batch)
            losses.append(loss.item())
            target = flat_batch['target']
            guidance_mask = flat_batch.get('guidance_mask', flat_batch['mask'])
            diff = pred - target
            l1 = torch.abs(diff).mean()
            mse = diff.square().mean()
            metric_sums['l1'] += float(l1.detach().cpu())
            metric_sums['mse'] += float(mse.detach().cpu())
            metric_sums['psnr'] += psnr(pred, target)
            metric_sums['ssim'] += float(ssim_index(pred, target).detach().cpu())
            for key in ['l1', 'mse', 'psnr', 'ssim']:
                metric_counts[key] += 1
            vessel_mask = guidance_mask.float()
            vessel_denom = vessel_mask.sum().clamp_min(1.0)
            vessel_l1 = (torch.abs(diff) * vessel_mask).sum() / vessel_denom
            vessel_mse = (diff.square() * vessel_mask).sum() / vessel_denom
            metric_sums['vessel_l1'] += float(vessel_l1.detach().cpu())
            metric_sums['vessel_mse'] += float(vessel_mse.detach().cpu())
            metric_sums['vessel_psnr'] += float((20.0 * torch.log10(1.0 / torch.sqrt(vessel_mse.clamp_min(1e-08)))).detach().cpu())
            metric_sums['vessel_ssim'] += float(ssim_index(pred * vessel_mask, target * vessel_mask).detach().cpu())
            for key in ['vessel_l1', 'vessel_mse', 'vessel_psnr', 'vessel_ssim']:
                metric_counts[key] += 1
            if pred_aux_mask is not None and 'keyframe_mask' in flat_batch:
                target_mask = flat_batch['keyframe_mask'].float()
                if target_mask.shape[-2:] != pred_aux_mask.shape[-2:]:
                    target_mask = torch.nn.functional.interpolate(target_mask, size=pred_aux_mask.shape[-2:], mode='nearest')
                target_bin = target_mask.ge(0.5)
                pred_bin = pred_aux_mask.ge(0.5)
                valid_masks = pred.shape[0]
                if 'keyframe_mask_available' in flat_batch:
                    valid = flat_batch['keyframe_mask_available'].view(-1, 1, 1, 1).bool()
                    pred_bin = pred_bin & valid
                    target_bin = target_bin & valid
                    valid_masks = int(flat_batch['keyframe_mask_available'].sum().detach().cpu().item())
                if valid_masks > 0:
                    tp = (pred_bin & target_bin).sum().float()
                    fp = (pred_bin & ~target_bin).sum().float()
                    fn = (~pred_bin & target_bin).sum().float()
                    metric_sums['aux_mask_precision'] += float((tp / (tp + fp).clamp_min(1.0)).detach().cpu())
                    metric_sums['aux_mask_recall'] += float((tp / (tp + fn).clamp_min(1.0)).detach().cpu())
                    metric_sums['aux_mask_dice'] += float((2.0 * tp / (2.0 * tp + fp + fn).clamp_min(1.0)).detach().cpu())
                    for key in ['aux_mask_precision', 'aux_mask_recall', 'aux_mask_dice']:
                        metric_counts[key] += 1
            if first_batch is None:
                first_batch = (flat_batch, pred.detach())
    if not losses:
        return ({'loss': float('nan'), 'psnr': float('nan'), 'ssim': float('nan')}, None)
    metrics = {'loss': sum(losses) / len(losses)}
    for (key, value) in metric_sums.items():
        metrics[key] = value / metric_counts[key] if metric_counts[key] else float('nan')
    return (metrics, first_batch)

def add_prefixed_metrics(metrics: dict, prefix: str, values: dict) -> None:
    for (key, value) in values.items():
        metrics[f'{prefix}_{key}'] = value

def best_checkpoint_metric(metrics: dict, train_cfg: dict) -> tuple[str, float, str]:
    """Resolve the configured metric used to select best.pt."""
    requested = str(train_cfg.get('best_metric', 'loss')).strip()
    mode = str(train_cfg.get('best_metric_mode', '')).strip().lower()
    if mode not in {'min', 'max'}:
        maximize_suffixes = ('ssim', 'psnr', 'dice', 'precision', 'recall')
        mode = 'max' if requested.lower().endswith(maximize_suffixes) else 'min'
    if requested in metrics:
        candidates = [requested]
    else:
        candidates = [f'val_seq_{requested}', f'val_ar_{requested}', f'val_tf_{requested}', f'train_{requested}']
    for name in candidates:
        if name in metrics:
            return (name, float(metrics[name]), mode)
    raise KeyError(f'Best-checkpoint metric {requested!r} was not found. Tried {candidates}; available metrics: {sorted(metrics)}')

def _unique_in_order(values) -> list:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            out.append(value)
            seen.add(value)
    return out

def select_preview_samples(dataset: PatchFrameDataset, patch_id: str | None=None, val_subject_index: int | None=None, case_index: int | None=None) -> list[dict]:
    if not dataset.samples:
        return []
    if patch_id is None:
        subjects = _unique_in_order((sample['subject'] for sample in dataset.samples))
        subject_idx = max(1, int(val_subject_index or 1)) - 1
        if subject_idx >= len(subjects):
            raise ValueError(f'preview_val_subject_index={subject_idx + 1} is out of range; validation set has {len(subjects)} subjects.')
        subject = subjects[subject_idx]
        patch_ids = _unique_in_order((sample['patch_id'] for sample in dataset.samples if sample['subject'] == subject))
        case_idx = max(1, int(case_index or 1)) - 1
        if case_idx >= len(patch_ids):
            raise ValueError(f'preview_case_index={case_idx + 1} is out of range for {subject}; subject has {len(patch_ids)} cases.')
        patch_id = patch_ids[case_idx]
    samples = [sample for sample in dataset.samples if sample['patch_id'] == patch_id]
    return sorted(samples, key=lambda sample: (float(sample.get('time', 0.0)), int(sample.get('frame_order', 0))))

def load_preview_history(sample: dict, image_size: int, previous_frames: int, device: torch.device) -> torch.Tensor:
    if previous_frames <= 0:
        return torch.zeros((1, 0, image_size, image_size), dtype=torch.float32, device=device)
    frames = [_load_gray(path, image_size, 'image').to(device) if path is not None else torch.zeros((1, image_size, image_size), dtype=torch.float32, device=device) for path in sample['prev_target_paths']]
    return torch.cat(frames, dim=0)[None]

def load_preview_history_times(sample: dict, previous_frames: int, device: torch.device) -> torch.Tensor:
    if previous_frames <= 0:
        return torch.zeros((1, 0), dtype=torch.float32, device=device)
    return torch.tensor([sample.get('prev_times', [0.0] * previous_frames)], dtype=torch.float32, device=device)

def load_preview_input_mask(sample: dict, global_mask: torch.Tensor, image_size: int, device: torch.device) -> torch.Tensor:
    input_mask_path = sample.get('input_mask_path')
    if input_mask_path is None:
        return torch.zeros_like(global_mask)
    source = str(sample.get('input_vessel_mask_source', 'global')).lower()
    kind = 'image' if source == 'rgb2vessel_prob' else 'mask'
    return _load_gray(input_mask_path, image_size, kind).to(device)

def preview_static_input_panels(cfg: dict, rgb: torch.Tensor, mask: torch.Tensor, mtt: torch.Tensor, lsci: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    model_cfg = cfg.get('model', {})
    panels: list[tuple[str, torch.Tensor]] = []
    if bool(model_cfg.get('include_rgb', True)):
        panels.append(('RGB', rgb))
    if bool(model_cfg.get('include_vessel_mask', True)):
        panels.append(('Vessel', mask))
    if bool(model_cfg.get('include_mtt', False)):
        panels.append(('MTT', mtt))
    if bool(model_cfg.get('include_lsci', False)):
        panels.append(('LSCI', lsci))
    return panels

def save_epoch_sequence_preview(model, dataset: PatchFrameDataset, cfg: dict, device: torch.device, out_dir: Path, epoch: int, logger: logging.Logger) -> None:
    train_cfg = cfg.get('train', {})
    font_size = int(train_cfg.get('preview_font_size', 24))
    patch_id = train_cfg.get('preview_patch_id')
    samples = select_preview_samples(dataset, patch_id=patch_id, val_subject_index=train_cfg.get('preview_val_subject_index'), case_index=train_cfg.get('preview_case_index'))
    if not samples:
        return
    image_size = cfg['data'].get('image_size', 512)
    previous_frames = int(cfg['model'].get('previous_frames', 0))
    epoch_dir = ensure_dir(out_dir / f'epoch_{epoch:03d}')
    ar_dir = ensure_dir(epoch_dir / 'ar')
    tf_dir = ensure_dir(epoch_dir / 'tf')
    first = samples[0]
    rgb = _load_rgb(first['rgb_path'], image_size).to(device)
    if first['mask_path'] is not None:
        global_mask = _load_gray(first['mask_path'], image_size, 'mask').to(device)
    else:
        global_mask = torch.zeros((1, image_size, image_size), dtype=torch.float32, device=device)
    mask = load_preview_input_mask(first, global_mask, image_size, device)
    if cfg.get('data', {}).get('include_mtt', False) and first.get('mtt_path') is not None:
        mtt = _load_gray(first['mtt_path'], image_size, 'image').to(device)
        if cfg.get('data', {}).get('mask_mtt_with_vessel', False):
            mtt = mtt * global_mask
    else:
        mtt = torch.zeros((1, image_size, image_size), dtype=torch.float32, device=device)
    prev_buffer = torch.zeros((1, previous_frames, image_size, image_size), dtype=torch.float32, device=device)
    prev_time_buffer = torch.zeros((1, previous_frames), dtype=torch.float32, device=device)
    ar_hidden = None
    model.eval()
    with torch.no_grad():
        for (i, sample) in enumerate(samples, start=1):
            target = _load_gray(sample['target_path'], image_size, 'image').to(device)
            time_tensor = torch.tensor([[sample['time']]], dtype=torch.float32, device=device)
            lsci = _load_gray(sample['lsci_path'], image_size, 'image').to(device) if cfg.get('data', {}).get('include_lsci', False) and sample.get('lsci_path') is not None else torch.zeros_like(mask)
            if hasattr(model, 'forward_step'):
                (ar_output, ar_hidden) = model(rgb[None], time_tensor, mask[None], prev_buffer, mtt[None], prev_time_buffer, lsci[None], hidden=ar_hidden, return_hidden=True)
            else:
                ar_output = model(rgb[None], time_tensor, mask[None], prev_buffer, mtt[None], prev_time_buffer, lsci[None])
            (ar_pred_batch, ar_aux_batch) = split_generator_output(ar_output)
            ar_pred = ar_pred_batch[0]
            tf_history = load_preview_history(sample, image_size, previous_frames, device)
            tf_history_times = load_preview_history_times(sample, previous_frames, device)
            tf_output = model(rgb[None], time_tensor, mask[None], tf_history, mtt[None], tf_history_times, lsci[None])
            (tf_pred_batch, tf_aux_batch) = split_generator_output(tf_output)
            tf_pred = tf_pred_batch[0]
            base_title = f"epoch {epoch:03d} | subject {sample['subject']} | patch {sample['patch_id']} | frame {i:03d}/{len(samples):03d} | t={sample['time']:.4f}"
            keyframe_mask = _load_gray(sample['keyframe_mask_path'], image_size, 'mask').to(device) if sample.get('keyframe_mask_path') is not None else mask
            input_panels = [(label, tensor.detach().cpu()) for (label, tensor) in preview_static_input_panels(cfg, rgb, mask, mtt, lsci)]
            ar_input_panels = input_panels + [('KeyVesselGT', keyframe_mask.detach().cpu())]
            if ar_aux_batch is not None:
                ar_input_panels.append(('AuxMask', ar_aux_batch[0].detach().cpu()))
            tf_input_panels = input_panels + [('KeyVesselGT', keyframe_mask.detach().cpu())]
            if tf_aux_batch is not None:
                tf_input_panels.append(('AuxMask', tf_aux_batch[0].detach().cpu()))
            save_preview_panel(rgb.detach().cpu(), mask.detach().cpu(), ar_pred.detach().cpu(), target.detach().cpu(), ar_dir / f'frame_{i:03d}.png', title=f'AR | {base_title}', font_size=font_size, input_panels=ar_input_panels)
            save_preview_panel(rgb.detach().cpu(), mask.detach().cpu(), tf_pred.detach().cpu(), target.detach().cpu(), tf_dir / f'frame_{i:03d}.png', title=f'TF | {base_title}', font_size=font_size, input_panels=tf_input_panels)
            if previous_frames > 0:
                prev_buffer = torch.cat([prev_buffer[:, 1:], ar_pred.detach()[None]], dim=1)
                prev_time_buffer = torch.cat([prev_time_buffer[:, 1:], time_tensor.detach()], dim=1)
    logger.info('Saved %d TF and AR sequence preview frames for %s to %s', len(samples), first['patch_id'], epoch_dir)

def load_finetune_weights(module: torch.nn.Module, checkpoint_state: dict[str, torch.Tensor], logger: logging.Logger, module_name: str, strict: bool=False) -> None:
    if strict:
        module.load_state_dict(checkpoint_state, strict=True)
        logger.info('Loaded strict fine-tune weights for %s: %d tensors', module_name, len(checkpoint_state))
        return
    current_state = module.state_dict()
    matched = {}
    skipped = []
    for (key, value) in checkpoint_state.items():
        if key in current_state and current_state[key].shape == value.shape:
            matched[key] = value
        else:
            target_shape = tuple(current_state[key].shape) if key in current_state else None
            skipped.append((key, tuple(value.shape), target_shape))
    current_state.update(matched)
    module.load_state_dict(current_state, strict=True)
    logger.info('Loaded fine-tune weights for %s: matched=%d, skipped=%d', module_name, len(matched), len(skipped))
    for (key, src_shape, target_shape) in skipped[:20]:
        logger.info('Skipped %s.%s: checkpoint_shape=%s current_shape=%s', module_name, key, src_shape, target_shape)
    if len(skipped) > 20:
        logger.info('Skipped %d additional %s tensors.', len(skipped) - 20, module_name)

def validate_training_scope(cfg: dict) -> None:
    """Reject optional baseline objectives removed from this focused package."""
    loss = cfg.get('loss', {})
    adversarial = dict(cfg.get('discriminator', {}))
    adversarial.update(loss.get('adversarial', {}))
    adv_weight = float(adversarial.get('weight', loss.get('adversarial_weight', 0.0)))
    if bool(adversarial.get('enabled', adv_weight > 0)) and adv_weight > 0:
        raise ValueError('Adversarial training is outside this ConvGRU-ResUNet snapshot.')
    if cfg.get('patchnce', {}).get('enabled', False) or any((float(loss.get(k, 0.0)) > 0 for k in ('patchnce_weight', 'masked_up_weight', 'masked_sp_weight', 'attention_weight'))):
        raise ValueError('PatchNCE / knowledge-attention objectives are outside this snapshot.')
    semantic = loss.get('vessel_semantic', {})
    if bool(semantic.get('enabled', False)) and float(semantic.get('weight', 0.0)) > 0:
        raise ValueError('A separate semantic segmenter is outside this snapshot; auxiliary key-frame supervision is retained.')

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/convgru_resunet.yaml')
    parser.add_argument('--time_template_file', default=None, help='Split-level normalized-time summary JSON; embeds a frozen mean vector in the new run.')
    parser.add_argument('--time_template_split', choices=['train', 'all'], default=None, help='Mean-time split (default train). All includes held-out statistics; exploratory only.')
    parser.add_argument('--resume', default=None, help='Path to a checkpoint, usually experiments/<run>/checkpoints/latest.pt.')
    parser.add_argument('--finetune_from', default=None, help='Initialize model weights from a checkpoint, then train a new run from epoch 1.')
    parser.add_argument('--finetune_strict', action='store_true', help='Require exact checkpoint/model key and shape match for --finetune_from.')
    parser.add_argument('--model_name', '--model_type', dest='model_name', default=None, help='Override model.name from the config for a new run.')
    parser.add_argument('--backbone', choices=['resunet', 'residual_unet', 'nnunet'], default=None, help='Override model.backbone.')
    parser.add_argument('--convgru', choices=['enabled', 'disabled', 'true', 'false', '1', '0'], default=None, help='Enable or disable the ConvGRU bottleneck.')
    parser.add_argument('--include_rgb', choices=['enabled', 'disabled', 'true', 'false', '1', '0'], default=None, help='Enable or disable RGB input.')
    parser.add_argument('--include_time', choices=['enabled', 'disabled', 'true', 'false', '1', '0'], default=None, help='Enable or disable time conditioning.')
    parser.add_argument('--time_reference', default=None, help='Override data.time_reference. Use sequence_index_1based for 1/N, 2/N, ..., N/N according to sequence position.')
    parser.add_argument('--include_vessel_mask', choices=['enabled', 'disabled', 'true', 'false', '1', '0'], default=None, help='Enable or disable vessel-mask model input.')
    parser.add_argument('--experiment_name', default=None, help='Override experiment_name from the config for a new run.')
    parser.add_argument('--output_mode', choices=['frame_only', 'frame_keyframe_mask', 'vessel_guided'], default=None, help='Override model.output_mode: single frame head, independent frame+keyframe-mask heads, or old vessel-guided heads.')
    parser.add_argument('--time_embedding_type', choices=['raw', 'fourier'], default=None, help='Override model.time_embedding.type.')
    parser.add_argument('--time_modulation', choices=['concat', 'film'], default=None, help='Override model.time_embedding.modulation.')
    parser.add_argument('--adversarial', choices=['enabled', 'disabled', 'true', 'false', '1', '0'], default=None, help='Enable or disable adversarial training from loss.adversarial.')
    parser.add_argument('--adv_weight', type=float, default=None, help='Override loss.adversarial.weight.')
    parser.add_argument('--ms_ssim_weight', type=float, default=None, help='Override loss.ms_ssim_weight.')
    parser.add_argument('--vessel_weight', type=float, default=None, help='Override loss.vessel_weight.')
    parser.add_argument('--aux_keyframe_vessel_weight', type=float, default=None, help='Override loss.aux_keyframe_vessel.weight.')
    parser.add_argument('--vessel_ssim_weight', type=float, default=None, help='Override loss.vessel_ssim_weight.')
    parser.add_argument('--vessel_ms_ssim_weight', type=float, default=None, help='Override loss.vessel_ms_ssim_weight.')
    parser.add_argument('--gradient_weight', type=float, default=None, help='Override loss.gradient_weight.')
    parser.add_argument('--vessel_gradient_weight', type=float, default=None, help='Override loss.vessel_gradient_weight.')
    parser.add_argument('--lr', type=float, default=None, help='Override train.lr.')
    parser.add_argument('--max_grad_norm', type=float, default=None, help='Override train.max_grad_norm. Use a positive value to enable gradient clipping, or 0 to disable it.')
    parser.add_argument('--epochs', type=int, default=None, help='Override train.epochs.')
    parser.add_argument('--best_metric', default=None, help='Metric used to save best.pt, e.g. ssim, loss, or the full key val_seq_ssim.')
    parser.add_argument('--best_metric_mode', choices=['min', 'max'], default=None, help='Whether a lower or higher best-checkpoint metric is better.')
    parser.add_argument('--split_file', default=None, help='Override split.file, relative to the config directory unless absolute.')
    parser.add_argument('--freeze_policy', choices=['none', 'encoder_only', 'input_encoder'], default=None, help='Partial fine-tuning policy. encoder_only freezes ConvGRU/decoder/output heads; input_encoder freezes all except shallow input encoder layers.')
    parser.add_argument('--input_vessel_mask_source', choices=['global', 'rgb2vessel_mask', 'rgb2vessel_prob', 'mixed_global_rgb2vessel_mask', 'mixed_global_rgb2vessel_prob'], default=None, help='Override data.input_vessel_mask_source.')
    parser.add_argument('--rgb2vessel_root', default=None, help='Override data.rgb2vessel_root.')
    args = parser.parse_args()
    resume_ckpt = None
    resume_path = Path(args.resume) if args.resume else None
    finetune_ckpt = None
    finetune_path = Path(args.finetune_from) if args.finetune_from else None
    if resume_path is not None and finetune_path is not None:
        raise ValueError('Use either --resume for interrupted training or --finetune_from for a new fine-tuning run, not both.')
    has_cli_override = any((value is not None for value in [args.model_name, args.backbone, args.convgru, args.include_rgb, args.include_time, args.time_reference, args.time_template_file, args.time_template_split, args.include_vessel_mask, args.experiment_name, args.output_mode, args.time_embedding_type, args.time_modulation, args.adversarial, args.adv_weight, args.ms_ssim_weight, args.vessel_weight, args.aux_keyframe_vessel_weight, args.vessel_ssim_weight, args.vessel_ms_ssim_weight, args.gradient_weight, args.vessel_gradient_weight, args.lr, args.max_grad_norm, args.epochs, args.best_metric, args.best_metric_mode, args.split_file, args.freeze_policy, args.input_vessel_mask_source, args.rgb2vessel_root]))
    if resume_path is not None and has_cli_override:
        raise ValueError('Do not use CLI config overrides with --resume; resume uses the checkpoint config.')
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        resume_ckpt = torch.load(resume_path, map_location='cpu')
        cfg = resume_ckpt.get('config', load_config(args.config))
    else:
        if finetune_path is not None:
            if not finetune_path.exists():
                raise FileNotFoundError(f'Fine-tune checkpoint not found: {finetune_path}')
            finetune_ckpt = torch.load(finetune_path, map_location='cpu')
        cfg = load_config(args.config)
    if args.model_name is not None:
        cfg.setdefault('model', {})['name'] = args.model_name
    if args.backbone is not None:
        cfg.setdefault('model', {})['backbone'] = args.backbone
    if args.convgru is not None:
        cfg.setdefault('model', {}).setdefault('convgru', {})['enabled'] = args.convgru in {'enabled', 'true', '1'}
    if args.include_rgb is not None:
        cfg.setdefault('model', {})['include_rgb'] = args.include_rgb in {'enabled', 'true', '1'}
    if args.include_time is not None:
        cfg.setdefault('model', {})['include_time'] = args.include_time in {'enabled', 'true', '1'}
    if args.time_reference is not None:
        cfg.setdefault('data', {})['time_reference'] = args.time_reference
    if args.include_vessel_mask is not None:
        cfg.setdefault('model', {})['include_vessel_mask'] = args.include_vessel_mask in {'enabled', 'true', '1'}
    if args.experiment_name is not None:
        cfg['experiment_name'] = args.experiment_name
    if args.output_mode is not None:
        cfg.setdefault('model', {})['output_mode'] = args.output_mode
    if args.time_embedding_type is not None or args.time_modulation is not None:
        time_embedding_cfg = cfg.setdefault('model', {}).setdefault('time_embedding', {})
        if args.time_embedding_type is not None:
            time_embedding_cfg['type'] = args.time_embedding_type
        if args.time_modulation is not None:
            time_embedding_cfg['modulation'] = args.time_modulation
    if args.adversarial is not None:
        cfg.setdefault('loss', {}).setdefault('adversarial', {})['enabled'] = args.adversarial in {'enabled', 'true', '1'}
    if args.adv_weight is not None:
        cfg.setdefault('loss', {}).setdefault('adversarial', {})['weight'] = args.adv_weight
    if args.ms_ssim_weight is not None:
        cfg.setdefault('loss', {})['ms_ssim_weight'] = args.ms_ssim_weight
    if args.vessel_weight is not None:
        cfg.setdefault('loss', {})['vessel_weight'] = args.vessel_weight
    if args.aux_keyframe_vessel_weight is not None:
        cfg.setdefault('loss', {}).setdefault('aux_keyframe_vessel', {})['weight'] = args.aux_keyframe_vessel_weight
    if args.vessel_ssim_weight is not None:
        cfg.setdefault('loss', {})['vessel_ssim_weight'] = args.vessel_ssim_weight
    if args.vessel_ms_ssim_weight is not None:
        cfg.setdefault('loss', {})['vessel_ms_ssim_weight'] = args.vessel_ms_ssim_weight
    if args.gradient_weight is not None:
        cfg.setdefault('loss', {})['gradient_weight'] = args.gradient_weight
    if args.vessel_gradient_weight is not None:
        cfg.setdefault('loss', {})['vessel_gradient_weight'] = args.vessel_gradient_weight
    if args.lr is not None:
        cfg.setdefault('train', {})['lr'] = args.lr
    if args.max_grad_norm is not None:
        cfg.setdefault('train', {})['max_grad_norm'] = args.max_grad_norm
    if args.epochs is not None:
        cfg.setdefault('train', {})['epochs'] = args.epochs
    if args.best_metric is not None:
        cfg.setdefault('train', {})['best_metric'] = args.best_metric
    if args.best_metric_mode is not None:
        cfg.setdefault('train', {})['best_metric_mode'] = args.best_metric_mode
    if args.split_file is not None:
        cfg.setdefault('split', {})['file'] = args.split_file
        cfg = expand_split_file(cfg, args.config)
    if args.freeze_policy is not None:
        cfg.setdefault('train', {})['freeze_policy'] = args.freeze_policy
    if args.input_vessel_mask_source is not None:
        cfg.setdefault('data', {})['input_vessel_mask_source'] = args.input_vessel_mask_source
    if args.rgb2vessel_root is not None:
        cfg.setdefault('data', {})['rgb2vessel_root'] = args.rgb2vessel_root
    if args.time_template_file and args.time_reference not in {None, 'shared_template'}:
        raise ValueError('--time_template_file cannot be combined with a different --time_reference.')
    configure_time_template(cfg, args.time_template_file, args.time_template_split)
    seed_everything(cfg.get('seed', 2026))
    base_output_root = Path(cfg['output']['root'])
    experiments_root = base_output_root if base_output_root.name.lower() == 'experiments' else base_output_root / 'experiments'
    if resume_path is not None:
        output_root = resume_path.resolve().parents[1]
        run_name = output_root.name
    else:
        date_tag = datetime.now().strftime('%y%m%d')
        run_name = f"{date_tag}-{cfg['experiment_name']}"
        output_root = ensure_dir(experiments_root / run_name)
    ckpt_dir = ensure_dir(output_root / 'checkpoints')
    preview_dir = ensure_dir(output_root / 'previews')
    logger = setup_logger(output_root / 'train.txt', append=resume_path is not None)
    jsonl_logger = JsonlLogger(output_root / 'logging.jsonl', append=resume_path is not None)
    if resume_path is not None:
        logger.info('Resuming from checkpoint: %s', resume_path)
    if finetune_path is not None:
        logger.info('Fine-tuning from checkpoint: %s', finetune_path)
    logger.info('Run directory: %s', output_root)
    logger.info('Config path: %s', args.config)
    cfg.setdefault('output', {})['experiments_root'] = str(experiments_root)
    cfg['output']['run_name'] = run_name
    cfg['output']['run_dir'] = str(output_root)
    if resume_ckpt is not None and resume_ckpt.get('splits') is not None:
        splits = resume_ckpt['splits']
    else:
        splits = build_splits(cfg)
    validate_subject_splits(splits)
    with open(output_root / 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    train_set = make_dataset(cfg, splits['train'])
    val_set = make_dataset(cfg, splits['val'])
    split_summary = {'split_unit': 'subject', 'train': summarize_dataset(train_set), 'val': summarize_dataset(val_set), 'test_subjects': splits.get('test', [])}
    save_json({'config': cfg, 'splits': splits, 'split_summary': split_summary}, output_root / 'run_config_and_split.json')
    save_json(split_summary, output_root / 'subject_split_summary.json')
    logger.info('Train samples: %d | Val samples: %d', len(train_set), len(val_set))
    logger.info('Train subjects: %s', splits['train'])
    logger.info('Val subjects: %s', splits['val'])
    logger.info('Split unit: subject')
    if len(train_set) == 0:
        raise RuntimeError('No training samples found. Check data.root, sample_mode, and subject split.')
    train_cfg = cfg['train']
    teacher_forcing_mode = str(train_cfg.get('teacher_forcing', 'always')).lower()
    sequential_training = teacher_forcing_mode not in {'true', '1', 'yes', 'always'}
    sequence_train_cfg = train_cfg.get('sequence_training', {})
    model_name = str(cfg.get('model', {}).get('name', '')).lower()
    convgru_enabled = bool(cfg.get('model', {}).get('convgru', {}).get('enabled', model_name.startswith('convgru')))
    use_sequence_training = bool(sequence_train_cfg.get('enabled', convgru_enabled))
    if sequential_training and train_cfg.get('batch_size', 4) != 1 and (not use_sequence_training):
        raise ValueError('teacher_forcing scheduled/never requires train.batch_size=1.')
    if use_sequence_training:
        sequence_batch_size = int(sequence_train_cfg.get('batch_size', 1))
        if sequence_batch_size != 1:
            logger.warning('sequence_training.batch_size=%d requires equal sequence lengths in each batch.', sequence_batch_size)
        train_sequence_set = PatchSequenceDataset(train_set)
        val_sequence_set = PatchSequenceDataset(val_set) if len(val_set) else None
        train_loader = DataLoader(train_sequence_set, batch_size=sequence_batch_size, shuffle=True, num_workers=train_cfg.get('num_workers', 0), pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_sequence_set, batch_size=int(sequence_train_cfg.get('val_batch_size', 1)), shuffle=False, num_workers=train_cfg.get('num_workers', 0), pin_memory=torch.cuda.is_available()) if val_sequence_set is not None and len(val_sequence_set) else None
        val_ar_loader = None
        logger.info('Sequence training: enabled | train patches: %d | val patches: %d', len(train_sequence_set), len(val_sequence_set) if val_sequence_set is not None else 0)
    else:
        train_sampler = build_early_frame_sampler(train_set, cfg, sequential_training)
        train_loader = DataLoader(train_set, batch_size=train_cfg.get('batch_size', 4), shuffle=not sequential_training and train_sampler is None, sampler=train_sampler, num_workers=train_cfg.get('num_workers', 0), pin_memory=torch.cuda.is_available())
        val_loader = DataLoader(val_set, batch_size=train_cfg.get('val_batch_size', train_cfg.get('batch_size', 4)), shuffle=False, num_workers=train_cfg.get('num_workers', 0), pin_memory=torch.cuda.is_available()) if len(val_set) else None
        val_ar_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=train_cfg.get('num_workers', 0), pin_memory=torch.cuda.is_available()) if len(val_set) else None
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    validate_training_scope(cfg)
    model = build_model(cfg['model']).to(device)
    loss_cfg = cfg.get('loss', {})
    temporal_change_weight = float(loss_cfg.get('temporal_change_weight', 0.0))
    sequence_curve_cfg = loss_cfg.get('sequence_vessel_curve', {})
    sequence_curve_weight = float(sequence_curve_cfg.get('weight', 0.0))
    use_sequence_curve = bool(sequence_curve_cfg.get('enabled', sequence_curve_weight > 0.0)) and sequence_curve_weight > 0.0
    sequence_curve_every = max(1, int(sequence_curve_cfg.get('every_n_steps', 1)))
    discriminator_cfg = dict(cfg.get('discriminator', {}))
    discriminator_cfg.update(loss_cfg.get('adversarial', {}))
    patchnce_cfg = cfg.get('patchnce', {})
    adv_weight = float(discriminator_cfg.get('weight', loss_cfg.get('adversarial_weight', 0.0)))
    patchnce_weight = float(loss_cfg.get('patchnce_weight', 0.0))
    masked_up_weight = float(loss_cfg.get('masked_up_weight', 0.0))
    masked_sp_weight = float(loss_cfg.get('masked_sp_weight', 0.0))
    attention_weight = 0.0
    vessel_semantic_cfg = loss_cfg.get('vessel_semantic', {})
    vessel_semantic_weight = float(vessel_semantic_cfg.get('weight', 0.0))
    use_vessel_semantic = False
    use_adversarial = False
    use_masked_patchnce = False
    use_patchnce = False
    discriminator = None
    patch_feature_encoder = None
    vessel_segmenter = None
    attention_activation: dict[str, torch.Tensor | None] = {'value': None}
    attention_hook = None
    logger.info('Device: %s', device)
    logger.info('Model: %s', model.__class__.__name__)
    logger.info('Backbone: %s | ConvGRU: %s', cfg.get('model', {}).get('backbone', 'from_name'), convgru_enabled)
    logger.info('Discriminator: %s', 'disabled')
    logger.info('PatchNCE: %s', 'disabled')
    logger.info('Vessel semantic loss: %s', 'disabled')
    logger.info('Train config: %s', train_cfg)
    logger.info('Teacher forcing: train=%s | eval=tf+ar', train_cfg.get('teacher_forcing', 'always'))
    logger.info('Early frame loss weight: %s', cfg.get('loss', {}).get('early_frame_weight', {'enabled': False}))
    logger.info('Early frame sampling: %s', train_cfg.get('early_frame_sampling', {'enabled': False}))
    freeze_policy = str(train_cfg.get('freeze_policy', 'none') or 'none').lower()
    (trainable_params, total_params) = apply_finetune_freeze_policy(model, freeze_policy)
    logger.info('Freeze policy: %s | trainable params: %d / %d (%.2f%%)', freeze_policy, trainable_params, total_params, 100.0 * trainable_params / max(1, total_params))
    generator_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not generator_parameters:
        raise RuntimeError('No trainable generator parameters after applying freeze_policy.')
    optimizer = torch.optim.AdamW(generator_parameters, lr=train_cfg.get('lr', 0.0002), weight_decay=train_cfg.get('weight_decay', 0.0), betas=(train_cfg.get('beta1', 0.5), train_cfg.get('beta2', 0.999)))
    optimizer_d = None
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.get('amp', True) and device.type == 'cuda')
    max_grad_norm = float(train_cfg.get('max_grad_norm', 0.0))
    if max_grad_norm < 0.0:
        raise ValueError(f'train.max_grad_norm must be non-negative, got {max_grad_norm}.')
    logger.info('Gradient clipping: %s', f'max norm {max_grad_norm:g}' if max_grad_norm > 0.0 else 'disabled')
    requested_best_metric = str(train_cfg.get('best_metric', 'loss')).strip()
    best_metric_mode = str(train_cfg.get('best_metric_mode', '')).strip().lower()
    if best_metric_mode not in {'min', 'max'}:
        maximize_suffixes = ('ssim', 'psnr', 'dice', 'precision', 'recall')
        best_metric_mode = 'max' if requested_best_metric.lower().endswith(maximize_suffixes) else 'min'
    best_val = float('inf') if best_metric_mode == 'min' else float('-inf')
    total_epochs = train_cfg.get('epochs', 80)
    best_start_epoch = int(train_cfg.get('best_start_epoch', 1))
    start_epoch = 1
    global_step = 0
    if resume_ckpt is not None:
        model.load_state_dict(resume_ckpt['model'])
        optimizer.load_state_dict(resume_ckpt['optimizer'])
        if None is not None and resume_ckpt.get('discriminator') is not None:
            None.load_state_dict(resume_ckpt['discriminator'])
        if None is not None and resume_ckpt.get('optimizer_d') is not None:
            None.load_state_dict(resume_ckpt['optimizer_d'])
        if None is not None and resume_ckpt.get('patch_feature_encoder') is not None:
            None.load_state_dict(resume_ckpt['patch_feature_encoder'])
        if resume_ckpt.get('scaler') is not None and scaler.is_enabled():
            scaler.load_state_dict(resume_ckpt['scaler'])
        start_epoch = int(resume_ckpt.get('epoch', 0)) + 1
        global_step = int(resume_ckpt.get('global_step', (start_epoch - 1) * max(1, len(train_loader))))
        best_ckpt_path = ckpt_dir / 'best.pt'
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location='cpu')
            best_metrics = best_ckpt.get('metrics', {})
            (best_metric_name, best_val, best_metric_mode) = best_checkpoint_metric(best_metrics, train_cfg)
        else:
            best_metric_name = requested_best_metric
        logger.info('Resume state: start_epoch=%d, global_step=%d, best_metric=%s, mode=%s, best_score=%.6f', start_epoch, global_step, best_metric_name, best_metric_mode, best_val)
        if start_epoch > int(total_epochs):
            logger.info('Checkpoint already reached total_epochs=%d; nothing to train.', total_epochs)
            jsonl_logger.close()
            return
    elif finetune_ckpt is not None:
        if 'model' not in finetune_ckpt:
            raise ValueError(f"Fine-tune checkpoint {finetune_path} does not contain a 'model' state dict.")
        load_finetune_weights(model, finetune_ckpt['model'], logger, 'model', strict=args.finetune_strict)
        logger.info('Fine-tune state: start_epoch=1, global_step=0, optimizer reinitialized, best_metric=%s, mode=%s', requested_best_metric, best_metric_mode)
    train_start_time = time.time()
    completed_epoch_times: list[float] = []
    for epoch in range(start_epoch, total_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        running = []
        accumulation_steps = max(1, int(train_cfg.get('gradient_accumulation_steps', 1)))
        log_every = max(1, int(train_cfg.get('log_every', 1)))
        if (sequential_training or use_sequence_training) and accumulation_steps != 1:
            raise ValueError('sequence/scheduled training currently requires gradient_accumulation_steps=1.')
        if False and accumulation_steps != 1:
            raise ValueError('adversarial training currently requires gradient_accumulation_steps=1.')
        tf_ratio = teacher_forcing_ratio_for_epoch(train_cfg, epoch)
        logger.info('Epoch %03d teacher_forcing_ratio=%.4f', epoch, tf_ratio)
        autoreg_state: dict = {}
        previous_frames = int(cfg.get('model', {}).get('previous_frames', 0))
        pbar = tqdm(train_loader, desc=f'Epoch {epoch:03d}')
        optimizer.zero_grad(set_to_none=True)
        for (step, batch) in enumerate(pbar, start=1):
            if use_sequence_training:
                batch = move_sequence_batch(batch, device)
                chunk_size = int(sequence_train_cfg.get('chunk_size', 0))
                if chunk_size > 0:
                    total_steps = int(batch['target'].shape[1])
                    hidden = None
                    weighted_parts = []
                    loss_value = 0.0
                    for chunk_start in range(0, total_steps, chunk_size):
                        chunk_end = min(total_steps, chunk_start + chunk_size)
                        chunk_batch = slice_sequence_batch(batch, chunk_start, chunk_end)
                        chunk_weight = float(chunk_end - chunk_start) / float(total_steps)
                        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                            (chunk_loss, current_parts, hidden) = compute_sequence_chunk_loss(model, chunk_batch, cfg, device, hidden)
                        scaler.scale(chunk_loss * chunk_weight).backward()
                        hidden = hidden.detach()
                        loss_value += float(chunk_loss.detach().cpu()) * chunk_weight
                        weighted_parts.append((current_parts, chunk_weight))
                    if max_grad_norm > 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(generator_parameters, max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    parts = {}
                    for (current_parts, chunk_weight) in weighted_parts:
                        for (key, value) in current_parts.items():
                            if isinstance(value, (int, float)):
                                parts[key] = parts.get(key, 0.0) + float(value) * chunk_weight
                    running.append(loss_value)
                    pbar.set_postfix(loss=sum(running) / len(running), l1=parts.get('l1', 0.0), early_w=parts.get('early_frame_weight', 1.0), aux=parts.get('aux_vessel_loss', 0.0))
                    global_step += 1
                    if global_step % log_every == 0 or step == len(train_loader):
                        jsonl_logger.write('train_iter', step=global_step, epoch=epoch, iter=step, total_iters=len(train_loader), loss=loss_value, l1=parts.get('l1', 0.0), masked_l1=parts.get('masked_l1', 0.0), ssim_loss=parts.get('ssim_loss', 0.0), aux_vessel_loss=parts.get('aux_vessel_loss', 0.0), sequence_chunk_size=chunk_size, sequence_length=total_steps, lr=optimizer.param_groups[0]['lr'])
                    continue
                with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                    detach_interval = int(sequence_train_cfg.get('detach_interval', cfg.get('model', {}).get('convgru', {}).get('detach_interval', 0)))
                    attention_activation['value'] = None
                    (loss, parts, pred, _) = compute_sequence_loss(model, batch, cfg, device, detach_interval=detach_interval)
                    parts['temporal_change_l1'] = 0.0
                    parts['temporal_change_masked_l1'] = 0.0
                    parts['temporal_change_curve_loss'] = 0.0
                    parts['sequence_curve_loss'] = 0.0
                    parts['sequence_curve_value_loss'] = 0.0
                    parts['sequence_curve_delta_loss'] = 0.0
                    parts['sequence_curve_corr_loss'] = 0.0
                    parts['vessel_semantic_loss'] = 0.0
                    parts['vessel_semantic_bce'] = 0.0
                    parts['vessel_semantic_dice'] = 0.0
                    parts['vessel_semantic_pred_mean'] = 0.0
                    parts['vessel_semantic_target_mean'] = 0.0
                    parts['patchnce_loss'] = 0.0
                    parts['masked_up_loss'] = 0.0
                    parts['masked_sp_loss'] = 0.0
                    parts['attention_loss'] = 0.0
                    parts['g_adv_loss'] = 0.0
                    parts['d_loss'] = 0.0
                scaler.scale(loss).backward()
                if max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(generator_parameters, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                running.append(loss.item())
                pbar.set_postfix(loss=sum(running) / len(running), l1=parts['l1'], early_w=parts.get('early_frame_weight', 1.0), aux=parts.get('aux_vessel_loss', 0.0))
                global_step += 1
                if global_step % log_every == 0 or step == len(train_loader):
                    jsonl_logger.write('train_iter', step=global_step, epoch=epoch, iter=step, total_iters=len(train_loader), loss=float(loss.detach().cpu()), l1=parts['l1'], masked_l1=parts['masked_l1'], ssim_loss=parts['ssim_loss'], early_frame_weight=parts.get('early_frame_weight', 1.0), unweighted_reconstruction_loss=parts.get('unweighted_reconstruction_loss', parts.get('loss', 0.0)), aux_vessel_loss=parts.get('aux_vessel_loss', 0.0), aux_vessel_frame_l1=parts.get('aux_vessel_frame_l1', 0.0), aux_vessel_frame_l1_weighted=parts.get('aux_vessel_frame_l1_weighted', 0.0), aux_vessel_frame_time_weight_mean=parts.get('aux_vessel_frame_time_weight_mean', 1.0), aux_vessel_frame_early_fraction=parts.get('aux_vessel_frame_early_fraction', 0.0), aux_vessel_mask_bce=parts.get('aux_vessel_mask_bce', 0.0), aux_vessel_mask_bce_raw=parts.get('aux_vessel_mask_bce_raw', 0.0), aux_vessel_mask_plain_bce=parts.get('aux_vessel_mask_plain_bce', 0.0), aux_vessel_mask_plain_bce_raw=parts.get('aux_vessel_mask_plain_bce_raw', 0.0), aux_vessel_mask_dice=parts.get('aux_vessel_mask_dice', 0.0), aux_vessel_mask_dice_raw=parts.get('aux_vessel_mask_dice_raw', 0.0), aux_vessel_pred_mask_frame_l1=parts.get('aux_vessel_pred_mask_frame_l1', 0.0), aux_vessel_pred_mask_frame_l1_weighted=parts.get('aux_vessel_pred_mask_frame_l1_weighted', 0.0), aux_vessel_pred_mean=parts.get('aux_vessel_pred_mean', 0.0), aux_vessel_target_mean=parts.get('aux_vessel_target_mean', 0.0), aux_vessel_available=parts.get('aux_vessel_available', 0.0), g_adv_loss=parts.get('g_adv_loss', 0.0), d_loss=parts.get('d_loss', 0.0), convgru_detach_interval=detach_interval, sequence_length=int(batch['target'].shape[1]), lr=optimizer.param_groups[0]['lr'])
                continue
            batch = move_batch(batch, device)
            step_teacher_forcing = should_use_teacher_forcing(tf_ratio)
            if sequential_training:
                ensure_history_state(batch, autoreg_state, previous_frames)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                history = get_history_frames(batch, autoreg_state, previous_frames, step_teacher_forcing)
                history_times = get_history_times(batch, autoreg_state, previous_frames, step_teacher_forcing)
                attention_activation['value'] = None
                output = model(batch['rgb'], batch['time'], batch['mask'], history, batch.get('mtt'), history_times, batch.get('lsci'))
                (pred, pred_aux_mask) = split_generator_output(output)
                guidance_mask = batch.get('guidance_mask', batch['mask'])
                (loss, parts) = reconstruction_loss(pred, batch['target'], guidance_mask, **reconstruction_loss_kwargs(cfg.get('loss', {})))
                early_weight = early_frame_loss_weight(batch, cfg, device)
                parts['unweighted_reconstruction_loss'] = parts['loss']
                parts['early_frame_weight'] = float(early_weight.detach().cpu())
                if parts['early_frame_weight'] != 1.0:
                    loss = loss * early_weight
                    parts['loss'] = float(loss.detach().cpu())
                (loss, parts) = add_aux_keyframe_vessel_loss(loss, parts, pred, pred_aux_mask, batch, loss_cfg)
                if temporal_change_weight > 0.0:
                    if previous_frames > 0:
                        next_history = torch.cat([history[:, 1:], pred.detach()], dim=1)
                        next_history_times = torch.cat([history_times[:, 1:], batch['time'].view(batch['time'].shape[0], 1)], dim=1)
                    else:
                        next_history = history
                        next_history_times = history_times
                    pred_next_output = model(batch['rgb'], batch['next_time'], batch['mask'], next_history, batch.get('mtt'), next_history_times, batch.get('lsci'))
                    (pred_next, _) = split_generator_output(pred_next_output)
                    (temporal_change_loss, temporal_change_parts) = temporal_difference_loss(pred, pred_next, batch['target'], batch['next_target'], guidance_mask, batch['has_next'].view(-1), vessel_weight=loss_cfg.get('temporal_change_vessel_weight', 0.0), curve_weight=loss_cfg.get('temporal_change_curve_weight', 0.0))
                    loss = loss + temporal_change_weight * temporal_change_loss
                    parts['temporal_change_l1'] = temporal_change_parts['temporal_l1']
                    parts['temporal_change_masked_l1'] = temporal_change_parts['temporal_masked_l1']
                    parts['temporal_change_curve_loss'] = temporal_change_parts['temporal_curve_loss']
                else:
                    parts['temporal_change_l1'] = 0.0
                    parts['temporal_change_masked_l1'] = 0.0
                    parts['temporal_change_curve_loss'] = 0.0
                if use_sequence_curve and step % sequence_curve_every == 0:
                    sequence_patch_id = _first_collated_value(batch.get('sequence_id', batch['patch_id']))
                    (seq_curve_loss, seq_curve_parts) = compute_sequence_vessel_curve_loss(model, train_set, sequence_patch_id, cfg, device)
                    loss = loss + sequence_curve_weight * seq_curve_loss
                    parts['sequence_curve_loss'] = float(seq_curve_loss.detach().cpu())
                    parts.update(seq_curve_parts)
                else:
                    parts['sequence_curve_loss'] = 0.0
                    parts['sequence_curve_value_loss'] = 0.0
                    parts['sequence_curve_delta_loss'] = 0.0
                    parts['sequence_curve_corr_loss'] = 0.0
                parts['vessel_semantic_loss'] = 0.0
                parts['vessel_semantic_bce'] = 0.0
                parts['vessel_semantic_dice'] = 0.0
                parts['vessel_semantic_pred_mean'] = 0.0
                parts['vessel_semantic_target_mean'] = 0.0
                parts['patchnce_loss'] = 0.0
                parts['masked_up_loss'] = 0.0
                parts['masked_sp_loss'] = 0.0
                parts['attention_loss'] = 0.0
                parts['g_adv_loss'] = 0.0
                scaled_loss = loss / accumulation_steps
            scaler.scale(scaled_loss).backward()
            if step % accumulation_steps == 0 or step == len(train_loader):
                if max_grad_norm > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(generator_parameters, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            parts['d_loss'] = 0.0
            if sequential_training:
                update_history_frames(autoreg_state, pred, batch['time'], previous_frames)
            running.append(loss.item())
            pbar.set_postfix(loss=sum(running) / len(running), l1=parts['l1'], early_w=parts.get('early_frame_weight', 1.0), change=parts.get('temporal_change_l1', 0.0), seq=parts.get('sequence_curve_loss', 0.0), aux=parts.get('aux_vessel_loss', 0.0), vsem=parts.get('vessel_semantic_loss', 0.0), adv=parts.get('g_adv_loss', 0.0), nce=parts.get('patchnce_loss', 0.0))
            global_step += 1
            if global_step % log_every == 0 or step == len(train_loader):
                jsonl_logger.write('train_iter', step=global_step, epoch=epoch, iter=step, total_iters=len(train_loader), loss=float(loss.detach().cpu()), l1=parts['l1'], masked_l1=parts['masked_l1'], ssim_loss=parts['ssim_loss'], early_frame_weight=parts.get('early_frame_weight', 1.0), unweighted_reconstruction_loss=parts.get('unweighted_reconstruction_loss', parts.get('loss', 0.0)), patchnce_loss=parts.get('patchnce_loss', 0.0), masked_up_loss=parts.get('masked_up_loss', 0.0), masked_sp_loss=parts.get('masked_sp_loss', 0.0), attention_loss=parts.get('attention_loss', 0.0), temporal_change_l1=parts.get('temporal_change_l1', 0.0), temporal_change_masked_l1=parts.get('temporal_change_masked_l1', 0.0), temporal_change_curve_loss=parts.get('temporal_change_curve_loss', 0.0), sequence_curve_loss=parts.get('sequence_curve_loss', 0.0), sequence_curve_value_loss=parts.get('sequence_curve_value_loss', 0.0), sequence_curve_delta_loss=parts.get('sequence_curve_delta_loss', 0.0), sequence_curve_corr_loss=parts.get('sequence_curve_corr_loss', 0.0), aux_vessel_loss=parts.get('aux_vessel_loss', 0.0), aux_vessel_frame_l1=parts.get('aux_vessel_frame_l1', 0.0), aux_vessel_frame_l1_weighted=parts.get('aux_vessel_frame_l1_weighted', 0.0), aux_vessel_frame_time_weight_mean=parts.get('aux_vessel_frame_time_weight_mean', 1.0), aux_vessel_frame_early_fraction=parts.get('aux_vessel_frame_early_fraction', 0.0), aux_vessel_mask_bce=parts.get('aux_vessel_mask_bce', 0.0), aux_vessel_mask_bce_raw=parts.get('aux_vessel_mask_bce_raw', 0.0), aux_vessel_mask_plain_bce=parts.get('aux_vessel_mask_plain_bce', 0.0), aux_vessel_mask_plain_bce_raw=parts.get('aux_vessel_mask_plain_bce_raw', 0.0), aux_vessel_mask_dice=parts.get('aux_vessel_mask_dice', 0.0), aux_vessel_mask_dice_raw=parts.get('aux_vessel_mask_dice_raw', 0.0), aux_vessel_pred_mask_frame_l1=parts.get('aux_vessel_pred_mask_frame_l1', 0.0), aux_vessel_pred_mask_frame_l1_weighted=parts.get('aux_vessel_pred_mask_frame_l1_weighted', 0.0), aux_vessel_pred_mean=parts.get('aux_vessel_pred_mean', 0.0), aux_vessel_target_mean=parts.get('aux_vessel_target_mean', 0.0), aux_vessel_available=parts.get('aux_vessel_available', 0.0), vessel_semantic_loss=parts.get('vessel_semantic_loss', 0.0), vessel_semantic_bce=parts.get('vessel_semantic_bce', 0.0), vessel_semantic_dice=parts.get('vessel_semantic_dice', 0.0), vessel_semantic_pred_mean=parts.get('vessel_semantic_pred_mean', 0.0), vessel_semantic_target_mean=parts.get('vessel_semantic_target_mean', 0.0), g_adv_loss=parts.get('g_adv_loss', 0.0), d_loss=parts.get('d_loss', 0.0), teacher_forcing_ratio=tf_ratio, teacher_forcing=step_teacher_forcing, lr=optimizer.param_groups[0]['lr'])
        metrics = {'train_loss': sum(running) / len(running)}
        if val_loader is not None:
            if use_sequence_training:
                (val_seq_metrics, _) = evaluate_sequence(model, val_loader, cfg, device)
                add_prefixed_metrics(metrics, 'val_seq', val_seq_metrics)
                metric_text = f"train {metrics['train_loss']:.5f}, val_seq {val_seq_metrics['loss']:.5f}, PSNR {val_seq_metrics['psnr']:.2f}, SSIM {val_seq_metrics['ssim']:.4f}"
            else:
                (val_tf_metrics, _) = evaluate(model, val_loader, cfg, device, teacher_forcing=True)
                (val_ar_metrics, _) = evaluate(model, val_ar_loader, cfg, device, teacher_forcing=False)
                add_prefixed_metrics(metrics, 'val_tf', val_tf_metrics)
                add_prefixed_metrics(metrics, 'val_ar', val_ar_metrics)
                metric_text = f"train {metrics['train_loss']:.5f}, val_tf {val_tf_metrics['loss']:.5f}, PSNR {val_tf_metrics['psnr']:.2f}, SSIM {val_tf_metrics['ssim']:.4f}, val_ar {val_ar_metrics['loss']:.5f}, PSNR {val_ar_metrics['psnr']:.2f}, SSIM {val_ar_metrics['ssim']:.4f}"
        else:
            metric_text = f"train {metrics['train_loss']:.5f}"
        epoch_seconds = time.time() - epoch_start_time
        completed_epoch_times.append(epoch_seconds)
        avg_epoch_seconds = sum(completed_epoch_times) / len(completed_epoch_times)
        remaining_epochs = total_epochs - epoch
        eta_seconds = remaining_epochs * avg_epoch_seconds
        elapsed_seconds = time.time() - train_start_time
        logger.info('Epoch %03d/%03d: %s | epoch_time %s | elapsed %s | eta %s', epoch, total_epochs, metric_text, format_duration(epoch_seconds), format_duration(elapsed_seconds), format_duration(eta_seconds))
        jsonl_logger.write('val_epoch', step=global_step, epoch=epoch, total_epochs=total_epochs, **metrics, epoch_seconds=epoch_seconds, elapsed_seconds=elapsed_seconds, eta_seconds=eta_seconds, epoch_time=format_duration(epoch_seconds), elapsed=format_duration(elapsed_seconds), eta=format_duration(eta_seconds))
        (selection_metric_name, score, selection_metric_mode) = best_checkpoint_metric(metrics, train_cfg)
        latest = {'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'discriminator': None, 'optimizer_d': None, 'patch_feature_encoder': None, 'scaler': scaler.state_dict() if scaler.is_enabled() else None, 'config': cfg, 'splits': splits, 'metrics': metrics, 'global_step': global_step, 'finetune_from': str(finetune_path) if finetune_path is not None else None, 'selection_metric': {'name': selection_metric_name, 'mode': selection_metric_mode, 'score': score}}
        torch.save(latest, ckpt_dir / 'latest.pt')
        is_better = score < best_val if selection_metric_mode == 'min' else score > best_val
        if epoch >= best_start_epoch and is_better:
            best_val = score
            torch.save(latest, ckpt_dir / 'best.pt')
            logger.info('Saved new best checkpoint: %s=%.6f (%s) at epoch %03d', selection_metric_name, score, selection_metric_mode, epoch)
        elif epoch < best_start_epoch:
            logger.info('Skip best checkpoint before epoch %d: epoch=%03d %s=%.6f (%s)', best_start_epoch, epoch, selection_metric_name, score, selection_metric_mode)
        preview_every = int(train_cfg.get('preview_every', 5))
        if val_loader is not None and preview_every > 0 and (epoch % preview_every == 0):
            save_epoch_sequence_preview(model, val_set, cfg, device, preview_dir, epoch, logger)
    jsonl_logger.close()
if __name__ == '__main__':
    main()
