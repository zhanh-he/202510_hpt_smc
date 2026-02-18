# pytorch/train_score_inf.py
from __future__ import annotations

import os
import sys
import time
import logging
import torch
from torch.optim import Adam, AdamW
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, Any, Optional, Tuple
from hydra import initialize, compose
from omegaconf import OmegaConf
import wandb

from pytorch_utils import move_data_to_device
from data_generator import (Maestro_Dataset, SMD_Dataset, MAPS_Dataset,
     Sampler, EvalSampler, collate_fn)
from utilities import create_folder, create_logging, get_model_name
from losses import get_loss_func
from evaluate import SegmentEvaluator

from model import build_adapter
from score_inf import build_score_inf
from score_inf.wrapper import ScoreInfWrapper


def init_wandb(cfg):
    """Initialize WandB for experiment tracking if configured."""
    if not hasattr(cfg, "wandb"):
        return
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )


def _note_segments(active_mask: np.ndarray) -> list:
    padded = np.pad(active_mask.astype(np.int8), (1, 1), mode="constant")
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def _post_process_rolls(
    pred_vis: np.ndarray,
    target_raw: np.ndarray,
    onset_roll: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    frames, keys = target_raw.shape
    frame_pick_roll = np.zeros_like(pred_vis, dtype=np.float32)
    onset_pick_roll = np.zeros_like(pred_vis, dtype=np.float32)

    for key in range(keys):
        note_active = target_raw[:, key] > 0
        for start, end in _note_segments(note_active):
            if end <= start:
                continue
            pred_note = pred_vis[start:end, key].copy()
            pred_note[pred_note <= 1e-4] = 0.0

            frame_val = float(np.max(pred_note)) if pred_note.size else 0.0
            frame_pick_roll[start:end, key] = frame_val

            onset_mask = onset_roll[start:end, key] > 0
            if np.any(onset_mask):
                onset_val = float(np.max(pred_note[onset_mask]))
            else:
                onset_val = 0.0
            onset_pick_roll[start:end, key] = onset_val

    return frame_pick_roll, onset_pick_roll


def log_velocity_rolls(cfg, iteration, batch_output_dict, batch_data_dict):
    """Log prediction vs target velocity rolls to WandB at configured intervals."""
    interval = getattr(cfg.wandb, "log_velocity_interval", None) if hasattr(cfg, "wandb") else None
    if not interval or interval <= 0 or wandb.run is None:
        return
    if iteration % interval != 0:
        return

    pred = batch_output_dict.get("velocity_output")
    if pred is None:
        pred = batch_output_dict.get("vel_corr")
    target = batch_data_dict.get("velocity_roll")
    onset = batch_data_dict.get("onset_roll")
    if pred is None or target is None:
        return

    velocity_scale = getattr(cfg.feature, "velocity_scale", 128)
    pred_img = pred[0].detach().cpu().numpy()
    pred_max = float(np.max(pred_img))
    pred_min = float(np.min(pred_img))
    if pred_max > 1.0 + 1e-3 or pred_min < -1e-3:
        pred_vis = np.clip(pred_img / velocity_scale, 0.0, 1.0)
    else:
        pred_vis = np.clip(pred_img, 0.0, 1.0)
    target_raw = target[0].detach().cpu().numpy()
    if float(np.max(target_raw)) <= 1.0 + 1e-3:
        target_raw = target_raw * velocity_scale
    target_img = np.clip(target_raw / velocity_scale, 0.0, 1.0)
    onset_roll = onset[0].detach().cpu().numpy() if onset is not None else np.zeros_like(target_raw)

    frame_pick_img, onset_pick_img = _post_process_rolls(
        pred_vis=pred_vis,
        target_raw=target_raw,
        onset_roll=onset_roll,
    )


    fig, axes = plt.subplots(2, 2, figsize=(20, 4))
    specs = [
        ("Ground Truth", target_img),
        ("Prediction", pred_vis),
        ("Post-Proc Frame-Pick", frame_pick_img),
        ("Post-Proc Onset-Pick", onset_pick_img),
    ]
    for ax, (title, data) in zip(np.ravel(axes), specs):
        im = ax.imshow(
            data.T,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(title)
        ax.set_xlabel("Frame")
        ax.set_ylabel("Key")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Velocity roll @ iter {iteration}")
    fig.tight_layout()

    wandb.log(
        {
            "velocity_roll_comparison": wandb.Image(fig),
        },
        step=iteration,
    )
    plt.close(fig)


def _select_velocity_metrics(statistics: Dict[str, float]) -> Dict[str, float]:
    keep_keys = (
        "frame_max_error",
        "frame_max_std",
        "onset_masked_error",
        "onset_masked_std",
    )
    return {k: statistics[k] for k in keep_keys if k in statistics}


def _write_training_stats(cfg, checkpoints_dir: str, model_name: str) -> None:
    stats_path = os.path.join(checkpoints_dir, "training_stats.txt")
    file_name = getattr(cfg.wandb, "name", None) if hasattr(cfg, "wandb") else None
    file_name = file_name or model_name

    condition_inputs = [cfg.model.input2, cfg.model.input3]
    condition_selected = [c for c in condition_inputs if c]
    condition_check = bool(condition_selected)
    condition_type = "+".join(condition_selected) if condition_selected else "default"
    condition_net = "N/A"

    score_cfg = getattr(cfg, "score_informed", None)
    score_method = getattr(score_cfg, "method", "direct_output") if score_cfg is not None else "direct_output"
    train_mode = getattr(score_cfg, "train_mode", "joint") if score_cfg is not None else "joint"
    switch_iteration = getattr(score_cfg, "switch_iteration", 100000) if score_cfg is not None else 100000
    if score_method == "direct_output":
        condition_check = False
        condition_type = "ignored"

    lines = [
        f"file name           :{file_name}",
        f"dev_env             :{getattr(cfg.exp, 'dev_env', 'local')}",
        f"condition_check     :{condition_check}",
        f"condition_net       :{condition_net}",
        f"loss_type           :{getattr(cfg.loss, 'loss_type', getattr(cfg.exp, 'loss_type', ''))}",
        f"condition_type      :{condition_type}",
        f"batch_size          :{cfg.exp.batch_size}",
        f"hop_seconds         :{cfg.feature.hop_seconds}",
        f"segment_seconds     :{cfg.feature.segment_seconds}",
        f"frames_per_second   :{cfg.feature.frames_per_second}",
        f"feature type        :{cfg.feature.audio_feature}",
        f"score_inf_method    :{score_method}",
        f"train_mode          :{train_mode}",
        f"switch_iteration    :{switch_iteration}",
    ]

    with open(stats_path, "w") as f:
        f.write("\n".join(lines))

def _select_input_conditions(cfg) -> list:
    cond_selected = []
    for key in [cfg.model.input2, cfg.model.input3]:
        if key and key not in cond_selected:
            cond_selected.append(key)
    return cond_selected


def _resolve_score_inf_conditioning(cfg, method: str, params: Dict[str, Any]) -> Tuple[Dict[str, Any], list]:
    cond_selected = _select_input_conditions(cfg)
    merged = dict(params)

    if method == "direct_output":
        return merged, []

    if method == "note_editor":
        use_cond_feats = [cfg.model.input3] if cfg.model.input3 else []
        merged["use_cond_feats"] = use_cond_feats
        cond_keys = ["onset"] + use_cond_feats
        return merged, cond_keys

    if method in ("bilstm", "scrr", "dual_gated"):
        merged["cond_keys"] = cond_selected
        return merged, cond_selected

    return merged, cond_selected


def _required_target_rolls(loss_type: str) -> list:
    if loss_type in ("velocity_bce", "velocity_mse"):
        return ["velocity_roll", "onset_roll"]
    if loss_type == "kim_bce_l1":
        return ["velocity_roll", "onset_roll", "frame_roll"]
    if loss_type == "score_inf_custom":
        return ["velocity_roll", "onset_roll"]
    raise ValueError(f"Unknown loss_type: {loss_type}")


def _resolve_train_schedule(cfg, score_cfg) -> Tuple[str, int]:
    if score_cfg is None:
        return "joint", 100000
    if isinstance(score_cfg, dict):
        mode = score_cfg.get("train_mode", "joint")
        switch_iteration = int(score_cfg.get("switch_iteration", 100000))
    else:
        mode = getattr(score_cfg, "train_mode", "joint")
        switch_iteration = int(getattr(score_cfg, "switch_iteration", 100000))
    return mode, switch_iteration


def _phase_at_iteration(train_mode: str, iteration: int, switch_iteration: int) -> str:
    if train_mode == "joint":
        return "joint"
    if iteration < switch_iteration:
        return "adapter_only"
    if train_mode == "adapter_then_score":
        return "score_only"
    if train_mode == "adapter_then_joint":
        return "joint"
    raise ValueError(f"Unknown score_informed.train_mode: {train_mode}")


def _apply_train_phase(model: ScoreInfWrapper, phase: str) -> None:
    if phase == "adapter_only":
        model.freeze_base = False
        for p in model.base_adapter.parameters():
            p.requires_grad = True
        for p in model.post.parameters():
            p.requires_grad = False
        return
    if phase == "score_only":
        model.freeze_base = True
        for p in model.base_adapter.parameters():
            p.requires_grad = False
        for p in model.post.parameters():
            p.requires_grad = True
        return
    if phase == "joint":
        model.freeze_base = False
        for p in model.base_adapter.parameters():
            p.requires_grad = True
        for p in model.post.parameters():
            p.requires_grad = True
        return
    raise ValueError(f"Unknown phase: {phase}")


def build_dataloaders(cfg):
    def get_sampler(cfg, purpose: str, split: str, is_eval: Optional[str] = None):
        sampler_mapping = {
            "train": Sampler,
            "eval": EvalSampler,
        }
        return sampler_mapping[purpose](cfg, split=split, is_eval=is_eval)
    dataset_classes = {
        "maestro": Maestro_Dataset,
        "smd": SMD_Dataset,
        "maps": MAPS_Dataset,
    }
    train_dataset = dataset_classes[cfg.dataset.train_set](cfg)
    train_sampler = get_sampler(cfg, purpose="train", split="train", is_eval=None)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )
    eval_train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_sampler=get_sampler(cfg, purpose="eval", split="train", is_eval=None),
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )
    eval_maestro_loader = torch.utils.data.DataLoader(
        dataset=dataset_classes["maestro"](cfg),
        batch_sampler=get_sampler(cfg, purpose="eval", split="test", is_eval="maestro"),
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )
    eval_smd_loader = torch.utils.data.DataLoader(
        dataset=dataset_classes["smd"](cfg),
        batch_sampler=get_sampler(cfg, purpose="eval", split="test", is_eval="smd"),
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )
    eval_maps_loader = torch.utils.data.DataLoader(
        dataset=dataset_classes["maps"](cfg),
        batch_sampler=get_sampler(cfg, purpose="eval", split="test", is_eval="maps"),
        collate_fn=collate_fn,
        num_workers=cfg.exp.num_workers,
        pin_memory=True,
    )
    eval_loaders = {
        "train": eval_train_loader,
        "maestro": eval_maestro_loader,
        "smd": eval_smd_loader,
        "maps": eval_maps_loader,
    }
    return train_loader, eval_loaders


def _prepare_batch(
    batch_data_dict,
    device: torch.device,
    cond_keys: list,
    target_rolls: list,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    audio = move_data_to_device(batch_data_dict["waveform"], device)
    cond = {k: move_data_to_device(batch_data_dict[f"{k}_roll"], device) for k in cond_keys}
    batch_torch = {k: move_data_to_device(batch_data_dict[k], device) for k in target_rolls}
    return audio, cond, batch_torch


def train(cfg):
    device = torch.device("cuda") if cfg.exp.cuda and torch.cuda.is_available() else torch.device("cpu")

    model_cfg = {"type": cfg.model.type, "params": cfg.model.params}
    adapter = build_adapter(model_cfg, model=None, cfg=cfg).to(device)

    score_cfg = getattr(cfg, "score_informed", None)
    if score_cfg is None:
        method = "direct_output"
        params = {}
    else:
        if OmegaConf.is_config(score_cfg):
            score_cfg = OmegaConf.to_container(score_cfg, resolve=True)
        if isinstance(score_cfg, dict):
            method = score_cfg.get("method", "direct_output") or "direct_output"
            params = score_cfg.get("params", {}) or {}
        else:
            method = getattr(score_cfg, "method", "direct_output") or "direct_output"
            params = getattr(score_cfg, "params", {}) or {}

    params, cond_keys = _resolve_score_inf_conditioning(cfg, method, params)
    target_rolls = _required_target_rolls(cfg.loss.loss_type)
    train_mode, switch_iteration = _resolve_train_schedule(cfg, score_cfg)
    post = build_score_inf(method, params).to(device)
    model = ScoreInfWrapper(adapter, post, freeze_base=False).to(device)

    # Paths for results
    model_name = get_model_name(cfg)
    if method and method != "direct_output":
        model_name = f"{model_name}+score_{method}"
    checkpoints_dir = os.path.join(cfg.exp.workspace, "checkpoints", model_name)
    logs_dir = os.path.join(cfg.exp.workspace, "logs", model_name)

    create_folder(checkpoints_dir)
    create_folder(logs_dir)
    _write_training_stats(cfg, checkpoints_dir, model_name)
    create_logging(logs_dir, filemode="w")
    logging.info(cfg)
    logging.info(f"Using {device}.")

    start_iteration = 0
    init_phase = _phase_at_iteration(train_mode, start_iteration, switch_iteration)
    _apply_train_phase(model, init_phase)

    # Match legacy seed tweak
    # if cfg.model.type == "hpt":
    #     cfg.exp.random_seed = 12

    train_loader, eval_loaders = build_dataloaders(cfg)
    # for split, loader in eval_loaders.items():
    #     max_eval_iters = getattr(loader.batch_sampler, "max_evaluate_iteration", None)
    #     logging.info(f"Fast eval sampler [{split}] max_evaluate_iteration={max_eval_iters}")

    # Optimizer
    optim_cfg = getattr(cfg, "optim", None)
    if optim_cfg is not None and OmegaConf.is_config(optim_cfg):
        optim_cfg = OmegaConf.to_container(optim_cfg, resolve=True)
    if isinstance(optim_cfg, dict):
        opt_name = str(optim_cfg.get("name", "adamw")).lower()
        lr = float(optim_cfg.get("lr", cfg.exp.learning_rate))
        wd = float(optim_cfg.get("weight_decay", 0.0))
    else:
        opt_name = str(getattr(cfg.exp, "optim", "adam")).lower()
        lr = float(getattr(cfg.exp, "learning_rate", 1e-4))
        wd = float(getattr(cfg.exp, "weight_decay", 0.0))

    params = list(model.parameters())
    if opt_name == "adamw":
        optimizer = AdamW(params, lr=lr, weight_decay=wd)
    else:
        optimizer = Adam(params, lr=lr, weight_decay=wd)

    init_wandb(cfg)

    # GPU info
    gpu_count = torch.cuda.device_count()
    logging.info(f"Number of GPUs available: {gpu_count}")
    if gpu_count > 1:
        torch.cuda.set_device(0)
    model.to(device)

    iteration = start_iteration
    train_bgn_time = time.time()
    train_loss = 0.0
    train_loss_steps = 0

    early_phase = 0
    early_step = int(early_phase * 0.1) if early_phase > 0 else 0

    evaluator = SegmentEvaluator(model, cfg, score_inf=True)
    loss_fn = get_loss_func(cfg=cfg)
    current_phase = None

    for batch_data_dict in train_loader:
        phase = _phase_at_iteration(train_mode, iteration, switch_iteration)
        if phase != current_phase:
            _apply_train_phase(model, phase)
            current_phase = phase
            logging.info(f"Train phase switched to: {phase} at iteration {iteration}")

        if cfg.exp.decay:
            if iteration % cfg.exp.reduce_iteration == 0 and iteration != 0:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.9

        model.train()
        audio, cond, batch_torch = _prepare_batch(batch_data_dict, device, cond_keys, target_rolls)
        out = model(audio, cond)
        loss = loss_fn(cfg, out, batch_torch, cond_dict=cond)

        print(iteration, loss)
        train_loss += loss.item()
        train_loss_steps += 1
        log_velocity_rolls(cfg, iteration, {"velocity_output": out["vel_corr"]}, batch_torch)

        loss.backward()
        optimizer.step()

        if ((iteration < early_phase and early_step > 0 and iteration % early_step == 0) or
            (iteration >= early_phase and iteration % cfg.exp.eval_iteration == 0)):
            logging.info("------------------------------------")
            logging.info(f"Iteration: {iteration}/{cfg.exp.total_iteration}")
            train_fin_time = time.time()
            avg_train_loss = None
            if train_loss_steps > 0 and iteration != 0:
                avg_train_loss = train_loss / train_loss_steps

            train_stats = _select_velocity_metrics(evaluator.evaluate(eval_loaders["train"]))
            maestro_stats = _select_velocity_metrics(evaluator.evaluate(eval_loaders["maestro"]))
            smd_stats = _select_velocity_metrics(evaluator.evaluate(eval_loaders["smd"]))
            maps_stats = _select_velocity_metrics(evaluator.evaluate(eval_loaders["maps"]))

            if avg_train_loss is not None:
                logging.info(f"    Train Loss: {avg_train_loss:.4f}")
            logging.info(f"    Train Stat: {train_stats}")
            logging.info(f"    Valid Maestro Stat: {maestro_stats}")
            logging.info(f"    Valid SMD Stat: {smd_stats}")
            logging.info(f"    Valid MAPS Stat: {maps_stats}")

            log_payload = {
                "iteration": iteration,
                "train_stat": train_stats,
                "valid_maestro_stat": maestro_stats,
                "valid_smd_stat": smd_stats,
                "valid_maps_stat": maps_stats,
            }
            if avg_train_loss is not None:
                log_payload["train_loss"] = avg_train_loss
            if wandb.run is not None:
                wandb.log(log_payload)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time
            logging.info(
                "Train time: {:.3f} s, validate time: {:.3f} s".format(train_time, validate_time)
            )

            train_loss = 0.0
            train_loss_steps = 0
            train_bgn_time = time.time()

            checkpoint_path = os.path.join(checkpoints_dir, f"{iteration}_iterations.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logging.info(f"Model saved to {checkpoint_path}")

        if iteration == cfg.exp.total_iteration:
            break

        optimizer.zero_grad(set_to_none=True)
        iteration += 1

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    initialize(config_path="./", job_name="train_score_inf", version_base=None)
    cfg = compose(config_name="config", overrides=sys.argv[1:])
    train(cfg)
