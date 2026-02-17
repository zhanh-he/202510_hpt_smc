# pytorch/train_score_inf.py
from __future__ import annotations

import os
import sys
import time
import logging
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
from torch.optim import Adam, AdamW
from hydra import initialize, compose
from omegaconf import OmegaConf
import wandb

from pytorch_utils import move_data_to_device, append_to_dict
from data_generator import (
    Maestro_Dataset,
    SMD_Dataset,
    MAPS_Dataset,
    Sampler,
    EvalSampler,
    collate_fn,
)
from utilities import create_folder, create_logging, get_model_name
from losses import compute_loss
from evaluate import _segments_from_output, _kim_metrics_from_segments

from amt_modules import build_adapter
from score_inf import build_score_inf
from score_inf.wrapper import ScoreInfWrapper


def init_wandb(cfg, wandb_run_id: Optional[str]):
    """Initialize WandB for experiment tracking if configured."""
    if not hasattr(cfg, "wandb"):
        return
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        id=wandb_run_id,
        resume="must" if wandb_run_id else "allow",
        config=OmegaConf.to_container(cfg, resolve=True),
    )


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
    target_img = np.clip(target[0].detach().cpu().numpy() / velocity_scale, 0.0, 1.0)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    specs = [
        ("Prediction", pred_vis),
        ("Ground Truth", target_img),
    ]
    for ax, (title, data) in zip(axes, specs):
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

    wandb.log({"velocity_roll_comparison": wandb.Image(fig)}, step=iteration)
    plt.close(fig)


def _select_velocity_metrics(statistics: Dict[str, float]) -> Dict[str, float]:
    keep_keys = ("velocity_mae", "velocity_std")
    return {k: statistics[k] for k in keep_keys if k in statistics}


def _write_training_stats(cfg, checkpoints_dir: str, model_name: str) -> None:
    stats_path = os.path.join(checkpoints_dir, "training_stats.txt")
    file_name = getattr(cfg.wandb, "name", None) if hasattr(cfg, "wandb") else None
    file_name = file_name or model_name

    condition_inputs = [getattr(cfg.model, "input2", None), getattr(cfg.model, "input3", None)]
    condition_check = any(condition_inputs)
    condition_type = next((c for c in condition_inputs if c), "none")
    condition_net = getattr(cfg.model, "condition_net", "N/A")

    score_cfg = getattr(cfg, "score_informed", None)
    score_method = getattr(score_cfg, "method", "direct_output") if score_cfg is not None else "direct_output"
    freeze_base = getattr(score_cfg, "freeze_base", False) if score_cfg is not None else False
    base_ckpt = getattr(score_cfg, "base_checkpoint", "") if score_cfg is not None else ""

    lines = [
        f"file name           :{file_name}",
        f"dev_env             :{getattr(cfg.exp, 'dev_env', 'local')}",
        f"condition_check     :{condition_check}",
        f"condition_net       :{condition_net}",
        f"loss_type           :{cfg.exp.loss_type}",
        f"condition_type      :{condition_type}",
        f"batch_size          :{cfg.exp.batch_size}",
        f"hop_seconds         :{cfg.feature.hop_seconds}",
        f"segment_seconds     :{cfg.feature.segment_seconds}",
        f"frames_per_second   :{cfg.feature.frames_per_second}",
        f"feature type        :{cfg.feature.audio_feature}",
        f"score_inf_method    :{score_method}",
        f"freeze_base         :{freeze_base}",
        f"base_checkpoint     :{base_ckpt}",
    ]

    with open(stats_path, "w") as f:
        f.write("\n".join(lines))


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    keys = list(state.keys())
    if keys and all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def _load_state_dict(ckpt_path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Unsupported checkpoint format in {ckpt_path}")

    checkpoint = _strip_prefix(checkpoint, "module.")
    return checkpoint


def _resolve_base_checkpoint(cfg) -> Optional[str]:
    score_cfg = getattr(cfg, "score_informed", None)
    if score_cfg is None:
        return None

    base_ckpt = getattr(score_cfg, "base_checkpoint", "")
    if base_ckpt:
        return base_ckpt

    base_iter = getattr(score_cfg, "base_iteration", "")
    if base_iter:
        model_name = get_model_name(cfg)
        return os.path.join(cfg.exp.workspace, "checkpoints", model_name, f"{base_iter}_iterations.pth")

    model_ckpt = getattr(cfg.model, "base_checkpoint", "")
    if model_ckpt:
        return model_ckpt

    return None


def build_base_model(cfg) -> torch.nn.Module:
    """Instantiate base AMT model and load checkpoint if provided."""
    # For hpt/hppnet adapters we allow model=None; but checkpoints might be loaded later via adapter.model
    model = None

    ckpt_path = _resolve_base_checkpoint(cfg)
    if ckpt_path:
        if model is None:
            logging.warning("cfg.score_informed.base_checkpoint provided but base model will be loaded inside adapter.")
        else:
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(f"Base checkpoint not found: {ckpt_path}")
            state_dict = _load_state_dict(ckpt_path, device=torch.device("cpu"))
            model.load_state_dict(state_dict, strict=True)
            logging.info(f"Loaded base checkpoint: {ckpt_path}")

    return model


def _normalize_adapter_cfg(cfg) -> Dict[str, Any]:
    adapter_cfg = getattr(cfg, "adapter", None)
    if adapter_cfg is None:
        return {"type": "hpt", "params": {}}

    if OmegaConf.is_config(adapter_cfg):
        adapter_cfg = OmegaConf.to_container(adapter_cfg, resolve=True)

    if not isinstance(adapter_cfg, dict):
        return {"type": "hpt", "params": {}}

    if "type" in adapter_cfg:
        params = adapter_cfg.get("params", {}) or {}
        return {"type": adapter_cfg["type"], "params": params}

    if "keymap" in adapter_cfg or "transpose" in adapter_cfg:
        params = {
            "keymap": adapter_cfg.get("keymap", {}),
            "transpose": adapter_cfg.get("transpose", {}) or {},
            "keep_extra": adapter_cfg.get("keep_extra", True),
        }
        return {"type": "keymap", "params": params}

    return {"type": "hpt", "params": {}}


def get_sampler(cfg, purpose: str, split: str, is_eval: Optional[str] = None):
    sampler_mapping = {
        "train": Sampler,
        "eval": EvalSampler,
    }
    return sampler_mapping[purpose](cfg, split=split, is_eval=is_eval)


def build_dataloaders(cfg, resume_sampler_state=None):
    dataset_classes = {
        "maestro": Maestro_Dataset,
        "smd": SMD_Dataset,
        "maps": MAPS_Dataset,
    }

    train_dataset = dataset_classes[cfg.dataset.train_set](cfg)

    train_sampler = get_sampler(cfg, purpose="train", split="train", is_eval=None)
    if resume_sampler_state is not None:
        train_sampler.load_state_dict(resume_sampler_state)

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

    return train_loader, train_sampler, eval_loaders


def _prepare_batch(cfg, batch_data_dict, device: torch.device) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
    batch_torch = {k: move_data_to_device(v, device) for k, v in batch_data_dict.items()}

    audio = batch_torch["waveform"]

    cond = {
        "onset": batch_torch["onset_roll"],
        "frame": batch_torch.get("frame_roll"),
        "exframe": batch_torch.get("exframe_roll"),
    }

    base_inputs = []
    if getattr(cfg.model, "input2", None):
        base_inputs.append(batch_torch[f"{cfg.model.input2}_roll"])
    if getattr(cfg.model, "input3", None):
        base_inputs.append(batch_torch[f"{cfg.model.input3}_roll"])

    return audio, cond, batch_torch, tuple(base_inputs)


def _score_inf_forward_dataloader(model, dataloader, cfg, device):
    output_dict: Dict[str, Any] = {}
    for batch_data_dict in dataloader:
        audio, cond, _, base_inputs = _prepare_batch(cfg, batch_data_dict, device)

        with torch.no_grad():
            model.eval()
            out = model(audio, cond, *base_inputs)

        vel = out["vel_corr"]
        append_to_dict(output_dict, "velocity_output", vel.data.cpu().numpy())

        for target_type in batch_data_dict.keys():
            if "roll" in target_type or "reg_distance" in target_type or "reg_tail" in target_type:
                append_to_dict(output_dict, target_type, batch_data_dict[target_type])

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict


def _evaluate_score_inf(model, dataloader, cfg, device) -> Dict[str, float]:
    output_dict = _score_inf_forward_dataloader(model, dataloader, cfg, device)
    segments, targets = _segments_from_output(output_dict)
    if not segments:
        return {}
    return _kim_metrics_from_segments(segments, targets)


def train(cfg):
    device = torch.device("cuda") if cfg.exp.cuda and torch.cuda.is_available() else torch.device("cpu")

    base_model = build_base_model(cfg)
    if base_model is not None:
        base_model = base_model.to(device)

    adapter_cfg = _normalize_adapter_cfg(cfg)
    adapter = build_adapter(adapter_cfg, base_model, cfg=cfg).to(device)

    score_cfg = getattr(cfg, "score_informed", None)
    if score_cfg is None:
        method = "direct_output"
        params = {}
        freeze_base = False
    else:
        if OmegaConf.is_config(score_cfg):
            score_cfg = OmegaConf.to_container(score_cfg, resolve=True)
        if isinstance(score_cfg, dict):
            method = score_cfg.get("method", "direct_output") or "direct_output"
            params = score_cfg.get("params", {}) or {}
            freeze_base = bool(score_cfg.get("freeze_base", False))
        else:
            method = getattr(score_cfg, "method", "direct_output") or "direct_output"
            params = getattr(score_cfg, "params", {}) or {}
            freeze_base = bool(getattr(score_cfg, "freeze_base", False))

    post = build_score_inf(method, params).to(device)

    if freeze_base:
        for p in base_model.parameters():
            p.requires_grad = False

    model = ScoreInfWrapper(adapter, post, freeze_base=freeze_base).to(device)

    model.kim_loss_alpha = getattr(cfg.exp, "kim_loss_alpha", 0.5)

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

    # Resume training if applicable
    start_iteration = 0
    wandb_run_id = None
    resume_sampler_state = None
    resume_optimizer_state = None
    if cfg.exp.resume_iteration > 0:
        state_path = os.path.join(checkpoints_dir, f"{cfg.exp.resume_iteration}_iterations.pth")
        resume_meta_path = os.path.join(checkpoints_dir, f"{cfg.exp.resume_iteration}_resume.pth")
        if os.path.exists(state_path):
            logging.info(f"Loading checkpoint {state_path} from iteration {cfg.exp.resume_iteration}")
            model_state = torch.load(state_path, map_location=device)
            if isinstance(model_state, dict) and "model" in model_state and "optimizer" in model_state:
                checkpoint = model_state
                model.load_state_dict(checkpoint["model"], strict=True)
                start_iteration = checkpoint["iteration"]
                wandb_run_id = checkpoint.get("wandb_run_id")
                resume_sampler_state = checkpoint.get("sampler")
                resume_optimizer_state = checkpoint.get("optimizer")
            else:
                if not os.path.exists(resume_meta_path):
                    raise FileNotFoundError(f"Missing resume metadata: {resume_meta_path}")
                checkpoint = torch.load(resume_meta_path, map_location=device)
                model.load_state_dict(model_state, strict=True)
                start_iteration = checkpoint["iteration"]
                wandb_run_id = checkpoint.get("wandb_run_id")
                resume_sampler_state = checkpoint.get("sampler")
                resume_optimizer_state = checkpoint.get("optimizer")
        else:
            logging.warning(f"Checkpoint {state_path} not found. Starting from scratch.")

    # Match legacy seed tweak
    if cfg.model.name == "Single_Velocity_HPT":
        cfg.exp.random_seed = 12

    # Build data loaders (use sampler state if resuming)
    train_loader, train_sampler, eval_loaders = build_dataloaders(cfg, resume_sampler_state)

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

    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "adamw":
        optimizer = AdamW(params, lr=lr, weight_decay=wd)
    else:
        optimizer = Adam(params, lr=lr, weight_decay=wd)

    if resume_optimizer_state is not None:
        optimizer.load_state_dict(resume_optimizer_state)

    # Initialize WandB after potentially loading run_id
    init_wandb(cfg, wandb_run_id)

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

    for batch_data_dict in train_loader:
        if cfg.exp.decay:
            if iteration % cfg.exp.reduce_iteration == 0 and iteration != cfg.exp.resume_iteration:
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.9

        model.train()
        audio, cond, batch_torch, base_inputs = _prepare_batch(cfg, batch_data_dict, device)
        out = model(audio, cond, *base_inputs)
        loss = compute_loss(cfg, model, out, batch_torch, cond_dict=cond)

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

            train_stats = _select_velocity_metrics(_evaluate_score_inf(model, eval_loaders["train"], cfg, device))
            maestro_stats = _select_velocity_metrics(_evaluate_score_inf(model, eval_loaders["maestro"], cfg, device))
            smd_stats = _select_velocity_metrics(_evaluate_score_inf(model, eval_loaders["smd"], cfg, device))
            maps_stats = _select_velocity_metrics(_evaluate_score_inf(model, eval_loaders["maps"], cfg, device))

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
            resume_meta_path = os.path.join(checkpoints_dir, f"{iteration}_resume.pth")
            torch.save(model.state_dict(), checkpoint_path)
            resume_payload = {
                "iteration": iteration,
                "sampler": train_sampler.state_dict(),
                "optimizer": optimizer.state_dict(),
                "wandb_run_id": wandb.run.id if wandb.run is not None else None,
            }
            torch.save(resume_payload, resume_meta_path)
            logging.info(
                f"Model saved to {checkpoint_path} (state) and {resume_meta_path} (resume)"
            )

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
