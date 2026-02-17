from __future__ import annotations

from typing import Dict, Type
import torch.nn as nn

from .hpt_adapter import Single_Velocity_HPT
from .dynest_adapter import DynestAudioCNN
from .hppnet_adapter import HPPNet_SP

MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "Single_Velocity_HPT": Single_Velocity_HPT,
    "DynestAudioCNN": DynestAudioCNN,
    "HPPNet_SP": HPPNet_SP,
}

MODEL_ALIASES = {
    "HPT": "Single_Velocity_HPT",
    "HPPNet": "HPPNet_SP",
    "Dynest": "DynestAudioCNN",
}


def build_model(cfg) -> nn.Module:
    name = cfg.model.name
    if name in {"FiLMUNetPretrained", "FiLMUNet"}:
        raise KeyError("FiLMUNet 仅用于 inference/test，不在 train_score_inf 框架内。")
    name = MODEL_ALIASES.get(name, name)
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown base model '{cfg.model.name}'. "
            f"Available: {list(MODEL_REGISTRY.keys())} "
            f"(aliases: {list(MODEL_ALIASES.keys())})"
        )
    return MODEL_REGISTRY[name](cfg)
