from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import torch
import torch.nn as nn

from .base import BaseAdapter, register_adapter, ensure_btp88, clamp_prob


def _first_existing(d: Dict[str, Any], candidates: Sequence[str]) -> Optional[str]:
    for k in candidates:
        if k in d:
            return k
    return None


@dataclass
class HPTKeySpec:
    # For dict output: unified_name -> dict_key
    dict_keys: Optional[Dict[str, str]] = None
    tuple_index: Optional[Dict[str, int]] = None  # no longer used (dict-only)
    # Optional: unified_name -> bool (True=logits, False=prob)
    assume_logits: Optional[Dict[str, bool]] = None


@register_adapter("hpt")
class HPTAdapter(BaseAdapter):
    """
    HPT-like adapter (velocity-only):
    - 假定基模型输出为 dict
    - 可自动猜测常见 vel/vel_logits key，或用 keyspec.dict_keys 指定

    returns:
      {"vel":..., "vel_logits":..., "extra":...}
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        cfg=None,
        keyspec: Optional[Dict[str, Any]] = None,
        keep_extra: bool = True,
        vel_candidates: Optional[List[str]] = None,
        vel_logit_candidates: Optional[List[str]] = None,
    ):
        if model is None:
            if cfg is None:
                raise ValueError("HPTAdapter: cfg is required when model is None.")
            from model_HPT import Single_Velocity_HPT  # local import to avoid circulars
            model = Single_Velocity_HPT(cfg)
        super().__init__(model=model)
        ks = keyspec or {}
        self.keyspec = HPTKeySpec(
            dict_keys=ks.get("dict_keys", None),
            tuple_index=None,
            assume_logits=ks.get("assume_logits", None),
        )
        self.keep_extra = keep_extra

        self.vel_candidates = vel_candidates or [
            "vel", "velocity", "velocity_output", "velocity_pred", "velocity_roll",
            "reg_velocity_output", "vel_output"
        ]
        self.vel_logit_candidates = vel_logit_candidates or [
            "vel_logits", "velocity_logits", "velocity_logit", "logits_velocity"
        ]

    def _guess_is_logits(self, key: Optional[str]) -> bool:
        if not key:
            return False
        k = key.lower()
        return ("logit" in k) or ("logits" in k)

    def _get_forced_key(self, unified: str) -> Optional[str]:
        if self.keyspec.dict_keys and unified in self.keyspec.dict_keys:
            return self.keyspec.dict_keys[unified]
        return None

    def _from_dict(self, out: Dict[str, Any]) -> Dict[str, Any]:
        vel_k = self._get_forced_key("vel") or _first_existing(out, self.vel_candidates)
        vel_logit_k = self._get_forced_key("vel_logits") or _first_existing(out, self.vel_logit_candidates)

        def get_tensor(k: Optional[str]) -> Optional[torch.Tensor]:
            if k is None:
                return None
            v = out.get(k, None)
            return v if torch.is_tensor(v) else None

        vel_raw = get_tensor(vel_k)
        vel_logit_raw = get_tensor(vel_logit_k)

        assume = self.keyspec.assume_logits or {}
        vel_is_logits = assume.get("vel", None)  # None -> guess by name
        vel_logit_is_logits = assume.get("vel_logits", True)

        vel = None
        vel_logits = None

        if vel_raw is not None:
            vel_raw = ensure_btp88(vel_raw, "vel")
            is_logits = self._guess_is_logits(vel_k) if vel_is_logits is None else bool(vel_is_logits)
            vel = torch.sigmoid(vel_raw) if is_logits else torch.clamp(vel_raw, 0.0, 1.0)

        if vel_logit_raw is not None:
            vel_logit_raw = ensure_btp88(vel_logit_raw, "vel_logits")
            # if user says it's prob, convert; else keep
            is_logits = True if vel_logit_is_logits is None else bool(vel_logit_is_logits)
            vel_logits = vel_logit_raw if is_logits else None  # if not logits, we'll let _finalize derive

        extra: Dict[str, torch.Tensor] = {}
        if self.keep_extra:
            used = set(k for k in [vel_k, vel_logit_k] if k)
            for k, v in out.items():
                if k in used:
                    continue
                if torch.is_tensor(v):
                    extra[k] = v

        return {"vel": vel, "vel_logits": vel_logits, "extra": extra}

    def forward(self, audio: torch.Tensor, *args, **kwargs) -> Dict[str, Any]:
        out = self.model(audio, *args, **kwargs)
        return self._finalize(self._from_dict(out))
