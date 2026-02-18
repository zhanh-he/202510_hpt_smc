import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import torch
from pytorch_utils import move_data_to_device, append_to_dict
from calculate_scores import frame_max_metrics_from_list, onset_pick_metrics_from_list

def _segments_from_output(output_dict):
    """Convert batched output/target rolls to per-segment dicts used by Kim metrics."""
    velocity = output_dict.get('velocity_output')
    if velocity is None:
        return [], []
    vel_roll = output_dict.get('velocity_roll')
    frame_roll = output_dict.get('frame_roll')
    onset_roll = output_dict.get('onset_roll')
    pedal_roll = output_dict.get('pedal_frame_roll')

    segments = []
    targets = []
    segs = velocity.shape[0]
    for idx in range(segs):
        pred = velocity[idx]
        gt_vel = vel_roll[idx]
        frames = min(pred.shape[0], gt_vel.shape[0])
        seg_pred = {'velocity_output': pred[:frames]}
        segments.append(seg_pred)

        pedal = pedal_roll[idx] if pedal_roll is not None else np.zeros(frames)
        if pedal.ndim > 1:
            pedal = np.squeeze(pedal, axis=-1)
        target_entry = {
            'velocity_roll': gt_vel[:frames],
            'frame_roll': frame_roll[idx][:frames],
            'onset_roll': onset_roll[idx][:frames],
            'pedal_frame_roll': pedal[:frames],
        }
        targets.append(target_entry)
    return segments, targets


def _kim_metrics_from_segments(output_dict_list, target_dict_list):
    """Run the same Kim-style metrics used in calculate_scores."""
    if not output_dict_list or not target_dict_list:
        return {}
    frame_max_err, frame_max_std = frame_max_metrics_from_list(output_dict_list, target_dict_list)
    onset_masked_error, onset_masked_std = onset_pick_metrics_from_list(output_dict_list, target_dict_list)
    stats = {
        'frame_max_error': round(frame_max_err, 4),
        'frame_max_std': round(frame_max_std, 4),
        'onset_masked_error': round(onset_masked_error, 4),
        'onset_masked_std': round(onset_masked_std, 4),
    }
    return stats


class SegmentEvaluator(object):

    def __init__(self, model, cfg, score_inf: bool = False):
        """Evaluate segment-wise metrics.
        Args:
            model: nn.Module
            cfg: OmegaConf config
            score_inf: set True for ScoreInfWrapper (expects cond dict)
        """
        self.model = model
        self.input2 = cfg.model.input2
        self.input3 = cfg.model.input3
        self.score_inf = score_inf
        score_cfg = getattr(cfg, "score_informed", None)
        self.score_method = getattr(score_cfg, "method", "direct_output") if score_cfg is not None else "direct_output"
        self.score_cond_keys = self._resolve_score_cond_keys() if self.score_inf else []

    def _resolve_score_cond_keys(self):
        cond_selected = []
        for key in [self.input2, self.input3]:
            if key and key not in cond_selected:
                cond_selected.append(key)

        if self.score_method == "direct_output":
            return []
        if self.score_method == "note_editor":
            return ["onset"] + ([self.input3] if self.input3 else [])
        return cond_selected

    def _forward_score_inf(self, batch_data_dict, device):
        audio = move_data_to_device(batch_data_dict["waveform"], device)
        cond = {k: move_data_to_device(batch_data_dict[f"{k}_roll"], device) for k in self.score_cond_keys}

        with torch.no_grad():
            self.model.eval()
            out = self.model(audio, cond)

        if "velocity_output" not in out and "vel_corr" in out:
            out = dict(out)
            out["velocity_output"] = out["vel_corr"]
        return out

    def _forward_legacy(self, batch_data_dict, device):
        batch_input1 = move_data_to_device(batch_data_dict["waveform"], device)
        batch_input2 = move_data_to_device(batch_data_dict[f"{self.input2}_roll"], device) if self.input2 is not None else None
        batch_input3 = move_data_to_device(batch_data_dict[f"{self.input3}_roll"], device) if self.input3 is not None else None

        with torch.no_grad():
            self.model.eval()
            if batch_input2 is not None:
                if batch_input3 is not None:
                    out = self.model(batch_input1, batch_input2, batch_input3)
                else:
                    out = self.model(batch_input1, batch_input2)
            else:
                out = self.model(batch_input1)
        if "velocity_output" not in out and "vel_corr" in out:
            out = dict(out)
            out["velocity_output"] = out["vel_corr"]
        return out

    def evaluate(self, dataloader):
        """Evaluate over dataloader and compute Kim metrics."""
        output_dict = {}
        device = next(self.model.parameters()).device
        required_target_keys = ("velocity_roll", "frame_roll", "onset_roll", "pedal_frame_roll")

        for batch_data_dict in dataloader:
            out = self._forward_score_inf(batch_data_dict, device) if self.score_inf else self._forward_legacy(batch_data_dict, device)
            pred = out.get("velocity_output")
            if torch.is_tensor(pred):
                append_to_dict(output_dict, "velocity_output", pred.data.cpu().numpy())

            for key in required_target_keys:
                append_to_dict(output_dict, key, batch_data_dict[key])

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        if 'velocity_output' in output_dict:
            segments, targets = _segments_from_output(output_dict)
            return _kim_metrics_from_segments(segments, targets)
        return {}
