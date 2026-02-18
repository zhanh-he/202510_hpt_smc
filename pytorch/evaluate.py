import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import torch
from pytorch_utils import move_data_to_device, append_to_dict
from calculate_scores import gt_to_note_list, eval_from_list

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


def _kim_metrics_from_segments(segments, targets):
    """Run the same Kim-style metrics used in calculate_scores."""
    if not segments or not targets:
        return {}

    (
        frame_max_err,
        frame_max_std,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = gt_to_note_list(segments, targets)
    onset_masked_error, onset_masked_std = eval_from_list(segments, targets)

    stats = {
        'frame_max_error': round(frame_max_err, 4),
        'frame_max_std': round(frame_max_std, 4),
        'onset_masked_error': round(onset_masked_error, 4),
        'onset_masked_std': round(onset_masked_std, 4),
    }
    # Backward-compatible aliases for legacy logging keys.
    stats['velocity_mae'] = stats['frame_max_error']
    stats['velocity_std'] = stats['frame_max_std']
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
        self.batch_size = cfg.exp.batch_size
        self.input2 = cfg.model.input2
        self.input3 = cfg.model.input3
        self.score_inf = score_inf

    def _forward_score_inf(self, batch_data_dict, device):
        audio = move_data_to_device(batch_data_dict["waveform"], device)
        cond = {
            "onset": move_data_to_device(batch_data_dict["onset_roll"], device),
            "frame": move_data_to_device(batch_data_dict.get("frame_roll"), device) if batch_data_dict.get("frame_roll") is not None else None,
            "exframe": move_data_to_device(batch_data_dict.get("exframe_roll"), device) if batch_data_dict.get("exframe_roll") is not None else None,
        }

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
        statistics = {}
        output_dict = {}
        device = next(self.model.parameters()).device

        for batch_data_dict in dataloader:
            out = self._forward_score_inf(batch_data_dict, device) if self.score_inf else self._forward_legacy(batch_data_dict, device)

            for key, val in out.items():
                if "_list" not in key:
                    append_to_dict(output_dict, key, val.data.cpu().numpy())

            for target_type, tval in batch_data_dict.items():
                if 'roll' in target_type or 'reg_distance' in target_type or 'reg_tail' in target_type:
                    append_to_dict(output_dict, target_type, tval)

        for key in output_dict.keys():
            output_dict[key] = np.concatenate(output_dict[key], axis=0)

        if 'velocity_output' in output_dict:
            segments, targets = _segments_from_output(output_dict)
            statistics.update(_kim_metrics_from_segments(segments, targets))
        return statistics
