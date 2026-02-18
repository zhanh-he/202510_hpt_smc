import torch
import torch.nn.functional as F


def _align_time_dim(*tensors):
    """Trim all tensors along time dimension to the minimum shared length."""
    if not tensors:
        return tensors
    min_steps = min(tensor.size(1) for tensor in tensors)
    if all(tensor.size(1) == min_steps for tensor in tensors):
        return tensors
    return tuple(tensor[:, :min_steps] for tensor in tensors)


def bce(output, target, mask):
    """Binary crossentropy (BCE) with mask. The positions where mask=0 will be 
    deactivated when calculation BCE."""
    output, target, mask = _align_time_dim(output, target, mask)
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return torch.sum(matrix * mask) / torch.sum(mask)


def mse(output, target, mask):
    """Mean squared error (MSE) with mask"""
    output, target, mask = _align_time_dim(output, target, mask)
    return torch.sum(((output - target) ** 2) * mask) / torch.sum(mask)


def _masked_l1(output, target, mask):
    """Mean absolute error restricted by mask."""
    output, target, mask = _align_time_dim(output, target, mask)
    mask = mask.to(output.dtype)
    diff = torch.abs(output - target) * mask
    denom = torch.sum(mask).clamp_min(1e-8)
    return torch.sum(diff) / denom


def _masked_huber(output, target, mask, delta=0.1):
    """Huber loss restricted by mask."""
    output, target, mask = _align_time_dim(output, target, mask)
    mask = mask.to(output.dtype)
    huber = F.huber_loss(output, target, reduction="none", delta=delta)
    denom = torch.sum(mask).clamp_min(1e-8)
    return torch.sum(huber * mask) / denom


def _masked_bce(output, target, mask):
    """Binary cross-entropy restricted by mask (probability-domain BCE)."""
    output, target, mask = _align_time_dim(output, target, mask)
    mask = mask.to(output.dtype)
    output = torch.clamp(output, 1e-7, 1.0 - 1e-7)
    bce_mat = F.binary_cross_entropy(output, target, reduction="none")
    denom = torch.sum(mask).clamp_min(1e-8)
    return torch.sum(bce_mat * mask) / denom

# def mae(output, target, mask):
#     """Mean absolute error (MAE) with mask"""
#     abs_diff = torch.abs(output - target)
#     abs_diff = abs_diff * mask
#     mean_abs_diff = torch.sum(abs_diff) / torch.sum(mask)
#     return mean_abs_diff

# def std(output, target, mask):
#     """Standard Deviation of Absolute Error (std_ae) with mask"""
#     abs_diff = torch.abs(output - target)
#     abs_diff = abs_diff * mask
#     mean_abs_diff = torch.sum(abs_diff) / torch.sum(mask)

#     squared_diff = (abs_diff - mean_abs_diff) ** 2
#     squared_diff = squared_diff * mask
#     variance = torch.sum(squared_diff) / torch.sum(mask)
#     return torch.sqrt(variance)

# def std_ae(output, target, mask):
#     """Standard Deviation of Absolute Error (std_ae) with mask"""
#     abs_diff = torch.abs(output - target)
#     masked_diff = abs_diff * mask
#     # Calculate mean absolute difference
#     mean_abs_diff = torch.mean(masked_diff)
#     # Calculate squared differences from mean
#     squared_diff = (masked_diff - mean_abs_diff) ** 2
#     # Mask out non-relevant entries
#     squared_diff_masked = squared_diff * mask
#     # Calculate variance (std_ae is square root of variance)
#     variance = torch.sum(squared_diff_masked) / torch.sum(mask)
#     return torch.sqrt(variance)


############ Velocity loss ############
def _get_velocity_pred(output_dict):
    """Fetch velocity prediction tensor, accepting both vel_corr and velocity_output keys."""
    if "vel_corr" in output_dict:
        return output_dict["vel_corr"]
    if "velocity_output" in output_dict:
        return output_dict["velocity_output"]
    raise KeyError("velocity prediction not found in output_dict (expected vel_corr or velocity_output)")


def velocity_bce(model, output_dict, target_dict, cond_dict=None):
    """velocity regression losses only, used bce in HPT"""
    pred = _get_velocity_pred(output_dict)
    velocity_loss = bce(pred, target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    return velocity_loss


def velocity_mse(model, output_dict, target_dict, cond_dict=None):
    """velocity regression losses only, used mse in ONF"""
    pred = _get_velocity_pred(output_dict)
    velocity_loss = mse(pred, target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    return velocity_loss


def kim_velocity_bce_l1(model, output_dict, target_dict, cond_dict=None):
    """
    BCE + L1 hybrid loss proposed by Kim et al. (ISMIR 2024) for velocity regression.
    """
    theta = getattr(model, "kim_loss_alpha", 0.5)
    pred = _get_velocity_pred(output_dict)
    bce_loss = bce(pred, target_dict['velocity_roll'] / 128, target_dict['frame_roll'])
    onset_target = target_dict['velocity_roll'] / 128
    l1_loss = _masked_l1(pred, onset_target, target_dict['onset_roll'])
    return theta * bce_loss + (1 - theta) * l1_loss


# def velocity_mae(model, output_dict, target_dict):
#     """Test the performance"""
#     velocity_loss = mae(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
#     return velocity_loss

# def velocity_std(model, output_dict, target_dict):
#     """Test the performance"""
#     velocity_loss = std_ae(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
#     return velocity_loss

def count_inversions(model, output_dict, target_dict, cond_dict=None):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mask = target_dict['frame_roll']
    tensor = _get_velocity_pred(output_dict)
    mask, tensor = _align_time_dim(mask, tensor)
    
    batch_size, rows, cols = tensor.shape
    inversions = torch.zeros((batch_size, rows), dtype=torch.float).to(device)
    
    total_num = torch.zeros((batch_size, rows), dtype=torch.float).to(device)
    
    for i in range(batch_size):
        for j in range(rows):
            
            total_num[i, j] += 1e-5 # 保证除法不为 0
            
            seq_start = -1  # 序列开始的索引
            for k in range(cols):
                # 检测连续序列的开始
                if mask[i, j, k] == 1 and seq_start == -1:
                    seq_start = k
                # 检测连续序列的结束
                if (mask[i, j, k] == 0 or k == cols - 1) and seq_start != -1:
                    seq_end = k if mask[i, j, k] == 0 else k + 1
                    # 对当前连续序列计算连续的顺序数
                    for m in range(seq_start, seq_end):
                        for n in range(m + 1, seq_end):
                            if tensor[i, j, m] < tensor[i, j, n]:
                                inversions[i, j] += 1
                    seq_len = seq_end - seq_start
                    seq_inversion_num = seq_len * (seq_len - 1) / 2
                    total_num[i, j] += seq_inversion_num
                    
                    seq_start = -1  # 重置序列开始的索引，为检测下一个序列做准备
   
    
    inversion_loss = (inversions / total_num)

    inversion_loss_mask = inversion_loss.unsqueeze(-1).repeat(1, 1, cols).to(device)
    
    inversion_loss = (mask * inversion_loss_mask * tensor).mean()
    
    return inversion_loss


def combined_loss(model, output_dict, target_dict, cond_dict=None): 
    velocity_bce_loss = velocity_bce(model, output_dict, target_dict, cond_dict)
    count_inversions_loss = count_inversions(model, output_dict, target_dict, cond_dict)
    
    return velocity_bce_loss + count_inversions_loss


def score_inf_custom_loss(output_dict, target_dict, cond_dict, cfg):
    """Custom score-informed loss (masked L1/Huber + masked BCE + optional delta penalty)."""
    vel_corr = output_dict["vel_corr"]
    velocity_scale = float(getattr(cfg.feature, "velocity_scale", 128))
    gt_vel = target_dict["velocity_roll"] / velocity_scale

    onset_mask = None
    if isinstance(cond_dict, dict):
        onset_mask = cond_dict.get("onset")
    if onset_mask is None:
        onset_mask = target_dict["onset_roll"]

    # Loss weights now pulled directly from cfg.loss (see config.yaml)
    loss_cfg = getattr(cfg, "loss", None)
    if loss_cfg is None:
        raise ValueError("cfg.loss is required. Add a 'loss' section to config.yaml.")

    w_l1 = float(loss_cfg.w_l1)
    w_bce = float(loss_cfg.w_bce)
    w_delta = float(loss_cfg.w_delta)
    use_huber = bool(loss_cfg.use_huber)
    w_huber = float(loss_cfg.w_huber)
    huber_delta = float(loss_cfg.huber_delta)

    if use_huber:
        loss_reg = w_huber * _masked_huber(vel_corr, gt_vel, onset_mask, delta=huber_delta)
    else:
        loss_reg = w_l1 * _masked_l1(vel_corr, gt_vel, onset_mask)

    loss_bce = _masked_bce(vel_corr, gt_vel, onset_mask)
    loss = loss_reg + w_bce * loss_bce

    if w_delta > 0.0 and output_dict.get("delta", None) is not None:
        delta = output_dict["delta"]
        if torch.is_tensor(delta) and delta.dim() == vel_corr.dim():
            loss = loss + w_delta * delta.abs().mean()

    return loss


def compute_loss(cfg, model, output_dict, target_dict, cond_dict=None):
    """
    Unified training loss entry.
    Delegates loss selection to get_loss_func for both legacy and score-informed trainers.
    """
    return get_loss_func(cfg=cfg)(model, output_dict, target_dict, cond_dict)
                                             

def get_loss_func(loss_type=None, cfg=None):
    """
    Return a callable with unified signature:
      fn(model, output_dict, target_dict, cond_dict=None) -> loss

    Selection order:
    - explicit loss_type if provided
    - cfg.loss.loss_type (preferred)
    - cfg.exp.loss_type (legacy fallback)
    """
    if loss_type == "score_inf_custom":
        if cfg is None:
            raise ValueError("cfg is required for score_inf_custom loss.")

        def _score_inf_loss(_, output_dict, target_dict, cond_dict=None):
            return score_inf_custom_loss(output_dict, target_dict, cond_dict, cfg)

        return _score_inf_loss

    legacy_map = {
        "velocity_bce": velocity_bce,
        "velocity_mse": velocity_mse,
        "kim_bce_l1": kim_velocity_bce_l1,
        "combine_bce": combined_loss,
    }
    if loss_type in legacy_map:
        return legacy_map[loss_type]

    raise Exception('Incorrect loss_type!')
