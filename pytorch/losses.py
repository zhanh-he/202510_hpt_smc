import torch
from score_inf.utils import masked_bce_with_logits, masked_huber, masked_l1, safe_logit


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
    diff = torch.abs(output - target) * mask
    denom = torch.sum(mask)
    if denom.item() == 0:
        return torch.zeros(1, device=output.device, dtype=output.dtype)
    return torch.sum(diff) / denom

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
    

def _bce_unmasked(output, target):
    """Binary crossentropy without masking."""
    output, target = _align_time_dim(output, target)
    eps = 1e-7
    output = torch.clamp(output, eps, 1. - eps)
    matrix = - target * torch.log(output) - (1. - target) * torch.log(1. - output)
    return matrix.mean()


############ Velocity loss ############
def velocity_bce(model, output_dict, target_dict):                                
    """velocity regression losses only, used bce in HPT"""                                                 ## frame_roll
    velocity_loss = bce(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    return velocity_loss

def velocity_mse(model, output_dict, target_dict):
    """velocity regression losses only, used mse in ONF"""
    velocity_loss = mse(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
    return velocity_loss


def kim_velocity_bce_l1(model, output_dict, target_dict):
    """
    BCE + L1 hybrid loss proposed by Kim et al. (ISMIR 2024) for velocity regression.
    """
    theta = getattr(model, "kim_loss_alpha", 0.5)
    bce_loss = bce(
        output_dict['velocity_output'],
        target_dict['velocity_roll'] / 128,
        target_dict['frame_roll'],
    )
    onset_target = target_dict['velocity_roll'] / 128
    l1_loss = _masked_l1(output_dict['velocity_output'], onset_target, target_dict['onset_roll'])
    return theta * bce_loss + (1 - theta) * l1_loss


def velocity_bce_tri(model, output_dict, target_dict,
                     onset_weight=1.0, frame_weight=1.0, global_weight=0.1):
    """Combine BCE over onset mask, frame mask, and an unmasked global term."""
    pred = output_dict['velocity_output']
    target = target_dict['velocity_roll'] / 128
    onset_loss = bce(pred, target, target_dict['onset_roll'])
    frame_loss = bce(pred, target, target_dict['frame_roll'])
    global_loss = _bce_unmasked(pred, target)
    return onset_weight * onset_loss + frame_weight * frame_loss + global_weight * global_loss

# def velocity_mae(model, output_dict, target_dict):
#     """Test the performance"""
#     velocity_loss = mae(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
#     return velocity_loss

# def velocity_std(model, output_dict, target_dict):
#     """Test the performance"""
#     velocity_loss = std_ae(output_dict['velocity_output'], target_dict['velocity_roll'] / 128, target_dict['onset_roll'])
#     return velocity_loss

def count_inversions(model, output_dict, target_dict):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    mask = target_dict['frame_roll']
    tensor = output_dict['velocity_output']
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


def combined_loss(model, output_dict, target_dict): 
    velocity_bce_loss = velocity_bce(model, output_dict, target_dict)
    count_inversions_loss = count_inversions(model, output_dict, target_dict)
    
    return velocity_bce_loss + count_inversions_loss


def _cfg_get(cfg_obj, key, default):
    if cfg_obj is None:
        return default
    if hasattr(cfg_obj, "get"):
        return cfg_obj.get(key, default)
    return getattr(cfg_obj, key, default)


def score_inf_custom_loss(output_dict, target_dict, cond_dict, cfg):
    """Custom score-informed loss (masked L1/Huber + masked BCE logits + optional delta penalty)."""
    vel_corr = output_dict["vel_corr"]
    velocity_scale = float(getattr(cfg.feature, "velocity_scale", 128))
    gt_vel = target_dict["velocity_roll"] / velocity_scale

    onset_mask = None
    if isinstance(cond_dict, dict):
        onset_mask = cond_dict.get("onset")
    if onset_mask is None:
        onset_mask = target_dict["onset_roll"]

    loss_cfg = getattr(cfg, "loss", None)
    w_l1 = float(_cfg_get(loss_cfg, "w_l1", 1.0))
    w_bce = float(_cfg_get(loss_cfg, "w_bce", 0.5))
    w_delta = float(_cfg_get(loss_cfg, "w_delta", 0.0))
    use_huber = bool(_cfg_get(loss_cfg, "use_huber", False))
    huber_delta = float(_cfg_get(loss_cfg, "huber_delta", 0.1))

    if use_huber:
        loss_reg = masked_huber(vel_corr, gt_vel, onset_mask, delta=huber_delta)
    else:
        loss_reg = masked_l1(vel_corr, gt_vel, onset_mask)

    vel_corr_logits = safe_logit(vel_corr)
    loss_bce = masked_bce_with_logits(vel_corr_logits, gt_vel, onset_mask)
    loss = w_l1 * loss_reg + w_bce * loss_bce

    if w_delta > 0.0 and output_dict.get("delta", None) is not None:
        delta = output_dict["delta"]
        if torch.is_tensor(delta) and delta.dim() == vel_corr.dim():
            loss = loss + w_delta * delta.abs().mean()

    return loss


def compute_loss(cfg, model, output_dict, target_dict, cond_dict=None):
    """
    Unified training loss entry for both legacy and score-informed trainers.
    - If cfg.loss exists: use score-inf custom loss.
    - Else: use cfg.exp.loss_type via get_loss_func.
    """
    if hasattr(cfg, "loss"):
        return score_inf_custom_loss(output_dict, target_dict, cond_dict, cfg)

    if "velocity_output" not in output_dict and "vel_corr" in output_dict:
        output_dict = {"velocity_output": output_dict["vel_corr"]}

    return get_loss_func(cfg.exp.loss_type)(model, output_dict, target_dict)
                                             

def get_loss_func(loss_type):
    if loss_type == 'velocity_bce':
        return velocity_bce
    elif loss_type == 'velocity_mse':
        return velocity_mse
    elif loss_type == 'kim_bce_l1':
        return kim_velocity_bce_l1
    elif loss_type == 'velocity_bce_tri':
        return velocity_bce_tri
    # elif loss_type == 'velocity_mae':
    #     return velocity_mae
    # elif loss_type == 'velocity_std':
    #     return velocity_std
    elif loss_type == 'combine_bce':
        return combined_loss
    else:
        raise Exception('Incorrect loss_type!')
