import numpy as np
import torch

def move_data_to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if np.issubdtype(x.dtype, np.floating):
        return torch.tensor(x, dtype=torch.float32, device=device)
    if np.issubdtype(x.dtype, np.integer):
        return torch.tensor(x, dtype=torch.long, device=device)
    return x


def append_to_dict(dict, key, value):
    if key in dict.keys():
        dict[key].append(value)
    else:
        dict[key] = [value]

def forward(model, x, batch_size):
    """Forward data to model in mini-batch. 
    Args: 
      model: object
      x: (N, segment_samples)
      batch_size: int
    Returns:
      output_dict: dict, e.g. {
        'frame_output': (segments_num, frames_num, classes_num),
        'onset_output': (segments_num, frames_num, classes_num),
        ...}
    """
    
    output_dict = {}
    device = next(model.parameters()).device
    
    pointer = 0
    while True:
        if pointer >= len(x):
            break
        batch_waveform = move_data_to_device(x[pointer : pointer + batch_size], device)
        pointer += batch_size
        with torch.no_grad():
            model.eval()
            batch_output_dict = model(batch_waveform)
            
        for key in batch_output_dict.keys():
            append_to_dict(output_dict, key, batch_output_dict[key].data.cpu().numpy())

    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict


def forward_velo(model, x1, x2, x3, batch_size):
    """Forward data to model in mini-batch.
    Args:
      model: object
      x1: audio_segments (N, segment_samples)
      x2: onset_roll(N, frames_num, classes_num)
      x3: frame_roll or exframe_roll (onset-excluded frame) (N, frames_num, classes_num)
      batch_size: int
    Returns:
      output_dict: dict, {'velocity_output': (segments_num, frames_num, classes_num)}
    """
    output_dict = {}
    device = next(model.parameters()).device
    pointer = 0

    while pointer < len(x1):
        # Process batches
        batch_audio = move_data_to_device(x1[pointer: pointer + batch_size], device)
        batch_input2 = move_data_to_device(x2[pointer: pointer + batch_size], device) if x2 is not None else None
        batch_input3 = move_data_to_device(x3[pointer: pointer + batch_size], device) if x3 is not None else None
        pointer += batch_size

        with torch.no_grad():
            model.eval()
            inputs = [batch_audio]
            if x2 is not None:
                inputs.append(batch_input2)
            if x3 is not None:
                inputs.append(batch_input3)
            batch_output_dict = model(*inputs)

        append_to_dict(output_dict, 'velocity_output', batch_output_dict['velocity_output'].data.cpu().numpy())

    output_dict['velocity_output'] = np.concatenate(output_dict['velocity_output'], axis=0)
    return output_dict
