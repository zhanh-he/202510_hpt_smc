import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from feature_extractor import get_feature_extractor_and_bins
from pytorch_utils import move_data_to_device
from einops import rearrange
from score_inf.utils import safe_logit

class Rearrange(nn.Module):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern
    def forward(self, x):
        return rearrange(x, self.pattern)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
            # nn.init.zeros_(layer.bias)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)
    # nn.init.zeros_(layer.bias)
    # nn.init.ones_(bn.weight)

def init_bilstm(lstm):
    """Initialize weights for a Bidirectional LSTM layer."""
    for name, param in lstm.named_parameters():
        if 'weight' in name:
            nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            # nn.init.constant_(param, 0.0)

def init_gru(rnn):
    """Initialize GRU weights and biases for better convergence."""
    def _concat_init(tensor, inits):
        fan_in = tensor.shape[0] // len(inits)
        for i, fn in enumerate(inits):
            fn(tensor[i * fan_in:(i + 1) * fan_in])
    def _inner_uniform(tensor):
        fan_in = nn.init._calculate_correct_fan(tensor, 'fan_in')
        bound = math.sqrt(3 / fan_in)
        nn.init.uniform_(tensor, -bound, bound)

    for i in range(rnn.num_layers):
        _concat_init(getattr(rnn, f'weight_ih_l{i}'),
                     [_inner_uniform, _inner_uniform, _inner_uniform])
        nn.init.constant_(getattr(rnn, f'bias_ih_l{i}'), 0.)

        _concat_init(getattr(rnn, f'weight_hh_l{i}'),
                     [_inner_uniform, _inner_uniform, nn.init.orthogonal_])
        nn.init.constant_(getattr(rnn, f'bias_hh_l{i}'), 0.)

class HPTConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, momentum):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channels,out_channels=out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum)
        self.init_weight()
    def init_weight(self):
        init_layer(self.conv1)
        init_bn(self.bn1)
        init_layer(self.conv2)
        init_bn(self.bn2)
    def forward(self, input):
        """input: (batch_size, in_channels,  time_steps, freq_bins)
          output: (batch_size, out_channels, classes_num)"""
        x = F.relu_(self.bn1(self.conv1(input)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.avg_pool2d(x, kernel_size=(1,2))
        return x


def _align_time_dim_tensors(*tensors):
    """Trim tensors along time dimension (dim=1) to the minimum shared length."""
    valid = [t for t in tensors if t is not None]
    if not valid:
        return tensors
    min_steps = min(t.size(1) for t in valid)
    aligned = []
    for t in tensors:
        if t is None:
            aligned.append(None)
        elif t.size(1) == min_steps:
            aligned.append(t)
        else:
            aligned.append(t[:, :min_steps])
    return tuple(aligned)

class OriginalModelHPT2020(nn.Module):
    def __init__(self, classes_num, input_shape, momentum):
        super().__init__()
        self.conv_block1 = HPTConvBlock(in_channels=1, out_channels=48, momentum=momentum)
        self.conv_block2 = HPTConvBlock(in_channels=48, out_channels=64, momentum=momentum)
        self.conv_block3 = HPTConvBlock(in_channels=64, out_channels=96, momentum=momentum)
        self.conv_block4 = HPTConvBlock(in_channels=96, out_channels=128, momentum=momentum)
        # Auto Calculate midfeat
        with torch.no_grad():
            dummy = torch.zeros((1, 1, 1000, input_shape))  # 1000个帧，freq维为 input_shape
            x = self.conv_block1(dummy)
            x = self.conv_block2(x)
            x = self.conv_block3(x)
            x = self.conv_block4(x)
            # Flatten later uses x.transpose(1,2).flatten(2) -> channels × freq
            midfeat = x.shape[1] * x.shape[3]
        self.fc5 = nn.Linear(in_features=midfeat, out_features=768, bias=False)
        self.bn5 = nn.BatchNorm1d(768, momentum=momentum)
        self.gru = nn.GRU(input_size=768, hidden_size=256, num_layers=2,
                          bias=True, batch_first=True, dropout=0., bidirectional=True)
        self.fc =  nn.Linear(in_features=512, out_features=classes_num, bias=True)
        self.init_weight()
    def init_weight(self):
        init_layer(self.fc5)
        init_bn(self.bn5)
        init_gru(self.gru)
        init_layer(self.fc)
    def forward(self, input):
        """Args: input: (batch_size, channels_num, time_steps, freq_bins)
        Outputs: output: (batch_size, time_steps, classes_num)
        Selections: #print("Shape after conv_block2:", x.shape)"""								  
        x = self.conv_block1(input)		# conB1 batch=8, chn=48,timstep=1001, mel=114 
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x)	  		# conB2 batch=8, chn=64,timstep=1001, mel=57 
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x)	  		# conB3 batch=8, chn=96,timstep=1001, mel=28 
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x)	  		# conB4 batch=8,chn=128,timstep=1001, mel=14 
        x = F.dropout(x, p=0.2, training=self.training)
        x = x.transpose(1, 2).flatten(2)	# tranpose 8,1001,48,114 ; flat 8,1001,1792 
        x = F.relu(self.bn5(self.fc5(x).transpose(1,2)).transpose(1,2)) # batch=8, timstep=1001, class=768
        x = F.dropout(x, p=0.5, training=self.training)
        (x, _) = self.gru(x)			# gru batch=8, timstep=1001, class=512 
        x = F.dropout(x, p=0.5, training=self.training)
        output = torch.sigmoid(self.fc(x))	# out batch=8, timstep=1001, class=88 
        return output


class Single_Velocity_HPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        sample_rate         = cfg.feature.sample_rate
        fft_size            = cfg.feature.fft_size
        frames_per_second   = cfg.feature.frames_per_second
        audio_feature       = cfg.feature.audio_feature
        classes_num         = cfg.feature.classes_num
        momentum            = 0.01
        self.feature_extractor, self.FRE = get_feature_extractor_and_bins(audio_feature, sample_rate, fft_size, frames_per_second)
        # midfeat = 1792
        self.bn0 = nn.BatchNorm2d(self.FRE, momentum)
        self.velocity_model = OriginalModelHPT2020(classes_num, self.FRE , momentum)  # OriginalModelHPT2020
        # self.velocity_model = OriginalModelHPT2020(classes_num, midfeat, momentum)  # OriginalModelHPT2020
        self.init_weight()
    def init_weight(self):
        init_bn(self.bn0)
    def forward(self, input):
        """
        Args: input: (batch_size, data_length)
        Outputs: output_dict: dict, {'velocity_output': (batch_size, time_steps, classes_num)}
        """
        x = self.feature_extractor(input)    	# batch=12, melbins=229, timsteps=1001 (new,torchaudio)
        x = x.unsqueeze(3)                      # batch=12, melbins=229, timsteps=1001, ch=1
        x = self.bn0(x)					        # batch=12, melbins=229, timsteps=1001, ch=1
        x = x.transpose(1, 3)			    	# batch=12, ch=1, timsteps=1001, melbins=229
        est_velocity = self.velocity_model(x)  	# batch=12, timsteps=1001, classes_num=88
        vel_logits = safe_logit(est_velocity)
        return {'velocity_output': est_velocity, 'velocity_logits': vel_logits}