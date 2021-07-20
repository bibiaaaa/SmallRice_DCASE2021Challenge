import torchaudio
from torch.nn.modules.utils import _pair, _quadruple
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch import utils as efficientnet_utils
from torch_audiomentations import AddColoredNoise, Gain, PolarityInversion, Shift
from collections import OrderedDict
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from torch.nn import init
from torch import Tensor, einsum
from typing import Any, Callable, List, Optional, Tuple
import math
import utils

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as audio_transforms

from utils import MyMelSpectrogram


class AutoEncoder(nn.Module):
    def __init__(self,
                 num_layer=4,
                 input_dim=640,
                 hidden_dim=128,
                 bottleneck_dim=8,
                 **kwargs):
        super(AutoEncoder, self).__init__()
        input_d = input_dim
        modules = []
        for i in range(num_layer):
            modules.append(
                nn.Sequential(nn.Linear(input_d, hidden_dim),
                              nn.BatchNorm1d(hidden_dim),
                              nn.ReLU(inplace=True)))
            input_d = hidden_dim
        self.encoder = nn.Sequential(*modules)
        self.bottleneck = nn.Sequential(nn.Linear(input_d, bottleneck_dim),
                                        nn.BatchNorm1d(bottleneck_dim),
                                        nn.ReLU(inplace=True))
        input_d = bottleneck_dim
        modules = []
        for i in range(num_layer):
            modules.append(
                nn.Sequential(nn.Linear(input_d, hidden_dim),
                              nn.BatchNorm1d(hidden_dim),
                              nn.ReLU(inplace=True)))
            input_d = hidden_dim
        self.decoder = nn.Sequential(*modules)
        self.output = nn.Linear(input_d, input_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1).flatten(1, -1)
        feature = x.detach()
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        x = self.output(x)
        return x, feature


class MobileNetV2(nn.Module):
    def __init__(self,
                 outputdim=3,
                 width_mult=1.0,
                 wavtransforms=None,
                 spectransforms=None,
                 inverted_residual_setting=None,
                 norm_layer=None,
                 **kwargs):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use

        """
        super(MobileNetV2, self).__init__()

        _block = InvertedResidual_Stride
        _convbn = ConvBNReLU_Stride

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        n_mels = kwargs.get('n_mels', 128)
        hop_size = kwargs.get('hop_size', 512)
        win_size = kwargs.get('win_size', 1024)
        f_min = kwargs.get('f_min', 0)

        input_channel = 32
        last_channel = kwargs.get('last_channel', 1280)

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(
                inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(
                                 inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(
            last_channel * width_mult) if width_mult > 1.0 else last_channel
        features = [_convbn(1, input_channel, stride=2, norm_layer=norm_layer)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    _block(input_channel,
                           output_channel,
                           stride,
                           expand_ratio=t,
                           norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(
            _convbn(input_channel,
                    self.last_channel,
                    kernel_size=1,
                    norm_layer=norm_layer))
        features.append(nn.AdaptiveAvgPool2d((1, None)))
        # make it nn.Sequential
        self.front_end = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=f_min,
                                            sample_rate=16000,
                                            win_length=win_size,
                                            n_fft=win_size,
                                            hop_length=hop_size,
                                            n_mels=n_mels),
            audio_transforms.AmplitudeToDB(top_db=120),
        )
        self.wavtransforms = wavtransforms if wavtransforms != None else nn.Sequential(
        )
        self.spectransforms = spectransforms if spectransforms != None else nn.Sequential(
        )

        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.last_channel, outputdim),
        )

        # weight initialization
        self.features.apply(init_weights)
        self.classifier.apply(init_weights)
        self.medianfilter = MedianPool2d((1, 31), same=True)

    def forward(self, x: Tensor, medianfilter=False):
        if self.training:
            x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end(x)
        if medianfilter:
            x = self.medianfilter(x.unsqueeze(1)).squeeze(1)
        if self.training:
            x = self.spectransforms(x)
        x = x.unsqueeze(1)  #Add channel dimension
        x = self.features(x)
        x = x.flatten(-2)
        x = x.mean(-1) + x.max(-1)[0]
        embedding = x
        x = self.classifier(x)
        return embedding, x


MobileNetV2_Pretrain = MobileNetV2


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class ConvBNReLU_Stride(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 norm_layer=None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        super(ConvBNReLU_Stride, self).__init__(
            nn.Conv2d(in_planes,
                      out_planes,
                      kernel_size,
                      stride,
                      padding,
                      groups=groups,
                      bias=False), norm_layer(out_planes),
            nn.ReLU6(inplace=True))


class InvertedResidual_Stride(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, norm_layer=None):
        super(InvertedResidual_Stride, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvBNReLU_Stride(inp,
                                  hidden_dim,
                                  kernel_size=1,
                                  norm_layer=norm_layer))
        layers.extend([
            # dw
            ConvBNReLU_Stride(hidden_dim,
                              hidden_dim,
                              stride=stride,
                              groups=hidden_dim,
                              norm_layer=norm_layer),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0],
                     self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1, )).median(dim=-1)[0]
        return x


class ContrastiveModel(nn.Module):
    def __init__(self, **kwargs):
        super(ContrastiveModel, self).__init__()
        # get parameters from kwargs
        ae_num_layer = kwargs.get('ae_num_layer', 4)
        ae_input_dim = kwargs.get('ae_input_dim', 8192)
        ae_hidden_dim = kwargs.get('ae_hidden_dim', 128)
        ae_bottleneck_dim = kwargs.get('ae_bottleneck_dim', 8)
        hidden_dim = kwargs.get('hidden_dim', 256)
        cnn_outputdim = kwargs.get('cnn_outputdim', 3)
        # init training framework
        self.ae = AutoEncoder(input_dim=ae_input_dim)
        self.cnn = MobileNetV2(outputdim=cnn_outputdim)
        self.ae2hidden = nn.Linear(ae_bottleneck_dim, hidden_dim)
        self.cnn2hidden = nn.Linear(1280, hidden_dim)
        self.hidden = nn.Sequential(nn.Linear(hidden_dim,
                                              hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim))
        self.front_end = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=0,
                                            f_max=8000,
                                            sample_rate=16000,
                                            win_length=1024,
                                            n_fft=1024,
                                            hop_length=512,
                                            n_mels=128,
                                            center=False),
            audio_transforms.AmplitudeToDB(top_db=120),
        )
        self.medianfilter = MedianPool2d((1, 31), same=True)

    def forward(self, x, medianfilter=False):
        if medianfilter:
            x = self.medianfilter(x)
        if self.training:
            x = self.front_end(x)
            ae_input = x.reshape(x.shape[0], -1)
            ae_bottleneck = self.ae.bottleneck(self.ae.encoder(ae_input))
            cnn_bottleneck = self.cnn.features(x.unsqueeze(1))
            cnn_bottleneck = cnn_bottleneck.flatten(-2)
            cnn_bottleneck = cnn_bottleneck.mean(-1) + cnn_bottleneck.max(
                -1)[0]

            ae_bottleneck = F.normalize(ae_bottleneck, dim=1)
            cnn_bottleneck = F.normalize(cnn_bottleneck, dim=1)

            ae_hidden = self.ae2hidden(ae_bottleneck)
            cnn_hidden = self.cnn2hidden(cnn_bottleneck)
            ae_hidden_output = self.hidden(ae_hidden)
            cnn_hidden_output = self.hidden(cnn_hidden)
            ae_output = self.ae.output(self.ae.decoder(ae_bottleneck))
            cnn_output = self.cnn.classifier(cnn_bottleneck)
            return ae_input, ae_hidden_output, cnn_hidden_output, ae_output, cnn_output
        else:
            return self.cnn(x, medianfilter=medianfilter)[1]
