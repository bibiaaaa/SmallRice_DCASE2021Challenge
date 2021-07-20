from typing import Tuple

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
import torch
from torch import Tensor
import torch.nn as nn
from torch_audiomentations import Gain, PolarityInversion, Shift
import torchaudio
from torchaudio import transforms as audio_transforms

import utils
torchaudio.set_audio_backend("sox_io")


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class UnlabeledShiftAugmentor(nn.Module):
    """docstring for Augmentor"""
    def __init__(
        self,
        min_shift: float = -0.15,
        max_shift: float = 0.15,
    ):
        # 2 second mask as default
        super(UnlabeledShiftAugmentor, self).__init__()
        self.augmentations = torch.nn.Sequential(
            Rearrange('b t -> b 1 t'),
            Shift(min_shift=min_shift, max_shift=max_shift, sample_rate=16000),
            Rearrange('b 1 t -> b t'),
        )
    def forward(self, x):
        with torch.no_grad():
            return self.augmentations(x)

class TimeMasker(nn.Module):
    """docstring for Augmentor"""
    def __init__(
        self,
        mask_length: int = 16000,
    ):
        # 1 second mask as default
        super(TimeMasker, self).__init__()
        self.augmentations = torch.nn.Sequential(
            Rearrange('b t -> b 1 1 t'),
            torchaudio.transforms.TimeMasking(mask_length),
            Rearrange('b 1 1 t -> b t'),
        )
    def forward(self, x):
        with torch.no_grad():
            return self.augmentations(x)



class UnlabeledAugmentor(nn.Module):
    """docstring for Augmentor"""
    def __init__(
        self,
        mask_length: int = 16000,
        gain_min: float = -20.,
        gain_max: float = 10,
        prob: float = 0.5,
        min_shift :float = None,
        max_shift :float = None,
        one_dim_mode:bool = True
    ):
        # 1 second mask as default
        super(UnlabeledAugmentor, self).__init__()
        if min_shift != None and max_shift != None:
            shift = Shift(min_shift=min_shift, max_shift=max_shift, p=prob)
        else:
            shift = nn.Identity()
        # standard is one dim, i.e., Raw wave in shape B, T
        if one_dim_mode:
            self.augmentations = torch.nn.Sequential(
                Rearrange('b t -> b 1 t'),
                Gain(gain_min, gain_max, p=prob, sample_rate=16000),
                PolarityInversion(p=prob),
                shift,
                Rearrange('b 1 t -> b 1 1 t'),
                torchaudio.transforms.TimeMasking(mask_length),
                Rearrange('b 1 1 t -> b t'),
            )
        else:  # Input is B, C, T, used when input is from Separation model
            self.augmentations = torch.nn.Sequential(
                Gain(gain_min, gain_max, p=prob, sample_rate=16000),
                PolarityInversion(p=prob),
                shift,
                Rearrange('b c t -> b c 1 t'),
                torchaudio.transforms.TimeMasking(mask_length),
                Rearrange('b c 1 t -> b c t'),
            )



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = torch.flip(x, [0,1])
        with torch.no_grad():
            return self.augmentations(x)



class MelSpectrogramFrontEnd(nn.Module):
    def __init__(self,
                 n_mels: int,
                 f_min: int,
                 win_size: int,
                 hop_size: int,
                 n_fft: int = None,
                 sample_rate: int = 16000,
                 wavtransforms: nn.Sequential = None,
                 spectransforms: nn.Sequential = None,
                 *args,
                 **kwargs):
        super(MelSpectrogramFrontEnd, self).__init__()

        n_fft = win_size if n_fft is None else n_fft

        self.front_end_feature = nn.Sequential(
            audio_transforms.MelSpectrogram(f_min=f_min,
                                            n_fft=n_fft,
                                            sample_rate=sample_rate,
                                            win_length=win_size,
                                            hop_length=hop_size,
                                            n_mels=n_mels,
                                            *args,
                                            **kwargs,
                                            ),
            audio_transforms.AmplitudeToDB(top_db=120),
        )
        self.wavtransforms = wavtransforms if wavtransforms != None else nn.Sequential()
        self.spectransforms = spectransforms if spectransforms != None else nn.Sequential()

    def forward(self, x: Tensor, mixup: Tensor = None) -> Tensor:
        if self.training:
            x = self.wavtransforms(x.unsqueeze(1)).squeeze(1)
        x = self.front_end_feature(x)
        if self.training and mixup != None:
            x = utils.mixup(x, mixup)
        if self.training:
            x = self.spectransforms(x)
        return x



class CDur_2Sub_ClipSmooth(nn.Module):
    def __init__(self,
                 outputdim: int = 10,
                 spectransforms=None,
                 wavtransforms=None,
                 input_channels:int =1 ,
                 freeze_bn: bool = False,
                 **feature_params):
        super().__init__()
        # Feature params are passed to the Melspecfrontend
        default_feature_params = {
            'n_mels': 64,
            'hop_size': 160,
            'win_size': 512,
            'f_min': 0
        }
        self.freeze_bn = freeze_bn
        # Update only existing keys
        default_feature_params.update(
            (k, feature_params[k])
            for k in set(feature_params).intersection(default_feature_params))

        self.front_end = MelSpectrogramFrontEnd(spectransforms=spectransforms,
                                                wavtransforms=wavtransforms,
                                                **default_feature_params)

        def _block(cin, cout, kernel_size=3, padding=1, stride=1):
            return nn.Sequential(
                nn.BatchNorm2d(cin),
                nn.Conv2d(cin,
                          cout,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=False,
                          stride=stride),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
            )

        self.network = nn.Sequential(
            Rearrange('b f t -> b 1 t f'), _block(input_channels, 32),
            nn.LPPool2d(4, (2, 4)), _block(32, 128), _block(128, 128),
            nn.LPPool2d(4, (1, 4)), _block(128, 128), _block(128, 128),
            nn.LPPool2d(4, (1, 4)), nn.Dropout(0.3),
            Rearrange('b c t f -> b t (f c)'),
            nn.GRU(128, 128, bidirectional=True, batch_first=True))
        self.outputdim = outputdim
        self.outputlayer = nn.Linear(256, outputdim)

        self.apply(init_weights)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CDur_2Sub_ClipSmooth, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        return self

    def forward(
        self,
        x: Tensor,
        mixup: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        x = self.front_end(x, mixup)
        *_, input_time_length = x.shape
        x, _ = self.network(x)
        time_out = torch.sigmoid(self.outputlayer(x))

        clip_out = reduce(time_out**2,
                          'b t events -> b events', 'sum') / reduce(
                              time_out, 'b t events -> b events', 'sum')
        # Upsample output to match labels
        time_out = rearrange(
            torch.nn.functional.interpolate(rearrange(time_out,
                                                      'b t f -> b f t'),
                                            input_time_length,
                                            mode='linear',
                                            align_corners=False),
            'b f t -> b t f')
        return clip_out, time_out.contiguous() * rearrange(
            clip_out, 'b events -> b 1 events')

class CDur_8Sub_ClipSmooth(nn.Module):
    def __init__(self,
                 outputdim: int = 10,
                 spectransforms=None,
                 wavtransforms=None,
                 input_channels:int =1 ,
                 freeze_bn: bool = False,
                 **feature_params):
        super().__init__()
        # Feature params are passed to the Melspecfrontend
        default_feature_params = {
            'n_mels': 64,
            'hop_size': 160,
            'win_size': 512,
            'f_min': 0
        }
        self.freeze_bn = freeze_bn
        # Update only existing keys
        default_feature_params.update(
            (k, feature_params[k])
            for k in set(feature_params).intersection(default_feature_params))

        self.front_end = MelSpectrogramFrontEnd(spectransforms=spectransforms,
                                                wavtransforms=wavtransforms,
                                                **default_feature_params)

        def _block(cin, cout, kernel_size=3, padding=1, stride=1):
            return nn.Sequential(
                nn.BatchNorm2d(cin),
                nn.Conv2d(cin,
                          cout,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=False,
                          stride=stride),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
            )

        self.network = nn.Sequential(
            Rearrange('b f t -> b 1 t f'), _block(input_channels, 32),
            nn.LPPool2d(4, (2, 4)), _block(32, 128), _block(128, 128),
            nn.LPPool2d(4, (2, 4)), _block(128, 128), _block(128, 128),
            nn.LPPool2d(4, (2, 4)), nn.Dropout(0.3),
            Rearrange('b c t f -> b t (f c)'),
            nn.GRU(128, 128, bidirectional=True, batch_first=True))
        self.outputdim = outputdim
        self.outputlayer = nn.Linear(256, outputdim)

        self.apply(init_weights)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CDur_8Sub_ClipSmooth, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        return self

    def forward(
        self,
        x: Tensor,
        mixup: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        x = self.front_end(x, mixup)
        *_, input_time_length = x.shape
        x, _ = self.network(x)
        time_out = torch.sigmoid(self.outputlayer(x))

        clip_out = reduce(time_out**2,
                          'b t events -> b events', 'sum') / reduce(
                              time_out, 'b t events -> b events', 'sum')
        # Upsample output to match labels
        time_out = rearrange(
            torch.nn.functional.interpolate(rearrange(time_out,
                                                      'b t f -> b f t'),
                                            input_time_length,
                                            mode='linear',
                                            align_corners=False),
            'b f t -> b t f')
        return clip_out, time_out.contiguous() * rearrange(
            clip_out, 'b events -> b 1 events')

class CDur_NoSub_ClipSmooth(nn.Module):
    def __init__(self,
                 outputdim: int = 10,
                 spectransforms=None,
                 wavtransforms=None,
                 input_channels:int =1 ,
                 freeze_bn: bool = False,
                 **feature_params):
        super().__init__()
        # Feature params are passed to the Melspecfrontend
        default_feature_params = {
            'n_mels': 64,
            'hop_size': 160,
            'win_size': 512,
            'f_min': 0
        }
        self.freeze_bn = freeze_bn
        # Update only existing keys
        default_feature_params.update(
            (k, feature_params[k])
            for k in set(feature_params).intersection(default_feature_params))

        self.front_end = MelSpectrogramFrontEnd(spectransforms=spectransforms,
                                                wavtransforms=wavtransforms,
                                                **default_feature_params)

        def _block(cin, cout, kernel_size=3, padding=1, stride=1):
            return nn.Sequential(
                nn.BatchNorm2d(cin),
                nn.Conv2d(cin,
                          cout,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=False,
                          stride=stride),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
            )

        self.network = nn.Sequential(
            Rearrange('b f t -> b 1 t f'), _block(input_channels, 32),
            nn.LPPool2d(4, (1, 4)), _block(32, 128), _block(128, 128),
            nn.LPPool2d(4, (1, 4)), _block(128, 128), _block(128, 128),
            nn.LPPool2d(4, (1, 4)), nn.Dropout(0.3),
            Rearrange('b c t f -> b t (f c)'),
            nn.GRU(128, 128, bidirectional=True, batch_first=True))
        self.outputdim = outputdim
        self.outputlayer = nn.Linear(256, outputdim)

        self.apply(init_weights)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CDur_NoSub_ClipSmooth, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        return self

    def forward(
        self,
        x: Tensor,
        mixup: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        x = self.front_end(x, mixup)
        *_, input_time_length = x.shape
        x, _ = self.network(x)
        time_out = torch.sigmoid(self.outputlayer(x))

        clip_out = reduce(time_out**2,
                          'b t events -> b events', 'sum') / reduce(
                              time_out, 'b t events -> b events', 'sum')
        # Upsample output to match labels
        time_out = rearrange(
            torch.nn.functional.interpolate(rearrange(time_out,
                                                      'b t f -> b f t'),
                                            input_time_length,
                                            mode='linear',
                                            align_corners=False),
            'b f t -> b t f')
        return clip_out, time_out.contiguous() * rearrange(
            clip_out, 'b events -> b 1 events')


class CDur(nn.Module):
    def __init__(self,
                 outputdim: int = 10,
                 spectransforms=None,
                 wavtransforms=None,
                 input_channels:int =1 ,
                 freeze_bn: bool = False,
                 **feature_params):
        super().__init__()
        # Feature params are passed to the Melspecfrontend
        default_feature_params = {
            'n_mels': 64,
            'hop_size': 160,
            'win_size': 512,
            'f_min': 0
        }
        self.freeze_bn = freeze_bn
        # Update only existing keys
        default_feature_params.update(
            (k, feature_params[k])
            for k in set(feature_params).intersection(default_feature_params))

        self.front_end = MelSpectrogramFrontEnd(spectransforms=spectransforms,
                                                wavtransforms=wavtransforms,
                                                **default_feature_params)

        def _block(cin, cout, kernel_size=3, padding=1, stride=1):
            return nn.Sequential(
                nn.BatchNorm2d(cin),
                nn.Conv2d(cin,
                          cout,
                          kernel_size=kernel_size,
                          padding=padding,
                          bias=False,
                          stride=stride),
                nn.LeakyReLU(inplace=True, negative_slope=0.1),
            )

        self.network = nn.Sequential(
            Rearrange('b f t -> b 1 t f'), _block(input_channels, 32),
            nn.LPPool2d(4, (2, 4)), _block(32, 128), _block(128, 128),
            nn.LPPool2d(4, (2, 4)), _block(128, 128), _block(128, 128),
            nn.LPPool2d(4, (1, 4)), nn.Dropout(0.3),
            Rearrange('b c t f -> b t (f c)'),
            nn.GRU(128, 128, bidirectional=True, batch_first=True))
        self.outputdim = outputdim
        self.outputlayer = nn.Linear(256, outputdim)

        self.apply(init_weights)

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        """
        super(CDur, self).train(mode)
        if self.freeze_bn:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    if self.freeze_bn:
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False
        return self

    def forward(
        self,
        x: Tensor,
        mixup: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        x = self.front_end(x, mixup)
        *_, input_time_length = x.shape
        x, _ = self.network(x)
        time_out = torch.sigmoid(self.outputlayer(x))

        clip_out = reduce(time_out**2,
                          'b t events -> b events', 'sum') / reduce(
                              time_out, 'b t events -> b events', 'sum')
        # Upsample output to match labels
        time_out = rearrange(
            torch.nn.functional.interpolate(rearrange(time_out,
                                                      'b t f -> b f t'),
                                            input_time_length,
                                            mode='linear',
                                            align_corners=False),
            'b f t -> b t f')
        return clip_out, time_out.contiguous()


class CDur_ClipSmooth(CDur):

    def __init__(self, *args, **kwargs):
        super(CDur_ClipSmooth, self).__init__(*args, **kwargs)

    def forward(
        self,
        x: Tensor,
        mixup: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        c, t = super().forward(x, mixup)
        # Multiply time output with its clip weights
        return c, t * rearrange(c, 'b events -> b 1 events')


if __name__ == "__main__":
    from pytorch_model_summary import summary
    model = CDur()
    x = torch.randn(1, 16000)
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    print(summary(model, x, show_input=True))
