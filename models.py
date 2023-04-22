""" Libraries """
import torch
import torch.nn as nn

# import json
# import urllib
# from pytorchvideo.data.encoded_video import EncodedVideo

# from torchvision.transforms import Compose, Lambda
# from torchvision.transforms._transforms_video import (
#     CenterCropVideo,
#     NormalizeVideo,
# )
# from pytorchvideo.transforms import (
#     ApplyTransformToKey,
#     ShortSideScale,
#     UniformTemporalSubsample
# )


""" Models """
class MyLinear1D(nn.Module):
    def __init__(self, in_channels, out_channels, act_fn=nn.SiLU()) -> None:
        super().__init__()
        self.linear        = nn.Linear(in_channels, out_channels)
        self.batch_norm_1d = nn.BatchNorm1d(out_channels)
        self.act_fn        = act_fn

    def forward(self, x) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm_1d(x)
        x = self.act_fn(x)
        return x
    

class MyLinear2D(nn.Module):
    def __init__(self, in_channels, out_channels, num_features, act_fn=nn.SiLU()) -> None:
        super().__init__()
        self.linear        = nn.Linear(in_channels, out_channels)
        self.batch_norm_1d = nn.BatchNorm1d(num_features)
        self.act_fn        = act_fn

    def forward(self, x) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm_1d(x)
        x = self.act_fn(x)
        return x


class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=2, act_fn=nn.SiLU()) -> None:
        super().__init__()
        self.conv_2d       = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=1)
        self.batch_norm_2d = nn.BatchNorm2d(out_channels)
        self.act_fn        = act_fn

    def forward(self, x) -> torch.Tensor:
        x = self.conv_2d(x)
        x = self.batch_norm_2d(x)
        x = self.act_fn(x)
        return x


class VideoProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            MyConv2d( 3,  8),        # (BS, 720, 1280,  3) --> (BS, 360, 640,  8)
            MyConv2d( 8, 16),        # (BS, 360,  640,  8) --> (BS, 180, 320, 16)
            MyConv2d(16, 24),        # (BS, 180,  320, 16) --> (BS,  90, 160, 24)
            MyConv2d(24, 24),        # (BS,  90,  160, 24) --> (BS,  45,  80, 24)
            MyConv2d(24, 32),        # (BS,  45,   80, 24) --> (BS,  22,  40, 32)
            MyConv2d(32, 32),        # (BS,  23,   40, 32) --> (BS,  12,  20, 32)
            nn.Flatten(),            # (BS,  12,   20, 32) --> (BS,         7680)
            MyLinear1D(7680, 4096),  # (BS,          7680) --> (BS,         4096)
            MyLinear1D(4096, 2048),  # (BS,          4096) --> (BS,         2048)
            MyLinear1D(2048, 1024),  # (BS,          2048) --> (BS,         1024)
            MyLinear1D(1024,  512),  # (BS,          1024) --> (BS,          512)
            MyLinear1D( 512,  256),  # (BS,           512) --> (BS,          256)
            MyLinear1D( 256,  128),  # (BS,           256) --> (BS,          128)
        )

    def forward(self, x) -> torch.Tensor:
        x = self.sequential(x)
        return x


class DataProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            MyLinear1D( 81, 128),  # (BS,  81) --> (BS, 128)
            MyLinear1D(128, 256),  # (BS, 128) --> (BS, 256)
            MyLinear1D(256, 512),  # (BS, 256) --> (BS, 512)
            MyLinear1D(512, 256),  # (BS, 512) --> (BS, 256)
            MyLinear1D(256, 128),  # (BS, 256) --> (BS, 128)
        )

    def forward(self, input) -> torch.Tensor:
        x = input
        x = self.sequential(x)
        return x
    

class MyProcessor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.video_processor = VideoProcessor()
        self.data_processor  = DataProcessor()

    def forward(self, video, data) -> torch.Tensor:
        video  = self.video_processor(video)         # (BS, 720,  1280,   3) --> (BS,   128)
        data   = self.data_processor(data)           # (BS,              81) --> (BS,   128)
        output = torch.concat([video, data], dim=1)  # (BS, 128) + (BS, 128) --> (BS,   256)
        return output


class MyModel(nn.Module):
    def __init__(self, length) -> None:
        super().__init__()
        self.length       = length
        self.my_processor = MyProcessor()
        self.sequential   = nn.Sequential(
            MyLinear1D(length*256, length*128),  # (BS, length * 256) --> (BS, length * 128)
            MyLinear1D(length*128, length* 64),  # (BS, length * 128) --> (BS, length *  64)
            MyLinear1D(length* 64, length* 32),  # (BS, length *  64) --> (BS, length *  32)
            MyLinear1D(length* 32, length* 16),  # (BS, length *  32) --> (BS, length *  16)
        )
        self.softmax_linear = MyLinear2D(16, 3, length, nn.Softmax(dim=-1))

    def forward(self, images, datas) -> torch.Tensor:
        images, datas = images.reshape((-1, 3, 720, 1280)), datas.reshape((-1, 81))
        x = [ self.my_processor(images, datas) ]
        x = torch.concat(x, dim=1)             # (BS *length,  256)
        x = x.reshape((-1, self.length, 256))  # (BS, length,  256)

        x = torch.flatten(x, start_dim=1)      # (BS, length,  256) --> (BS, length * 256)
        x = self.sequential(x)                 # (BS, length * 256) --> (BS, length *  64)
        x = x.reshape((-1, self.length,  16))  # (BS, length *  16) --> (BS, length,   16)
        x = self.softmax_linear(x)             # (BS, length,   16) --> (BS, length,    3)
        x = x.transpose(dim0=1, dim1=2)
        return x