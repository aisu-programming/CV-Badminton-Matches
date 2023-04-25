""" Libraries """
import torch
import torch.nn as nn
import torchvision


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


class ImageProcessor(nn.Module):
    def __init__(self, length) -> None:
        super().__init__()
        self.length = length
        self.resnet       = torchvision.models.resnet50(width_per_group=500, num_classes=length*64)
        self.resnet.conv1 = nn.Conv2d((length+1)*3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.bn1   = nn.BatchNorm2d(64)
        self.multihead_attn = nn.MultiheadAttention(64, 8, dropout=0.1, batch_first=True)

    def forward(self, x) -> torch.Tensor:
        shape = (-1, self.length, 64)
        x = self.resnet(x)                   # (BS, (length+1)*3, 500, 500) --> (BS,  length*64)
        x = torch.reshape(x, shape)          # (BS,              length*64) --> (BS, length, 64)
        x, _ = self.multihead_attn(x, x, x)
        return x


class DataProcessor(nn.Module):
    def __init__(self, length) -> None:
        super().__init__()
        self.sequential = nn.Sequential(
            MyLinear2D( 81, 128, length),  # (BS, length,  81) --> (BS, length, 128)
            MyLinear2D(128, 256, length),  # (BS, length, 128) --> (BS, length, 256)
            MyLinear2D(256, 128, length),  # (BS, length, 256) --> (BS, length, 128)
            MyLinear2D(128,  64, length),  # (BS, length, 128) --> (BS, length,  64)
        )
        self.multihead_attn = nn.MultiheadAttention(64, 8, dropout=0.1, batch_first=True)

    def forward(self, x) -> torch.Tensor:
        x = self.sequential(x)
        self.multihead_attn(x, x, x)
        return x


class MyProcessor(nn.Module):
    def __init__(self, length) -> None:
        super().__init__()
        self.image_processor = ImageProcessor(length)
        self.data_processor  = DataProcessor(length)

    def forward(self, images, datas) -> torch.Tensor:
        images = self.image_processor(images)          # (BS, length*3, 360, 640) --> (BS, length,  64)
        datas  = self.data_processor(datas)            # (BS,   length,       81) --> (BS, length,  64)
        output = torch.concat([images, datas], dim=2)  # (BS,   length,  64)   *2 --> (BS, length, 128)
        return output


class MyModel(nn.Module):
    def __init__(self, length) -> None:
        super().__init__()
        self.my_processor = MyProcessor(length)
        self.sequential   = nn.Sequential(
            MyLinear2D(128,  64, length),             # (BS, length, 128) --> (BS, length,  64)
            MyLinear2D( 64,  32, length),             # (BS, length,  64) --> (BS, length,  32)
            MyLinear2D( 32,  16, length),             # (BS, length,  32) --> (BS, length,  16)
            MyLinear2D( 16,   8, length),             # (BS, length,  16) --> (BS, length,   8)
            nn.Flatten(),                             # (BS, length, 8) --> (BS, length*8)
            nn.Dropout(0.2),
            MyLinear1D(length*8, 512),                # (BS,  length*8) --> (BS,      512)
            MyLinear1D(     512, 128),                # (BS,       512) --> (BS,      128)
            MyLinear1D(     128,  32),                # (BS,       128) --> (BS,       32)
            MyLinear1D(      32,   8),                # (BS,        32) --> (BS,        8)
            MyLinear1D(       8,   1, nn.Sigmoid()),  # (BS,         8) --> (BS,        1)
        )

    def forward(self, images, datas) -> torch.Tensor:
        x = self.my_processor(images, datas)  # (BS,         ...) --> (BS, length, 128)
        x = self.sequential(x)                # (BS, length, 128) --> (BS,           1)
        x = torch.squeeze(x)                  # (BS,           1) --> (BS             )
        return x