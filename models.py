""" Libraries """
import torch
import torch.nn as nn



""" Classes """
class CheckShape(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.shape = shape

    def forward(self, x) -> torch.Tensor:
        # print(x.shape)
        assert x.shape[1:] == self.shape
        return x


class MyLinearBN1D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 act_fn=nn.SiLU(), dropout=0.15) -> None:
        super().__init__()
        self.linear     = nn.Linear(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.act_fn     = act_fn
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.linear(x)
        x = self.batch_norm(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MyLinearBN2D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 act_fn=nn.SiLU(), dropout=0.15) -> None:
        super().__init__()
        self.linear     = nn.Linear(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm1d(out_channels)
        self.act_fn     = act_fn
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.linear(x)
        x = torch.permute(x, dims=(0,2,1))
        x = self.batch_norm(x)
        x = torch.permute(x, dims=(0,2,1))
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MyLinearBN3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 act_fn=nn.SiLU(), dropout=0.15) -> None:
        super().__init__()
        self.linear     = nn.Linear(in_channels, out_channels)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.act_fn     = act_fn
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.linear(x)
        x = torch.permute(x, dims=(0,3,1,2))
        x = self.batch_norm(x)
        x = torch.permute(x, dims=(0,2,3,1))
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3, stride=2, act_fn=nn.SiLU(), dropout=0.15) -> None:
        super().__init__()
        self.conv_2d       = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=1)
        self.batch_norm_2d = nn.BatchNorm2d(out_channels)
        self.act_fn        = act_fn
        self.dropout       = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_2d(x)
        x = self.batch_norm_2d(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MyConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3,
                 stride=1, padding=1, act_fn=nn.SiLU(), dropout=0.15) -> None:
        super().__init__()
        self.conv_3d       = nn.Conv3d(in_channels, out_channels, kernel, stride, padding=padding)
        self.batch_norm_3d = nn.BatchNorm3d(out_channels)
        self.act_fn        = act_fn
        self.dropout       = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_3d(x)
        x = self.batch_norm_3d(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class MultiOutputModel(nn.Module):
    def __init__(self, length, output_dim, softmax) -> None:
        super().__init__()
        self.length  = length
        self.softmax = softmax
        self.imgs_processor = nn.Sequential(
            MyConv3d(   2,   32, kernel=(3,7,7), stride=(1,2,2)),  # (BS,    2, length, 64, 64) --> (BS,   32, length, 32, 32)
            CheckShape((  32, length, 32, 32)),
            MyConv3d(  32,   64, kernel=(3,5,5), stride=(1,2,2)),  # (BS,   32, length, 32, 32) --> (BS,   64, length, 16, 16)
            CheckShape((  64, length, 16, 16)),
            MyConv3d(  64,  128, kernel=(3,3,3), stride=(1,2,2)),  # (BS,   64, length, 16, 16) --> (BS,  128, length,  8,  8)
            CheckShape(( 128, length,  8,  8)),
            MyConv3d( 128,  256, kernel=(3,3,3), stride=(1,2,2)),  # (BS,  128, length,  8,  8) --> (BS,  256, length,  4,  4)
            CheckShape(( 256, length,  4,  4)),
            MyConv3d( 256,  512, kernel=(3,3,3), stride=(1,2,2)),  # (BS,  256, length,  4,  4) --> (BS,  512, length,  2,  2)
            CheckShape(( 512, length,  2,  2)),
            MyConv3d( 512, 1024, kernel=(3,3,3), stride=(1,2,2)),  # (BS,  512, length,  2,  2) --> (BS, 1024, length,  1,  1)
            CheckShape((1024, length,  1,  1)),
            nn.Flatten(start_dim=-3),                              # (BS, 1024, length,  1,  1) --> (BS, 1024, length        )
            CheckShape((1024, length        )),
        )
        self.kpts_processor = nn.Sequential(
            MyLinearBN3D(17,  32),     # (BS, length, 4,  17) --> (BS, length, 4,  32)
            MyLinearBN3D(32,  64),     # (BS, length, 4,  32) --> (BS, length, 4,  64)
            MyLinearBN3D(64, 128),     # (BS, length, 4,  64) --> (BS, length, 4, 128)
            nn.Flatten(start_dim=-2),  # (BS, length, 4, 128) --> (BS, length,    512)
            MyLinearBN2D(512, 256),    # (BS, length,    512) --> (BS, length,    256)
            MyLinearBN2D(256, 128),    # (BS, length,    256) --> (BS, length,    128)
        )
        self.balls_processor = nn.Sequential(
            MyLinearBN2D( 2,  8),  # (BS, length,  2) --> (BS, length,  8)
            MyLinearBN2D( 8, 16),  # (BS, length,  8) --> (BS, length, 16)
            MyLinearBN2D(16, 32),  # (BS, length, 16) --> (BS, length, 32)
        )
        self.times_processor = nn.Sequential(
            MyLinearBN2D( 1,  6),  # (BS, length,  1) --> (BS, length,  6)
            MyLinearBN2D( 6, 16),  # (BS, length,  6) --> (BS, length, 16)
            MyLinearBN2D(16, 32),  # (BS, length, 16) --> (BS, length, 32)
        )
        self.bg_id_processor = nn.Sequential(
            MyLinearBN2D(13, 24),  # (BS, length, 12) --> (BS, length, 24)
            MyLinearBN2D(24, 32),  # (BS, length, 24) --> (BS, length, 32)
        )
        self.lstm = nn.LSTM(
            input_size=1024+128+32+32+32,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.sequential = nn.Sequential(
            MyLinearBN2D(512,        128),  # (BS, length, 512) --> (BS, length,        128)
            MyLinearBN2D(128,         32),  # (BS, length, 128) --> (BS, length,         32)
            nn.Linear(    32, output_dim),  # (BS, length,  32) --> (BS, length, output_dim)
            nn.Sigmoid(),
        )

    def forward(self, imgs, kpts, balls, times, bg_id) -> torch.Tensor:
        x = torch.concat([
            torch.permute(self.imgs_processor(imgs), dims=(0,2,1)),
            self.kpts_processor(kpts),
            self.balls_processor(balls),
            self.times_processor(torch.unsqueeze(times, dim=-1)),
            self.bg_id_processor(torch.concat([torch.unsqueeze(bg_id, dim=1)]*self.length, dim=1)),
        ], dim=-1)
        x = self.sequential(self.lstm(x)[0])
        if self.softmax: x = nn.Softmax(dim=-1)(x)
        return x


class SingleOutputModel(nn.Module):
    def __init__(self, length, output_dim, softmax, sigmoid=True) -> None:
        super().__init__()
        self.length  = length
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.imgs_processor = nn.Sequential(
            MyConv3d(   2,   32, (3,7,7), (1,2,2), (1,3,3)),  # (BS,    2, length, 64, 64) --> (BS,   32, length, 32, 32)
            CheckShape((  32, length, 32, 32)),
            MyConv3d(  32,   64, (3,5,5), (1,2,2), (1,2,2)),  # (BS,   32, length, 32, 32) --> (BS,   64, length, 16, 16)
            CheckShape((  64, length, 16, 16)),
            MyConv3d(  64,  128, (3,3,3), (1,2,2)),           # (BS,   64, length, 16, 16) --> (BS,  128, length,  8,  8)
            CheckShape(( 128, length,  8,  8)),
            MyConv3d( 128,  256, (3,3,3), (1,2,2)),           # (BS,  128, length,  8,  8) --> (BS,  256, length,  4,  4)
            CheckShape(( 256, length,  4,  4)),
            MyConv3d( 256,  512, (3,3,3), (1,2,2)),           # (BS,  256, length,  4,  4) --> (BS,  512, length,  2,  2)
            CheckShape(( 512, length,  2,  2)),
            MyConv3d( 512, 1024, (3,3,3), (1,2,2)),           # (BS,  512, length,  2,  2) --> (BS, 1024, length,  1,  1)
            CheckShape((1024, length,  1,  1)),
            nn.Flatten(start_dim=-3),                         # (BS, 1024, length,  1,  1) --> (BS, 1024, length        )
            CheckShape((1024, length        )),
        )
        self.kpts_processor = nn.Sequential(
            MyLinearBN3D(17,  32),     # (BS, length, 4,  17) --> (BS, length, 4,  32)
            MyLinearBN3D(32,  64),     # (BS, length, 4,  32) --> (BS, length, 4,  64)
            MyLinearBN3D(64, 128),     # (BS, length, 4,  64) --> (BS, length, 4, 128)
            nn.Flatten(start_dim=-2),  # (BS, length, 4, 128) --> (BS, length,    512)
            MyLinearBN2D(512, 256),    # (BS, length,    512) --> (BS, length,    256)
            MyLinearBN2D(256, 128),    # (BS, length,    256) --> (BS, length,    128)
        )
        self.balls_processor = nn.Sequential(
            MyLinearBN2D( 2,  8),  # (BS, length,  2) --> (BS, length,  8)
            MyLinearBN2D( 8, 16),  # (BS, length,  8) --> (BS, length, 16)
            MyLinearBN2D(16, 32),  # (BS, length, 16) --> (BS, length, 32)
        )
        self.times_processor = nn.Sequential(
            MyLinearBN2D( 1,  6),  # (BS, length,  1) --> (BS, length,  6)
            MyLinearBN2D( 6, 16),  # (BS, length,  6) --> (BS, length, 16)
            MyLinearBN2D(16, 32),  # (BS, length, 16) --> (BS, length, 32)
        )
        self.bg_id_processor = nn.Sequential(
            MyLinearBN2D(13, 24),  # (BS, length, 12) --> (BS, length, 24)
            MyLinearBN2D(24, 32),  # (BS, length, 24) --> (BS, length, 32)
        )
        self.hitter_processor = nn.Sequential(
            MyLinearBN2D( 2,  8),  # (BS, length,  2) --> (BS, length,  8)
            MyLinearBN2D( 8, 16),  # (BS, length,  8) --> (BS, length, 16)
            MyLinearBN2D(16, 32),  # (BS, length, 16) --> (BS, length, 32)
        )
        self.lstm = nn.LSTM(
            input_size=1024+128+32+32+32+32,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.sequential = nn.Sequential(
            MyLinearBN2D(512, 128),               # (BS, length, 512) --> (BS, length,128)
            MyLinearBN2D(128,  32),               # (BS, length, 128) --> (BS, length, 32)
            nn.Flatten(),                         # (BS, length,  32) --> (BS,  length*32)
            MyLinearBN1D(length*32,        256),  # (BS,         ???) --> (BS,        256)
            MyLinearBN1D(      256,         64),  # (BS,         256) --> (BS,         64)
            MyLinearBN1D(       64,         16),  # (BS,          64) --> (BS,         16)
            nn.Linear(          16, output_dim),  # (BS,          16) --> (BS, output_dim)
            # nn.BatchNorm1d(18),
        )

    def forward(self, imgs, kpts, balls, times, bg_id, hitter) -> torch.Tensor:
        x = torch.concat([
            torch.permute(self.imgs_processor(imgs), dims=(0,2,1)),
            self.kpts_processor(kpts),
            self.balls_processor(balls),
            self.times_processor(torch.unsqueeze(times, dim=-1)),
            self.bg_id_processor(torch.concat([torch.unsqueeze(bg_id, dim=1)]*self.length, dim=1)),
            self.hitter_processor(torch.concat([torch.unsqueeze(hitter, dim=1)]*self.length, dim=1)),
        ], dim=-1)
        x = self.sequential(self.lstm(x)[0])
        if self.sigmoid: x = nn.Sigmoid()(x)
        if self.softmax: x = nn.Softmax(dim=-1)(x)
        return x
    

class MyResConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel=3,
                 stride=1, padding=1, act_fn=nn.SiLU(), dropout=0.15) -> None:
        super().__init__()
        self.in_channels   = in_channels
        self.conv_3d       = nn.Conv3d(in_channels, out_channels, kernel, stride, padding=padding)
        self.batch_norm_3d = nn.BatchNorm3d(out_channels)
        self.act_fn        = act_fn
        self.dropout       = nn.Dropout(dropout)

    def forward(self, x) -> torch.Tensor:
        x_tmp = self.conv_3d(x)
        x_tmp[:, :, :, self.in_channels] += x
        x = x_tmp
        x = self.batch_norm_3d(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        return x


class LocationModel(nn.Module):
    def __init__(self, length, output_dim, softmax, sigmoid=True) -> None:
        super().__init__()
        self.length  = length
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.imgs_processor = nn.Sequential(
            MyConv3d(   2,   32, (3,7,7), (1,2,2), (1,3,3)),  # (BS,    2, length, 64, 64) --> (BS,   32, length, 32, 32)
            CheckShape((  32, length, 32, 32)),
            MyConv3d(  32,   64, (3,5,5), (1,2,2), (1,2,2)),  # (BS,   32, length, 32, 32) --> (BS,   64, length, 16, 16)
            CheckShape((  64, length, 16, 16)),
            MyConv3d(  64,  128, (3,3,3), (1,2,2)),           # (BS,   64, length, 16, 16) --> (BS,  128, length,  8,  8)
            CheckShape(( 128, length,  8,  8)),
            MyConv3d( 128,  256, (3,3,3), (1,2,2)),           # (BS,  128, length,  8,  8) --> (BS,  256, length,  4,  4)
            CheckShape(( 256, length,  4,  4)),
            MyConv3d( 256,  512, (3,3,3), (1,2,2)),           # (BS,  256, length,  4,  4) --> (BS,  512, length,  2,  2)
            CheckShape(( 512, length,  2,  2)),
            MyConv3d( 512, 1024, (3,3,3), (1,2,2)),           # (BS,  512, length,  2,  2) --> (BS, 1024, length,  1,  1)
            CheckShape((1024, length,  1,  1)),
            nn.Flatten(start_dim=-3),                         # (BS, 1024, length,  1,  1) --> (BS, 1024, length        )
            CheckShape((1024, length        )),
        )
        self.kpts_processor = nn.Sequential(
            MyLinearBN3D(17,  32),     # (BS, length, 4,  17) --> (BS, length, 4,  32)
            MyLinearBN3D(32,  64),     # (BS, length, 4,  32) --> (BS, length, 4,  64)
            MyLinearBN3D(64, 128),     # (BS, length, 4,  64) --> (BS, length, 4, 128)
            nn.Flatten(start_dim=-2),  # (BS, length, 4, 128) --> (BS, length,    512)
            MyLinearBN2D(512, 256),    # (BS, length,    512) --> (BS, length,    256)
            MyLinearBN2D(256, 128),    # (BS, length,    256) --> (BS, length,    128)
        )
        self.balls_processor = nn.Sequential(
            MyLinearBN2D( 2,  8),  # (BS, length,  2) --> (BS, length,  8)
            MyLinearBN2D( 8, 16),  # (BS, length,  8) --> (BS, length, 16)
            MyLinearBN2D(16, 32),  # (BS, length, 16) --> (BS, length, 32)
        )
        self.times_processor = nn.Sequential(
            MyLinearBN2D( 1,  6),  # (BS, length,  1) --> (BS, length,  6)
            MyLinearBN2D( 6, 16),  # (BS, length,  6) --> (BS, length, 16)
            MyLinearBN2D(16, 32),  # (BS, length, 16) --> (BS, length, 32)
        )
        self.bg_id_processor = nn.Sequential(
            MyLinearBN2D(13, 24),  # (BS, length, 12) --> (BS, length, 24)
            MyLinearBN2D(24, 32),  # (BS, length, 24) --> (BS, length, 32)
        )
        self.hitter_processor = nn.Sequential(
            MyLinearBN2D( 2,  8),  # (BS, length,  2) --> (BS, length,  8)
            MyLinearBN2D( 8, 16),  # (BS, length,  8) --> (BS, length, 16)
            MyLinearBN2D(16, 32),  # (BS, length, 16) --> (BS, length, 32)
        )
        self.lstm = nn.LSTM(
            input_size=1024+128+32+32+32+32,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.sequential = nn.Sequential(
            MyLinearBN2D(512, 128),               # (BS, length, 512) --> (BS, length,128)
            MyLinearBN2D(128,  32),               # (BS, length, 128) --> (BS, length, 32)
            nn.Flatten(),                         # (BS, length,  32) --> (BS,  length*32)
            MyLinearBN1D(length*32,        256),  # (BS,         ???) --> (BS,        256)
            MyLinearBN1D(      256,         64),  # (BS,         256) --> (BS,         64)
            MyLinearBN1D(       64,         16),  # (BS,          64) --> (BS,         16)
            nn.Linear(          16, output_dim),  # (BS,          16) --> (BS, output_dim)
            # nn.BatchNorm1d(18),
        )

    def forward(self, imgs, kpts, balls, times, bg_id, hitter) -> torch.Tensor:
        x = torch.concat([
            torch.permute(self.imgs_processor(imgs), dims=(0,2,1)),
            self.kpts_processor(kpts),
            self.balls_processor(balls),
            self.times_processor(torch.unsqueeze(times, dim=-1)),
            self.bg_id_processor(torch.concat([torch.unsqueeze(bg_id, dim=1)]*self.length, dim=1)),
            self.hitter_processor(torch.concat([torch.unsqueeze(hitter, dim=1)]*self.length, dim=1)),
        ], dim=-1)
        x = self.sequential(self.lstm(x)[0])
        if self.sigmoid: x = nn.Sigmoid()(x)
        if self.softmax: x = nn.Softmax(dim=-1)(x)
        return x