from typing import List
from torch import nn
from chord.model.convnext import ConvNeXtBlock, TimeDownsamplingBlock
import torch
from einops import rearrange
import gin


class OctavePool(nn.Module):
    r"""Average log-frequency axis across octaves, thus producing a chromagram."""
    def __init__(self, bins_per_octave):
        super().__init__()
        self.bins_per_octave = bins_per_octave

    def forward(self, x):
        # x: (batch_size, channel, H, W)
        x = rearrange(x, "B C (j k) W -> B C k j W", k=self.bins_per_octave)
        x = x.mean(dim=3)
        return x


@gin.configurable
class ChromaNet(nn.Module):
    def __init__(self,
                 n_bins: int,
                 n_harmonics: int,
                 out_channels: List[int],
                 kernels: List[int],
                 temperature: float,
                 pool_type: str="octave"
                 ):
        super().__init__()
        assert pool_type == "octave" or "fully_connected"
        assert len(kernels) == len(out_channels)
        self.n_harmonics = n_harmonics
        self.n_bins = n_bins
        in_channel = self.n_harmonics
        self.out_channels = out_channels
        self.kernels = kernels
        self.temperature = temperature
        self.drop_path = 0.1
        self.pool_type = pool_type
        convnext_blocks = []
        time_downsampling_blocks = []
        for i, out_channel in enumerate(self.out_channels):
            time_downsampling_block = TimeDownsamplingBlock(in_channel, out_channel)
            kernel = self.kernels[i]
            convnext_block = ConvNeXtBlock(out_channel, out_channel, kernel_size=kernel, padding=kernel//2, drop_path = self.drop_path)
            time_downsampling_blocks.append(time_downsampling_block)
            convnext_blocks.append(convnext_block)
            in_channel = out_channel
        self.convnext_blocks = nn.ModuleList(convnext_blocks)
        self.time_downsampling_blocks = nn.ModuleList(time_downsampling_blocks)
        if self.pool_type == "octave":
            self.octave_pool = OctavePool(12)
        else:
        # TODO: change the hard coded 20 here, it's related to audio length
            self.fully_connected = nn.Linear(self.n_bins*20, 12*20)
        self.global_average_pool = nn.AdaptiveAvgPool2d((12, 1))
        self.classifier = nn.Conv2d(out_channel, 2, kernel_size=(1, 1))
        self.flatten = nn.Flatten()
        self.batch_norm = nn.BatchNorm2d(2, affine=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        block_zip = zip(self.convnext_blocks, self.time_downsampling_blocks)
        for convnext_block, time_downsampling_block in block_zip:
            x = time_downsampling_block(x)
            x = convnext_block(x)
        if self.pool_type == "octave":
            x = self.octave_pool(x)
        else:
            batch_size, channels, _, w = x.size()
            x = x.view(batch_size, channels, -1)
            x = self.fully_connected(x)
            x = x.view(batch_size, channels, 12, w)
        x = self.global_average_pool(x)
        x = self.classifier(x) # before: B, C, n_bins, T, after: B, C, n_bins, 1
        x = self.batch_norm(x)

        x = self.flatten(x)
        x = self.softmax(x/self.temperature)
        assert x.shape[1] == 24

        return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, dim_h, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm((dim_h, dim))  # norm layer
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 3, 2, 1)  # (N, C, F, T) -> (N, T, F, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 2, 1)  # (N, T, F, C) -> (N, C, F, T)

        x = input + x
        return x

class TimeDownsamplingLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dim_h, downsample_rate=2):
        super().__init__()
        self.norm = nn.LayerNorm((dim_h, in_channels))
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, downsample_rate), stride=(1, downsample_rate))
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = x.permute(0, 3, 2, 1)  # (N, C, F, T) -> (N, T, F, C)
        x = self.norm(x)
        x = x.permute(0, 3, 2, 1)  # (N, T, F, C) -> (N, C, F, T)

        x = self.conv(x)
        x = self.act(x)
        return x

class ChromaNetSeq2Seq(nn.Module):
    def __init__(
        self,

        num_octaves: int = 7,
        bins_per_octave: int = 12,

        out_channels: List[int] = [2, 3, 40, 40, 30, 10, 3],
        downsample_rate: List[int] = [2, 2, 2, 2, 1, 1, 1],
    ):
        super().__init__()

        self.num_octaves = num_octaves
        self.bins_per_octave = bins_per_octave
        self.n_bins = num_octaves * bins_per_octave

        blocks = []

        in_channels = 1
        for out_channel, downsample_rate in zip(out_channels, downsample_rate):
            blocks.append(
                nn.Sequential(
                    TimeDownsamplingLayer(in_channels, out_channel, dim_h=self.n_bins, downsample_rate=downsample_rate),
                    Block(out_channel, dim_h=self.n_bins)
                )
            )
            in_channels = out_channel
        self.blocks = nn.Sequential(*blocks)

        self.classifier = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_octaves * bins_per_octave, seq_len)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 12)
        """
        assert x.shape[1] == self.num_octaves * self.bins_per_octave
        x = x.unsqueeze(dim=1)

        for block in self.blocks:
            x = block(x)

        x = x.reshape(x.shape[0], x.shape[1], self.num_octaves, self.bins_per_octave, x.shape[-1])
        x = x.mean(dim=2)  # (N, C, O, B, T) -> (N, C, B, T)

        x = self.classifier(x)  # (N, C, B, T) -> (N, 1, B, T)
        x = torch.softmax(x, dim=2)
        return x.squeeze(dim=1)  # (N, 1, B, T) -> (N, B, T)
