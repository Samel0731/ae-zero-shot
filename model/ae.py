import torch
import torch.nn as nn
import torch.nn.functional as F
from .weight_init import trunc_normal_
from .drop import DropPath

class Autoencoder(nn.Module):
    """ Autoencoder learned multiple heads with specific ripple shape distributions in the latent space
    """
    def __init__(self) -> None:
        super().__init__()
        self.encoder = ConvNeXtVerEncoder()
        self.decoder = Decoder()
    
    def forward(self, imgs):
        latent = self.encoder(imgs)
        pred = self.decoder(latent)
        return latent, pred


class Encoder(nn.Module):
    """ Inspired by VGG Structure
    """
    def __init__(self):
        super().__init__()

        self.layer1 = EncoderLayer(in_channels=1, out_channels=32, stride=2, batch_norm=False)
        self.layer2 = EncoderLayer(in_channels=32, out_channels=16, stride=2, batch_norm=False)
        # self.layer3 = EncoderLayer(in_channels=128, out_channels=256, batch_norm=False)
        # self.layer4 = EncoderLayer(in_channels=256, out_channels=512, batch_norm=False)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        # x = self.layer4(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        # Start with latent space 6x6x512
        # self.layer1 = DecoderLayer(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        # self.layer2 = DecoderLayer(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.layer2 = DecoderLayer(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1)
        self.layer3 = DecoderLayer(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.layer4 = DecoderLayer(in_channels=32, out_channels=32, kernel_size=5, stride=2, padding=1, output_padding=1)
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1, output_padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x = self.layer1(x)  # Upsample to 12x12x256
        x = self.layer2(x)  # Upsample to 24x24x128
        x = self.layer3(x)  # Upsample to 48x48x64
        x = self.layer4(x)  # Upsample to 100x100x64
        x = self.layer5(x)  # Final upsample to 200x200x1 (output size)
        return x
    

class ConvNeXtVerEncoder(nn.Module):
    def __init__(self, depths=[1,3], dims=[24, 48], drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()

        self.downsample_layer = nn.ModuleList() # stem and 1 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(1, dims[0], kernel_size=7, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layer.append(stem)
        downsample_layer = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=5, stride=4),
        )
        self.downsample_layer.append(downsample_layer)
        
        self.stages = nn.ModuleList() # 2 feature resolution stages, consisting of 1, 3 residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(2):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.head = nn.Conv2d(dims[-1], 16, kernel_size=1)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(2):
            x = self.downsample_layer[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


# Copied from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
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
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, batch_norm=False) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.batch_norm = batch_norm
        if self.batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels, 0.8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        if self.batch_norm:
            x = self.batch_norm(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.relu(x)
        return x


def init_weights(m):
    """ Custom weight initialization function """
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.01)
