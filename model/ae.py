import torch
import torch.nn as nn
from .convnext import ConvNeXtVerEncoder

class Autoencoder(nn.Module):
    """ Autoencoder learned multiple heads with specific ripple shape distributions in the latent space
    """
    def __init__(self, encoder:str) -> None:
        super().__init__()
        if encoder == 'convnext':
            self.encoder = ConvNeXtVerEncoder()
        else:
            self.encoder = Encoder()
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
