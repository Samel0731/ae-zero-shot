import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """ Autoencoder learned multiple heads with specific ripple shape distributions in the latent space
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward_encoder(self, x):
        return x

    def forward_decoder(self, x):
        return x
    
    def forward_loss(self, imgs, pred):
        loss = 1
        return loss
    
    def forward(self, imgs):
        latent = self.forward_encoder(imgs)
        self.latent = latent
        pred = self.forward_decoder(latent)
        loss = self.forward_loss(imgs, pred)
        return loss, pred


class Encoder(nn.Module):
    """ Inspired by VGG Structure
    """
    def __init__(self):
        super().__init__()

        self.layer1 = EncoderLayer(in_channels=1, out_channels=64)
        self.layer2 = EncoderLayer(in_channels=64, out_channels=128)
        self.layer3 = EncoderLayer(in_channels=128, out_channels=256)
        self.layer4 = EncoderLayer(in_channels=256, out_channels=512)

    def forward(self):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = DecoderLayer(in_channels=512, out_channels=256)
        self.layer2 = DecoderLayer(in_channels=256, out_channels=128)
        self.layer3 = DecoderLayer(in_channels=128, out_channels=64)
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=1, padding=0) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2, padding=0, output_padding=0):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.relu(x)
        return x
