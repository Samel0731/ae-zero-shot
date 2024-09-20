from .ae import *
from torchsummary import summary


def build_AE():
    return Autoencoder()


def test_model_architecture():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = build_AE()
    autoencoder.to(device)
    channels, H, W = 1, 200, 200
    x = torch.randn(1,1,200,200).to(device)
    latent, recon = autoencoder(x)
    print(f"[{device}] latent space shape: {latent.shape}, output shape: {recon.shape}")
    summary(autoencoder, input_size=(channels, H, W))