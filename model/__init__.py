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
    output = autoencoder(x)
    print(f"[{device}] output shape: {output.shape}")
    summary(autoencoder, input_size=(channels, H, W))