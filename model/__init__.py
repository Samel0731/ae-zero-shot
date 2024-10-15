from .ae import *
from torchsummary import summary


def build_AE(encoder_type:str='convnext') -> Autoencoder:
    return Autoencoder(encoder=encoder_type)


def test_model_architecture():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = build_AE(encoder_type='convnext')
    autoencoder.to(device)
    channels, H, W = 1, 200, 200
    x = torch.randn(1,1,200,200).to(device)
    latent, recon = autoencoder(x)
    print(f"[{device}] latent space shape: {latent.shape}, output shape: {recon.shape}")
    summary(autoencoder, input_size=(channels, H, W))