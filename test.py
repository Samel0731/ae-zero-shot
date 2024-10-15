# Evaluate the model
import lpips
import torch
import torch.nn as nn
import numpy as np
from math import ceil
from model import build_AE
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.loader import TestDataset


def evaluate_autoencoder(autoencoder, test_loader, device):
    """
    Evaluate the autoencoder on the test set using PSNR, SSIM, and LPIPS.
    
    Parameters:
    - autoencoder: Trained autoencoder model
    - test_loader: DataLoader for the test dataset
    - device: Device to run the evaluation on (cpu/cuda)
    
    Returns:
    - Average PSNR, SSIM, LPIPS scores
    """
    # Initialize the LPIPS loss model
    lpips_loss_fn = lpips.LPIPS(net='vgg')  # You can use 'alex', 'vgg', or 'squeeze'

    psnr_total = 0
    ssim_total = 0
    lpips_total = 0
    psnr_history = []
    ssim_history = []
    lpips_history = []
    num_batches = 0
    
    autoencoder.eval()  # Set model to evaluation mode

    with torch.no_grad():  # No need for gradients during evaluation
        for imgs, _ in test_loader:
            imgs = imgs.to(device)
            
            # Forward pass through the autoencoder
            _, recon_imgs = autoencoder(imgs)

            # Move to CPU and convert to numpy for PSNR and SSIM
            imgs_np = imgs.squeeze().cpu().numpy()
            recon_imgs_np = recon_imgs.squeeze().cpu().numpy()
            
            # Compute PSNR and SSIM for each image
            for i in range(imgs.size(0)):
                psnr_value = psnr(imgs_np[i], recon_imgs_np[i], data_range=imgs_np[i].max() - imgs_np[i].min())
                ssim_value = ssim(imgs_np[i], recon_imgs_np[i], data_range=imgs_np[i].max() - imgs_np[i].min())
                
                psnr_total += psnr_value
                ssim_total += ssim_value
                psnr_history.append(psnr_value)
                ssim_history.append(ssim_value)
                
                # LPIPS needs the images to be in 3 channels, so repeat grayscale images across channels
                imgs_3ch = imgs[i].repeat(3, 1, 1).unsqueeze(0)  # Shape (1, 3, 200, 200)
                recon_imgs_3ch = recon_imgs[i].repeat(3, 1, 1).unsqueeze(0)
                
                # Ensure the LPIPS model is also on the correct device
                lpips_loss_fn = lpips_loss_fn.to(device)
                lpips_value = lpips_loss_fn(imgs_3ch, recon_imgs_3ch)
                lpips_total += lpips_value.item()
                lpips_history.append(lpips_value.item())

            num_batches += imgs.size(0)

    # Calculate average scores
    psnr_avg = psnr_total / num_batches
    ssim_avg = ssim_total / num_batches
    lpips_avg = lpips_total / num_batches

    psnr_std = np.std(psnr_history)
    ssim_std = np.std(ssim_history)
    lpips_std = np.std(lpips_history)

    return psnr_avg, psnr_std, ssim_avg, ssim_std, lpips_avg, lpips_std


def main():
    # --------------------------------------------------------------------------
    # Preparing data
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((200,200), Image.BICUBIC),   # Resize images to 200x200
        transforms.ToTensor(),  # Convert images to tensor (values between 0 and 1)
    ])
    testing_set = TestDataset('./data', transform=transform)
    testing_loader = DataLoader(testing_set, batch_size=batch_size, shuffle=False)
    iters_per_epoch = ceil(len(testing_set) / batch_size)
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Load the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    autoencoder = build_AE(encoder_type='convnext')  # Make sure this matches the original model architecture
    autoencoder.load_state_dict(torch.load('saved/models/model_20240928_204659_99.pt', weights_only=True))
    autoencoder.to(device)
    autoencoder50 = build_AE(encoder_type='convnext')  # Make sure this matches the original model architecture
    autoencoder50.load_state_dict(torch.load('saved/models/model_20240928_204659_50.pt', weights_only=True))
    autoencoder50.to(device)
    # --------------------------------------------------------------------------

    psnr_avg, psnr_std, ssim_avg, ssim_std, lpips_avg, lpips_std = evaluate_autoencoder(autoencoder, testing_loader, device=device)
    print(f"Model: {97}")
    print(f"Average PSNR: {psnr_avg:.4f}±{psnr_std:.4f}")
    print(f"Average SSIM: {ssim_avg:.4f}±{ssim_std:.4f}")
    print(f"Average LPIPS: {lpips_avg:.4f}±{lpips_std:.4f}")

    psnr_avg, psnr_std, ssim_avg, ssim_std, lpips_avg, lpips_std = evaluate_autoencoder(autoencoder50, testing_loader, device=device)
    print(f"Model: {50}")
    print(f"Average PSNR: {psnr_avg:.4f}±{psnr_std:.4f}")
    print(f"Average SSIM: {ssim_avg:.4f}±{ssim_std:.4f}")
    print(f"Average LPIPS: {lpips_avg:.4f}±{lpips_std:.4f}")


if __name__=='__main__':
    main()
