import torch
from model import build_AE


def torch_check_GPU():
    # --------------------------------------------------------------------------
    # check GPU
    torch.cuda.is_available()
    torch.cuda.device_count()
    torch.cuda.current_device()
    torch.cuda.device(0)
    torch.cuda.get_device_name(0)
    # --------------------------------------------------------------------------


def main():
    autoencoder = build_AE()


if __name__=='__main__':
    main()
