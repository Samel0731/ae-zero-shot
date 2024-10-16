import argparse
import os
import torch
import sys
from math import ceil
from matplotlib import pyplot as plt
from model import build_AE, init_weights, test_model_architecture
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils.loader import KFoldDataset
from utils.openmax import compute_train_score_and_mavs_and_dists
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def torch_check_GPU():
    # --------------------------------------------------------------------------
    # check GPU
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.get_device_name(0))
    # --------------------------------------------------------------------------


def main(config_args:argparse.Namespace=None):
    dataset_root = './data'
    field_names = ['p_seabass', 'sea', 'barrel', 'lng', 'noon']

    # --------------------------------------------------------------------------
    # train setting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_number = 0

    EPOCHS = config_args.epochs

    best_vloss = 1_000_000.
    lr = 0.0001
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparing data
    batch_size = 16
    transform = transforms.Compose([
        transforms.Resize((200,200), Image.BICUBIC),   # Resize images to 200x200
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.ConvertImageDtype(torch.float32),
        transforms.ToTensor(),  # Convert images to tensor (values between 0 and 1)
    ])
    training_set = KFoldDataset(root_dir=dataset_root, field_names=field_names, k=0, train=True, transform=transform)
    validation_set = KFoldDataset(root_dir=dataset_root, field_names=field_names, k=0, train=False, transform=transform)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    iters_per_epoch = ceil(len(training_set) / batch_size)
    viters_per_epoch = ceil(len(validation_set) / batch_size)
    # --------------------------------------------------------------------------

    training_epoch_loss = []
    validation_epoch_loss = []
    autoencoder = build_AE(encoder_type='convnext')
    autoencoder.apply(init_weights)
    autoencoder.to(device)
    optimizer = torch.optim.AdamW(autoencoder.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    # training per-epoch activity
    for epoch in range(EPOCHS):
        print('EPOCH {}:'.format(epoch_number + 1))
        autoencoder.train(True)
        avg_loss = -1
        avg_vloss = -1

        # --------------------------------------------------------------------------
        # train one epoch
        train_loss = 0.
        pbar = tqdm(enumerate(training_loader), desc=f"Epoch: [{epoch+1}/{EPOCHS}] Iter: [{0}/{iters_per_epoch}]", total=iters_per_epoch)
        for iter, (inputs, labels, _) in pbar:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            _, outputs = autoencoder(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            pbar.set_description(f"Epoch(Training): [{epoch+1}/{EPOCHS}] Iter: [{iter+1}/{iters_per_epoch}] Loss: {train_loss / (iter+1):.8f}")
        train_loss /= len(training_loader)
        training_epoch_loss.append(train_loss)
        avg_loss = train_loss
        # --------------------------------------------------------------------------

        torch.cuda.empty_cache()
        running_vloss = 0.
        autoencoder.eval()
        vbar = tqdm(enumerate(validation_loader), desc=f"Epoch(Validating):[{epoch+1}/{EPOCHS}]", total=viters_per_epoch)
        with torch.no_grad():
            for viter, vdata in vbar:
                vinputs, vlabels, _ = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                _, voutputs = autoencoder(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
                vbar.set_description(f"Epoch(Validating): [{epoch+1}/{EPOCHS}] Iter: [{viter+1}/{viters_per_epoch}] Loss: {running_vloss / (viter+1):.8f}")

        avg_vloss = running_vloss / len(validation_loader)
        validation_epoch_loss.append(avg_vloss)
        print('LOSS train {:.8f} valid {:.8f}'.format(avg_loss, avg_vloss))

        # # Fit the Weibull distribution form training data (OpenMax Alg. 1: EVT Meta-Recognition Calibration)
        # _, mavs, dists = compute_train_score_and_mavs_and_dists(len(field_names), training_loader, autoencoder)

        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss and epoch > 20:
            best_vloss = avg_vloss
            model_path = 'saved/models/model_best_vloss.pt'
            torch.save(autoencoder.state_dict(), model_path)
        
        # Save the model's state every 10 epochs
        if (epoch % 50 == 0 or epoch+1==EPOCHS) and epoch > 0:
            model_path = 'saved/models/model_{}_{}.pt'.format(timestamp, epoch_number)
            torch.save(autoencoder.state_dict(), model_path)

        epoch_number += 1

    # Plot combined loss curve
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(range(len(training_epoch_loss)), training_epoch_loss, label='train_loss')
    plt.plot(range(len(validation_epoch_loss)), validation_epoch_loss, label='val_loss')
    plt.grid()
    plt.legend()
    plt.title('Loss Curve')

    skip_head = 10
    plt.subplot(1, 2, 2)
    plt.plot(range(skip_head, len(training_epoch_loss)), training_epoch_loss[skip_head:], label='train_loss')
    plt.plot(range(skip_head, len(validation_epoch_loss)), validation_epoch_loss[skip_head:], label='val_loss')
    plt.grid()
    plt.legend()
    plt.title('Loss Curve (Skip First 10 Epochs)')

    plt.tight_layout()
    plt.savefig('combined_loss_curve.png', dpi=300)

    # # Plot loss curve
    # plt.figure()
    # plt.plot(range(len(training_epoch_loss)), training_epoch_loss, label='train_loss')
    # plt.plot(range(len(validation_epoch_loss)), validation_epoch_loss,label='val_loss')
    # plt.grid()
    # plt.legend()
    # plt.savefig('loss_curve.png', dpi=300)

    # skip_head = 10
    # plt.figure() # Skip the first 10 epochs
    # plt.plot(range(skip_head, len(training_epoch_loss)), training_epoch_loss[skip_head:], label='train_loss')
    # plt.plot(range(skip_head, len(validation_epoch_loss)), validation_epoch_loss[skip_head:],label='val_loss')
    # plt.grid()
    # plt.legend()
    # plt.savefig('loss_curve_skip_10.png', dpi=300)


def parse_args(epilog=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        epilog=epilog
        or f"""
Examples:

Show model Architecture:
    $ python {sys.argv[0]} -a
    $ python {sys.argv[0]} --show_architecture

Pytorch GPU setup check
    $ python {sys.argv[0]} -t
    $ python {sys.argv[0]} --test_torch_gpu

Run on single machine:
    $ python {sys.argv[0]} 
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('-a', '--show_architecture', action='store_true', help='show model architecture')
    parser.add_argument('-t', '--test_torch_gpu', action='store_true', help='check pytorch gpu setup')
    parser.add_argument('-e', '--epochs', type=int, default=200, help="number of epochs")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = parse_args()
    if args.test_torch_gpu:
        torch_check_GPU()
        sys.exit(0)
    if args.show_architecture:
        test_model_architecture()
        sys.exit(0)
    main(args)
