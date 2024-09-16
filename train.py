import argparse
import os
import torch
from math import ceil
from matplotlib import pyplot as plt
from model import build_AE, init_weights, test_model_architecture
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from tqdm import tqdm
from utils.loader import KFoldDataset
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


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
    dataset_root = './data'
    field_names = ['p_seabass', 'sea', 'barrel', 'lng', 'noon']


    # --------------------------------------------------------------------------
    # train setting
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/fashion_trainer_{}'.format(timestamp))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epoch_number = 0

    EPOCHS = 200

    best_vloss = 1_000_000.
    lr = 0.0001
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparing data
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((200,200)),   # Resize images to 200x200
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandAugment(),
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
    autoencoder = build_AE()
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
        running_loss = 0.
        last_loss = 0.
        pbar = tqdm(enumerate(training_loader), desc=f"Epoch: [{epoch+1}/{EPOCHS}] Iter: [{0}/{iters_per_epoch}]", total=iters_per_epoch)
        for iter, (inputs, labels) in pbar:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = autoencoder(inputs)
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
        viter_number = 0
        with torch.no_grad():
            for viter, vdata in vbar:
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = autoencoder(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss.item()
                vbar.set_description(f"Epoch(Validating): [{epoch+1}/{EPOCHS}] Iter: [{viter+1}/{viters_per_epoch}] Loss: {running_vloss / (viter+1):.8f}")
                viter_number = viter

        avg_vloss = running_vloss / len(validation_loader)
        validation_epoch_loss.append(avg_vloss)
        print('LOSS train {:.8f} valid {:.8f}'.format(avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss and epoch>100:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(autoencoder.state_dict(), model_path)

        epoch_number += 1

    # Plot loss curve
    skip_head = 10
    plt.plot(training_epoch_loss[skip_head:], label='train_loss')
    plt.plot(validation_epoch_loss[skip_head:],label='val_loss')
    plt.grid()
    plt.legend()
    plt.savefig('loss_curve.png', dpi=300)


if __name__=='__main__':
    # main()
    test_model_architecture()
