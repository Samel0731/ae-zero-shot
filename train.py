import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from model import build_AE
from utils.loader import KFoldDataset
from matplotlib import pyplot as plt
# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime





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

    EPOCHS = 5

    best_vloss = 1_000_000.
    lr = 0.001
    # --------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # Preparing data
    batch_size = 32
    transform = transforms.Compose([
        transforms.Resize((200,200)),   # Resize images to 200x200
        transforms.ToTensor(),  # Convert images to tensor (values between 0 and 1)
    ])
    training_set = KFoldDataset(root_dir=dataset_root, field_names=field_names, k=0, train=True, transform=transform)
    validation_set = KFoldDataset(root_dir=dataset_root, field_names=field_names, k=0, train=False, transform=transform)
    training_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    iters_per_epoch = len(training_set) // batch_size
    viters_per_epoch = len(validation_set) // batch_size
    # --------------------------------------------------------------------------

    training_epoch_loss = []
    validation_epoch_loss = []
    autoencoder = build_AE()
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
        pbar = tqdm(enumerate(training_loader),desc=f"Epoch: [{epoch+1}/{EPOCHS}] Iter: [{0}/{iters_per_epoch}]", total=iters_per_epoch)
        for iter, (inputs, labels) in pbar:
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = autoencoder(inputs)
            print('check shape',outputs.shape, labels.shape)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
            train_loss += loss.item()
            if iter % 1000 == 999:
                last_loss = running_loss / 1000
                tb_x = epoch * len(training_loader) + iter + 1
                writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.
            pbar.set_description(f"Epoch(Training): [{epoch+1}/{EPOCHS}] Iter: [{iter+1}/{iters_per_epoch}] Loss: {train_loss / (iter+1):.8f}")
        train_loss /= len(training_set)
        training_epoch_loss.append(train_loss)
        avg_loss = last_loss
        # --------------------------------------------------------------------------

        running_vloss = 0.
        autoencoder.eval()
        vbar = tqdm(enumerate(validation_loader),total=len(validation_set),desc=f"Epoch(Validating):[{epoch+1}/{EPOCHS}]")
        viter_number = 0
        with torch.no_grad():
            for viter, vdata in vbar:
                vinputs, vlabels = vdata
                vinputs = vinputs.to(device)
                vlabels = vlabels.to(device)
                voutputs = autoencoder(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss
                vbar.set_description(f"Epoch(Validating): [{epoch+1}/{EPOCHS}] Iter: [{viter+1}/{viters_per_epoch}] Loss: {running_vloss / (viter+1):.8f}")
                viter_number = viter

        avg_vloss = running_vloss / (viter_number + 1)
        validation_epoch_loss.append(avg_vloss)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

        writer.add_scalars('Training vs. Validation Loss',
                    { 'Training' : avg_loss, 'Validation' : avg_vloss },
                    epoch_number + 1)
        writer.flush()

        # # Track best performance, and save the model's state
        # if avg_vloss < best_vloss:
        #     best_vloss = avg_vloss
        #     model_path = 'model_{}_{}'.format(timestamp, epoch_number)
        #     torch.save(autoencoder.state_dict(), model_path)

        epoch_number += 1

    # Plot loss curve
    plt.plot(training_epoch_loss, label='train_loss')
    plt.plot(validation_epoch_loss,label='val_loss')
    plt.legend()
    plt.show


from model import test
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
if __name__=='__main__':
    # main()
    test()
