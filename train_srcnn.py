import time
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from srcnn_model import SRCNN
from datasets import SRDataset
from utils import *


# Data parameters
data_folder = './data/output'  
crop_size = 96
scaling_factor = 2

# Learning parameters
checkpoint = "saved_models\srcnn\checkpoint_srcnn_25.pth.tar"
batch_size = 16
start_epoch = 0
iterations = 1e6
workers = 0
print_freq = 100
lr = 1e-4
grad_clip = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SRCNN()
        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              split='train',
                              if_srcnn=True,
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='y-channel',
                              hr_img_type='y-channel')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)
    print(f"Total Epochs:{epochs}\n")
    # Epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        torch.save({'epoch': epoch,
                    'model': model,
                    'optimizer': optimizer},
                   './saved_models/srcnn/checkpoint_srcnn_%d.pth.tar'%epoch)
        
def train(train_loader, model, criterion, optimizer, epoch):

    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)
        
        # idk why and how
        optimizer.zero_grad()

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        # Backward prop.
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        message = 'Epoch: [{0}][{1}/{2}]----' \
              'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----' \
              'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----' \
              'Loss {loss.val:.5f} ({loss.avg:.5f})'.format(epoch, i, len(train_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time, loss=losses)
        # Print status
        if i % print_freq == 0:
            custom_print(message,output_file="./logs/srcnnbs16.txt")
    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()