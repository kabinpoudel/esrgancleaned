import time
import torch.backends.cudnn as cudnn
from torch import nn
from models import *
from datasets import SRDataset
from utils import *

# Data parameters
data_folder = './data/output'
crop_size = 96
scaling_factor = 4

# Learning parameters
checkpoint = None  # path to model (SRGAN) checkpoint, None if none
batch_size = 16
start_epoch = 0
iterations = 2e5
workers = 4
vgg19_i = 5
vgg19_j = 4
beta = 1e-3
print_freq = 100
lr = 1e-5
grad_clip = None
LAMBDA_GP = 10
# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint, srresnet_checkpoint
    

    # Initialize model or load checkpoint
    if checkpoint is None:
        # Generator
        generator = ESRGANGenerator(in_channels=3).to(device)
        discriminator =  ESRGDiscriminator(in_channels=3).to(device)
        optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
        optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.9))
        l1 = nn.L1Loss()
        generator.train()
        discriminator.train()
        vgg_loss = VGGLoss()
        g_scaler = torch.cuda.amp.GradScaler()
        d_scaler = torch.cuda.amp.GradScaler()

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        generator = checkpoint['generator']
        discriminator = checkpoint['discriminator']
        optimizer_g = checkpoint['optimizer_g']
        optimizer_d = checkpoint['optimizer_d']
        print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              split='train',
                              if_srcnn=False,
                              crop_size=crop_size,
                              scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',
                              hr_img_type='imagenet-norm')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)
    print(epochs)
    # Epochs
    for epoch in range(start_epoch, epochs):

        # At the halfway point, reduce learning rate to a tenth
        if epoch == int((iterations / 2) // len(train_loader) + 1):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        # One epoch's training
        train(train_loader=train_loader,
              generator=generator,
              discriminator=discriminator,
              optimizer_g=optimizer_g,
              optimizer_d=optimizer_d,
              epoch=epoch,
              d_scaler=d_scaler,
              g_scaler=g_scaler,
              vgg_loss=vgg_loss,
              l1=l1)
    
        # Save checkpoint
        torch.save({'epoch': epoch,
                    'generator': generator,
                    'discriminator': discriminator,
                    'optimizer_g': optimizer_g,
                    'optimizer_d': optimizer_d},
                   './pretrained/checkpoint_esrgan.pth_%d.tar'%epoch)


def train(train_loader, generator, discriminator, optimizer_g, optimizer_d, epoch, d_scaler, g_scaler, vgg_loss,l1):
    
    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), imagenet-normed

        # GENERATOR UPDATE

        # Generate
        with torch.cuda.amp.autocast():
            sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
            critic_real = discriminator(hr_imgs)
            critic_fake = discriminator(sr_imgs.detach())
            gp = gradient_penalty(discriminator, hr_imgs, sr_imgs, device=device)
            loss_critic = (
                -(torch.mean(critic_real) - torch.mean(critic_fake))
                + LAMBDA_GP * gp
            )
        optimizer_d.zero_grad()
        d_scaler.scale(loss_critic).backward()
        d_scaler.step(optimizer_d)
        d_scaler.update()

        with torch.cuda.amp.autocast():
            l1_loss = 1e-2 * l1(sr_imgs, hr_imgs)
            adversarial_loss = 5e-3 * -torch.mean(discriminator(sr_imgs))
            loss_for_vgg = vgg_loss(sr_imgs, hr_imgs)
            gen_loss = l1_loss + loss_for_vgg + adversarial_loss

        optimizer_g.zero_grad()
        g_scaler.scale(gen_loss).backward()
        g_scaler.step(optimizer_g)
        g_scaler.update()


        # Keep track of batch times
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # message = 'Epoch: [{0}][{1}/{2}]----' \
        #       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----' \
        #       'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----' \
        #       'L1_loss {loss_c.val:.4f} ({loss_c.avg:.4f})----' \
        #       'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})----' \
        #       'gen_loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
        #                                                               i,
        #                                                               len(train_loader),
        #                                                               batch_time=batch_time,
        #                                                               data_time=data_time,
        #                                                               loss_c=l1_loss,
        #                                                               loss_a=adversarial_loss,
        #                                                               loss_d=gen_loss)
    
        # # Print status
        # if i % print_freq == 0:
        #     custom_print(message,"./logs/esrganbs16.txt")

if __name__ == '__main__':
    main()
