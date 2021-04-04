import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import yaml

import models.geno_searched as geno_types
from torchvision.utils import save_image
from util.datasets.pascal_voc import VOCSegmentation
from models import get_segmentation_model

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from util.utils import weights_init
from models.discriminator import *

import torch.nn as nn
import torch.nn.functional as F
import torch


def run():

    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
    parser.add_argument("--dataset_name", type=str, default="pascal_voc", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_height", type=int, default=256, help="size of image height")
    parser.add_argument("--img_width", type=int, default=256, help="size of image width")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument('--config', nargs='?', type=str, default='../configs/nas_unet/nas_unet_voc.yml',
                        help='Configuration file to use')
    parser.add_argument(
        "--sample_interval", type=int, default=500, help="interval between sampling of images from generators"
    )
    parser.add_argument('--model', nargs='?', type=str, default='nasunet',
                                help='Model to train and evaluation')
    parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
    parser.add_argument('--h_image_size', type=int, default=256)
    parser.add_argument('--w_image_size', type=int, default=256)
    opt = parser.parse_args()

    os.makedirs("images/%s" % opt.dataset_name, exist_ok=True)
    os.makedirs("saved_models/%s" % opt.dataset_name, exist_ok=True)

    with open(opt.config) as fp:
        cfg = yaml.load(fp)
        print('load configure file at {}'.format(opt.config))

    model_name = opt.model

    cuda = True if torch.cuda.is_available() else False

    # Loss functions
    criterion_GAN = torch.nn.MSELoss()
    criterion_pixelwise = torch.nn.L1Loss()

    # Loss weight of L1 pixel-wise loss between translated image and real image
    lambda_pixel = 100

    # Calculate output of image discriminator (PatchGAN)
    patch = (1, opt.img_height // 2 ** 4, opt.img_width // 2 ** 4)

    # Initialize generator and discriminator
    genotype = None
    init_channels = None
    depth = None
    # Setup Model
    try:
        genotype = eval('geno_types.%s' % cfg['training']['geno_type'])
        init_channels = cfg['training']['init_channels']
        depth = cfg['training']['depth']

    except:
        print('error')
        genotype = None
        init_channels = 0
        depth = 0
    # aux_weight > 0 and the loss is cross_entropy, we will use FCN header for auxiliary layer. and the aux set to True
    # aux_weight > 0 and the loss is cross_entropy_with_dice, we will combine cross entropy loss with dice loss
    aux = False

    generator = get_segmentation_model(model_name,
                                   dataset=cfg['data']['dataset'],
                                   backbone=cfg['training']['backbone'],
                                   aux=aux,
                                   c=init_channels,
                                   depth=depth,
                                   # the below two are special for nasunet
                                   genotype=genotype,
                                   double_down_channel=cfg['training']['double_down_channel']
                                   )

    discriminator = Discriminator()

    if cuda:
        generator = generator.cuda()
        discriminator = discriminator.cuda()
        criterion_GAN.cuda()
        criterion_pixelwise.cuda()

    if opt.epoch != 0:
        # Load pretrained models
        generator.load_state_dict(torch.load("saved_models/%s/generator_%d.pth" % (opt.dataset_name, opt.epoch)))
        discriminator.load_state_dict(torch.load("saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, opt.epoch)))
    else:
        # Initialize weights
        # init weight using hekming methods
        generator.apply(weights_init)
        discriminator.apply(weights_init_normal)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # Configure dataloaders
    trainset = VOCSegmentation(root='../', split='train', mode='train')
    valset = VOCSegmentation(root='../', split='val', mode='val')
    # testset = get_dataset(self.cfg['data']['dataset'], split='test', mode='test')
    nweight = trainset.class_weight
    n_classes = trainset.num_class
    batch_size = opt.batch_size
    kwargs = {'num_workers': 6, 'pin_memory': True}

    dataloader = DataLoader(trainset, batch_size=batch_size,
                                                   drop_last=True, shuffle=True, **kwargs)
    val_dataloader = DataLoader(valset, batch_size=10,
                                                   drop_last=False, shuffle=False, **kwargs)

    # root = '../'
    #
    # transform = transforms.Compose([
    #     transforms.Pad(10),
    #     transforms.CenterCrop((opt.h_image_size, opt.w_image_size)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    # ])
    #
    # train_data_set = VOC(root=root,
    #                      image_size=(opt.h_image_size, opt.w_image_size),
    #                      dataset_type='train',
    #                      transform=transform)
    # train_data_loader = DataLoader(train_data_set,
    #                                batch_size=opt.batch_size,
    #                                shuffle=True, num_workers=16)
    #
    # val_data_set = VOC(root=root,
    #                    image_size=(opt.h_image_size, opt.w_image_size),
    #                    dataset_type='val',
    #                    transform=transform)
    #
    # val_data_loader = DataLoader(val_data_set, batch_size=opt.batch_size, shuffle=False, num_workers=16)  # For
    # make samples out of various models, shuffle=False

    print('dataset is loaded')
    # Tensor type
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    print("CUDA Tensor")

    def sample_images(batches_done):
        """Saves a generated sample from the validation set"""
        """Saves a generated sample from the validation set"""
        (input, target) = next(iter(val_dataloader))
        real_A = Variable(input.type(Tensor))
        real_B = Variable(target.type(Tensor))
        fake_B = generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=False)

        # (input, target) = next(iter(val_data_loader))
        # real_A = Variable(input.type(Tensor))
        # real_B = target
        # real_B = torch.from_numpy(to_rgb(real_B))
        # fake_B = generator(real_A)
        # pred_B = torch.max(fake_B, dim=1)[1].cpu()
        # pred_B = torch.from_numpy(to_rgb(pred_B))
        # img_sample = torch.cat((pred_B.data, real_B.data), -2)
        # save_image(img_sample, "images/%s/%s.png" % (opt.dataset_name, batches_done), nrow=5, normalize=True)

    # ----------
    #  Training
    # ---------

    print("Start Training")
    prev_time = time.time()
    for epoch in range(opt.epoch, opt.n_epochs):

        for i, (input, target) in enumerate(dataloader):

            # Model inputs
            real_A = input.cuda()
            real_B = target.cuda()
            # real_B = torch.from_numpy(to_rgb(target)).type(Tensor).cuda()

            print('Adversarial ground truths')
            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------

            optimizer_G.zero_grad()

            # GAN loss
            fake_B = generator(real_A) # batch_size, 21,256,256
            # pred_B = torch.unsqueeze(torch.max(fake_B, dim=1)[1], 1)  # batch_size,1,256,256
            # pred_B = torch.max(fake_B, dim=1)[1]
            # print(fake_B.shape)
            # pred_B = torch.from_numpy(to_rgb(pred_B.cpu())).type(Tensor)
            # print(pred_B.shape)
            # print(real_B.shape)

            pred_fake = discriminator(fake_B, real_A)
            # print(pred_fake.shape)
            loss_GAN = criterion_GAN(pred_fake, valid)
            # Pixel-wise loss
            loss_pixel = criterion_pixelwise(fake_B, real_B)

            # Total loss
            loss_G = loss_GAN + lambda_pixel * loss_pixel

            loss_G.backward()

            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Real loss
            # real_B_N = real_B + (torch.randn(real_B.size()) * 1 + np.random.random).type(Tensor).cuda()
            # real_A_N = real_A + (torch.randn(real_A.size()) * 1 + torch.rand(1,)[0]).type(Tensor).cuda()
            pred_real = discriminator(real_B, real_A)
            loss_real = criterion_GAN(pred_real, valid)



            # Fake loss
            pred_fake = discriminator(fake_B.detach(), real_A)
            loss_fake = criterion_GAN(pred_fake, fake)

            # Total loss
            loss_D = 0.5 * (loss_real + loss_fake)

            # if i % 5 == 0:
            loss_D.backward()
            optimizer_D.step()

            # --------------
            #  Log Progress
            # --------------

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = opt.n_epochs * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, pixel: %f, adv: %f] ETA: %s"
                % (
                    epoch,
                    opt.n_epochs,
                    i,
                    len(dataloader),
                    loss_D.item(),
                    loss_G.item(),
                    loss_pixel.item(),
                    loss_GAN.item(),
                    time_left,
                )
            )

            # If at sample interval save image
            if batches_done % opt.sample_interval == 0:
                sample_images(batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), "saved_models/%s/generator_%d.pth" % (opt.dataset_name, epoch))
            torch.save(discriminator.state_dict(), "saved_models/%s/discriminator_%d.pth" % (opt.dataset_name, epoch))


if __name__ == '__main__':
    run()