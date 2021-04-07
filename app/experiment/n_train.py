import os
import sys
import yaml
import time
import shutil
import datetime
from tqdm import tqdm
import torch.nn as nn
from torch.utils import data
import torch.backends.cudnn as cudnn

# sys.path.append('..')
from options.train_options import TrainOptions
from data import create_dataset
from util.utils import get_logger, save_checkpoint, calc_time
from util.utils import average_meter, weights_init
from util.utils import get_gpus_memory_info, calc_parameters_count
from util.schedulers import get_scheduler

from util.metrics import *
from util.utils import *
from models.load import get_segmentation_model
from models.discriminator import *
import models.geno_searched as geno_types

cb = 0

from tensorboardX import SummaryWriter

# for ignoring warnings
import warnings


warnings.filterwarnings("ignore")

epoch, train_discriminator_loss_meter, train_generator_loss_meter, train_pixel_loss, train_adversarial_loss_meter, pixAcc = 0,0,0,0,0,0
check = 0

class Network(object):
    
    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        self._check_resume()


    def _init_configure(self):

        self.args = TrainOptions().parse()

        with open(self.args.config) as fp:
            self.cfg = yaml.load(fp)
            print('load configure file at {}'.format(self.args.config))
        self.model_name = self.args.model
        # print('Usage model :{}'.format(self.model_name))

    def _init_logger(self):
        log_dir = '../logs/' + self.model_name + '/train' + '/{}'.format(self.cfg['data']['dataset']) \
                  + '/{}'.format(time.strftime('%Y%m%d-%H%M%S'))
        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        self.logger.info('{}-Train'.format(self.model_name))
        self.save_path = log_dir
        self.save_tbx_log = self.save_path + '/tbx_log'
        self.save_image_path = os.path.join(self.save_path, 'saved_val_images')
        self.writer = SummaryWriter(self.save_tbx_log)
        shutil.copy(self.args.config, self.save_path)

    def _init_device(self):
        if not torch.cuda.is_available():
            self.logger.info('no gpu device available')
            sys.exit(1)

        np.random.seed(self.cfg.get('seed', 1337))
        torch.manual_seed(self.cfg.get('seed', 1337))
        torch.cuda.manual_seed(self.cfg.get('seed', 1337))
        cudnn.enabled = True
        cudnn.benchmark = True
        self.device_id, self.gpus_info = get_gpus_memory_info()
        self.device = torch.device('cuda:{}'.format(0 if self.cfg['training']['multi_gpus'] else self.device_id))

    def _init_dataset(self):
        self.Tensor = torch.cuda.FloatTensor
        self.train_queue, self.valid_queue = create_dataset(
            self.args)  # create a dataset given opt.dataset_mode and other options

    def _init_model(self):

        # Loss functions
        criterion_GAN = GANLoss(self.args.gan_mode)
        criterion_pixelwise = torch.nn.L1Loss()

        self.criterion_GAN = criterion_GAN.to(self.device)
        self.criterion_pixelwise = criterion_pixelwise.to(self.device)

        self.logger.info("Using criterion_GAN loss {}".format(self.criterion_GAN))
        self.logger.info("Using criterion_pixelwise loss {}".format(self.criterion_pixelwise))

        # Setup Model
        try:
            genotype = eval('geno_types.%s' % self.cfg['training']['geno_type'])
            init_channels = self.cfg['training']['init_channels']
            depth = self.cfg['training']['depth']

        except:
            genotype = None
            init_channels = 0
            depth = 0

        # Loss weight of L1 pixel-wise loss between translated image and real image
        self.lambda_pixel = 100
        # Calculate output of image discriminator (PatchGAN)
        self.patch = (1, self.args.crop_size // 2 ** 4, self.args.crop_size // 2 ** 4)
        self.aux = False
        generator = get_segmentation_model(self.model_name,
                                           dataset=self.cfg['data']['dataset'],
                                           backbone=self.cfg['training']['backbone'],
                                           aux=self.aux,
                                           c=init_channels,
                                           depth=depth,
                                           # the below two are special for nasunet
                                           genotype=genotype,
                                           double_down_channel=self.cfg['training']['double_down_channel']
                                           )
        discriminator = define_D(self.args.input_nc + self.args.output_nc, self.args.ndf, self.args.netD,
                                 self.args.n_layers_D, self.args.norm, self.args.init_type, self.args.init_gain)
        # init weight using hekming methods
        generator.apply(weights_init)
        self.logger.info('Initialize the Generator model weights: kaiming_uniform')
        # already Initialize in model defination
        self.logger.info('Initialize the Discriminator model weights: Normal Distribution')

        if torch.cuda.device_count() > 1 and self.cfg['training']['multi_gpus']:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            generator = nn.DataParallel(generator)
            discriminator = nn.DataParallel(discriminator)

        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)

        self.generator = generator.to(self.device)
        self.logger.info('param size of generator = %fMB', calc_parameters_count(generator))

        self.discriminator = discriminator.to(self.device)
        self.logger.info('param size of discriminator = %fMB', calc_parameters_count(discriminator))

        # Optimizers
        self.optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.args.lr, betas=(self.args.beta1, 0.999))

        self.logger.info("Using model optimizer {}".format(self.optimizer_G))

    def _check_resume(self):
        self.dur_time = 0
        self.start_epoch = 0
        self.best_generator_loss, self.best_pix_loss, self.best_adv_loss, self.best_discriminator_loss = np.inf, np.inf, np.inf, np.inf
        self.best_pixAcc, self.best_mIoU = 0, 0
        # optionally resume from a checkpoint for model
        resume = self.cfg['training']['resume'] if self.cfg['training']['resume'] is not None else None
        if resume is not None:
            if os.path.isfile(resume):
                self.logger.info("Loading model and optimizer from checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume, map_location=self.device)
                if not self.args.ft:  # no fine-tuning
                    self.start_epoch = checkpoint['epoch']
                    self.dur_time = checkpoint['dur_time']
                    print(self.dur_time)
                    self.best_mIoU = checkpoint['best_mIoU']
                    self.best_pixAcc = checkpoint['best_pixAcc']
                    self.best_generator_loss = checkpoint['best_generator_loss']
                    self.best_discriminator_loss = checkpoint['best_discriminator_loss']
                    self.best_pix_loss = checkpoint['best_pix_loss']
                    self.best_adv_loss = checkpoint['best_adv_loss']
                    self.optimizer_G.load_state_dict(checkpoint['generator_model_optimizer'])
                    self.optimizer_D.load_state_dict(checkpoint['discriminator_model_optimizer'])

                self.generator.load_state_dict(checkpoint['generator_model_state'])
                self.discriminator.load_state_dict(checkpoint['discriminator_model_state'])
            else:
                self.logger.info("No checkpoint found at '{}'".format(resume))

        # init LR_scheduler
        scheduler_params = {k: v for k, v in self.cfg['training']['lr_schedule'].items()}
        if 'max_iter' in self.cfg['training']['lr_schedule']:
            scheduler_params['max_iter'] = self.cfg['training']['epoch']
            # Note: For step in train epoch !!!!  must use the value below
            # scheduler_params['max_iter'] = len(self.train_queue) * self.cfg['training']['epoch'] \
            #                                // self.cfg['training']['batch_size']
        if 'T_max' in self.cfg['training']['lr_schedule']:
            scheduler_params['T_max'] = self.cfg['training']['epoch']
            # Note: For step in train epoch !!!!  must use the value below
            # scheduler_params['T_max'] = len(self.train_queue) * self.cfg['training']['epoch'] \
            #                                // self.cfg['training']['batch_size']

        scheduler_params['last_epoch'] = -1 if self.start_epoch == 0 else self.start_epoch
        self.scheduler = get_scheduler(self.optimizer_G, scheduler_params)

    def run(self):
        self.logger.info('args = %s', self.cfg)
        # Setup Metrics
        self.metric_train = SegmentationMetric(19)  # classes
        self.metric_val = SegmentationMetric(19)

        self.train_generator_loss_meter = average_meter()
        self.train_discriminator_loss_meter = average_meter()

        self.val_generato_loss_meter = average_meter()
        self.val_discriminator_loss_meter = average_meter()

        self.train_adversarial_loss_meter = average_meter()
        self.val_adversarial_loss_meter = average_meter()

        self.train_pixel_loss = average_meter()
        self.val_pixel_loss = average_meter()

        self.patience = 0
        self.save_best = True
        # time
        run_start = time.time()
        self.prev_time = run_start

        # Set up results folder
        if not os.path.exists(self.save_image_path):
            os.makedirs(self.save_image_path)

        for epoch in range(self.start_epoch, self.cfg['training']['epoch']):

            self.epoch = epoch

            self.scheduler.step()

            self.logger.info('=> Epoch {}, lr {}'.format(self.epoch, self.scheduler.get_lr()[-1]))

            # train and search the model
            self.train()

            # valid the model
            self.val()

            self.logger.info('current best generator loss {}, best pix loss{}, pixAcc {}, mIoU {}'.format(
                self.best_generator_loss, self.best_pix_loss, self.best_pixAcc, self.best_mIoU,
            ))

            if self.save_best:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'dur_time': self.dur_time + time.time() - run_start,
                    'generator_model_state': self.generator.state_dict(),
                    'generator_model_optimizer': self.optimizer_G.state_dict(),
                    'discriminator_model_state': self.discriminator.state_dict(),
                    'discriminator_model_optimizer': self.optimizer_D.state_dict(),
                    'best_generator_loss': self.best_generator_loss,
                    'best_discriminator_loss': self.best_discriminator_loss,
                    'best_adv_loss': self.best_adv_loss,
                    'best_pix_loss': self.best_pix_loss,
                    'best_pixAcc': self.best_pixAcc,
                    'best_mIoU': self.best_mIoU,

                }, True, self.save_path)

                self.logger.info('save checkpoint (epoch %d) in %s  dur_time: %s',
                                 epoch, self.save_path, calc_time(self.dur_time + time.time() - run_start))

                self.save_best = False

            if self.patience == self.cfg['training']['max_patience'] or epoch == self.cfg['training']['epoch'] - 1:
                print('Early stopping')
                break
            else:
                self.logger.info('current patience :{}'.format(self.patience))

            # Reset Matrix
            self.val_generato_loss_meter.reset()
            self.val_discriminator_loss_meter.reset()

            self.train_generator_loss_meter.reset()
            self.train_discriminator_loss_meter.reset()

            self.train_adversarial_loss_meter.reset()
            self.val_adversarial_loss_meter.reset()

            self.train_pixel_loss.reset()
            self.val_pixel_loss.reset()

            self.metric_train.reset()
            self.metric_val.reset()
            self.logger.info('cost time: {}'.format(calc_time(self.dur_time + time.time() - run_start)))

        # export scalar data to JSON for external processing
        self.writer.export_scalars_to_json(self.save_tbx_log + "/all_scalars.json")
        self.writer.close()
        self.logger.info('cost time: {}'.format(calc_time(self.dur_time + time.time() - run_start)))
        self.logger.info('log dir in : {}'.format(self.save_path))

    def sample_images(self, real_A, real_B, fake_B, epoch, step, phase):

        """Saves a generated sample from the validation set"""

        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        # random_image = np.random.randint(15) # batch size of val data
        image_numpy = tensor2im(img_sample)
        img_path = self.save_image_path + "/%s_phase_%s_step %s_epoch.png" % (phase, step, epoch)
        save_image(image_numpy, img_path)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.discriminator(fake_AB.detach())
        self.loss_D_fake = self.criterion_GAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.discriminator(real_AB)
        self.loss_D_real = self.criterion_GAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.train_discriminator_loss_meter.update(self.loss_D.item())
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.discriminator(fake_AB)
        self.loss_G_GAN = self.criterion_GAN(pred_fake, True)
        self.train_adversarial_loss_meter.update(self.loss_G_GAN.item())
        # Second, G(A) = B
        self.loss_G_L1 = self.criterion_pixelwise(self.fake_B, self.real_B) * self.args.lambda_L1
        self.train_pixel_loss.update(self.loss_G_L1.item())
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.train_generator_loss_meter.update(self.loss_G.item())
        self.metric_train.update(self.real_B, self.fake_B)
        self.loss_G.backward()

    def train(self):
        global train_generator_loss_meter, train_adversarial_loss_meter, train_pixel_loss, epoch,  pixAcc, mIoU

        self.generator.train()
        self.discriminator.train()
        tbar = tqdm(self.train_queue)
        for step, data in enumerate(tbar):

            AtoB = self.args.direction == 'AtoB'
            self.real_A = data['A' if AtoB else 'B'].to(self.device)
            self.real_B = data['B' if AtoB else 'A'].to(self.device)

            self.fake_B = self.generator(self.real_A)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.set_requires_grad(self.discriminator, True)  # enable backprop for D
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

            # update G
            self.set_requires_grad(self.discriminator, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate graidents for G
            self.optimizer_G.step()  # udpate G's weights

            # Determine approximate time left
            batches_done = self.epoch * len(self.train_queue) + step
            batches_left = self.cfg['training']['epoch'] * len(self.train_queue) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - self.prev_time))
            self.prev_time = time.time()

            if step % self.cfg['training']['report_freq'] == 0:
                pixAcc, mIoU = self.metric_train.get()

                # save image
                self.sample_images(self.real_A, self.real_B, self.fake_B, self.epoch, step, 'train')

                self.logger.info(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %.3f] [G loss: %f, pixel: %f, adv: %f] [pixAcc: %.3f, "
                    "mIoU %.6f] ETA: %s "
                    % (
                        self.epoch,
                        self.cfg['training']['epoch'],
                        step,
                        len(self.train_queue),
                        self.train_discriminator_loss_meter.mloss,
                        self.train_generator_loss_meter.mloss,
                        self.train_pixel_loss.mloss,
                        self.train_adversarial_loss_meter.mloss,
                        pixAcc,
                        mIoU,
                        time_left,
                    )
                )

                tbar.set_description(
                    "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %.3f] [G loss: %f, pixel: %f, adv: %f] [pixAcc: %.3f, "
                    "mIoU %.6f] ETA: %s "
                    % (
                        self.epoch,
                        self.cfg['training']['epoch'],
                        step,
                        len(self.train_queue),
                        self.train_discriminator_loss_meter.mloss,
                        self.train_generator_loss_meter.mloss,
                        self.train_pixel_loss.mloss,
                        self.train_adversarial_loss_meter.mloss,
                        pixAcc,
                        mIoU,
                        time_left
                    )
                )
                
                # views.epoch_ = 300
                # reciver.call(  self.epoch, self.train_discriminator_loss_meter.mloss,
                # self.train_generator_loss_meter.mloss,
                # self.train_pixel_loss.mloss,
                # self.train_adversarial_loss_meter.mloss,
                # pixAcc,
                # mIoU)

        # save in tensorboard scalars
        self.writer.add_scalar('Train_Generator/loss', self.train_generator_loss_meter.mloss, self.epoch)
        self.writer.add_scalar('Train_adversarial/loss', self.train_adversarial_loss_meter.mloss, self.epoch)
        self.writer.add_scalar('Train_Pixel/loss', self.train_pixel_loss.mloss, self.epoch)

        return self.train_generator_loss_meter.mloss, self.train_adversarial_loss_meter.mloss, 
        self.train_pixel_loss.mloss, self.epoch,  pixAcc, mIoU,

    def val(self):
        self.generator.eval()
        self.discriminator.eval()

        tbar = tqdm(self.valid_queue)

        with torch.no_grad():
            for step, data in enumerate(tbar):

                AtoB = self.args.direction == 'AtoB'
                real_A = data['A' if AtoB else 'B'].to(self.device)
                real_B = data['B' if AtoB else 'A'].to(self.device)

                fake_B = self.generator(real_A)

                fake_AB = torch.cat((real_A, fake_B),
                                    1)  # we use conditional GANs; we need to feed both input and output to the discriminator
                pred_fake = self.discriminator(fake_AB.detach())
                loss_D_fake = self.criterion_GAN(pred_fake, False)
                # Real
                real_AB = torch.cat((real_A, real_B), 1)
                pred_real = self.discriminator(real_AB)
                loss_D_real = self.criterion_GAN(pred_real, True)
                # combine loss and calculate gradients
                loss_D = (loss_D_fake + loss_D_real) * 0.5
                self.val_discriminator_loss_meter.update(loss_D.item())

                # generator
                fake_AB = torch.cat((real_A, fake_B), 1)
                pred_fake = self.discriminator(fake_AB)
                loss_G_GAN = self.criterion_GAN(pred_fake, True)
                self.val_adversarial_loss_meter.update(loss_G_GAN.item())
                # Second, G(A) = B
                loss_G_L1 = self.criterion_pixelwise(fake_B, real_B) * self.args.lambda_L1
                self.val_pixel_loss.update(loss_G_L1.item())
                # combine loss and calculate gradients
                loss_G = loss_G_GAN + loss_G_L1
                self.val_generato_loss_meter.update(loss_G.item())
                self.metric_val.update(real_B, fake_B)

                if step % self.cfg['training']['report_freq'] == 0:
                    # save image
                    self.sample_images(real_A, real_B, fake_B, self.epoch, step, 'val')
                    pixAcc, mIoU = self.metric_val.get()
                    self.logger.info(
                        "\r[Batch %d/%d] [Val D loss: %f] [Val G loss: %f, Val pixel: %f, Val adv: %f] "
                        "[Val pixAcc: %.3f, Val mIoU %.6f] "
                        % (
                            step,
                            len(self.valid_queue),
                            self.val_discriminator_loss_meter.mloss,
                            self.val_generato_loss_meter.mloss,
                            self.val_pixel_loss.mloss,
                            self.val_adversarial_loss_meter.mloss,
                            pixAcc,
                            mIoU
                        )
                    )

                    tbar.set_description(
                        "\r[Batch %d/%d] [Val D loss: %f] [Val G loss: %f, Val pixel: %f, Val adv: %f] "
                        "[Val pixAcc: %.3f, Val mIoU %.6f] "
                        % (
                            step,
                            len(self.valid_queue),
                            self.val_discriminator_loss_meter.mloss,
                            self.val_generato_loss_meter.mloss,
                            self.val_pixel_loss.mloss,
                            self.val_adversarial_loss_meter.mloss,
                            pixAcc,
                            mIoU
                        )
                    )



        # save in tensorboard scalars
        pixAcc, mIoU = self.metric_val.get()
        cur_gen_loss = self.val_generato_loss_meter.mloss
        cur_pix_loss = self.val_pixel_loss.mloss
        cur_dis_loss = self.val_discriminator_loss_meter.mloss
        cur_adv_loss = self.val_adversarial_loss_meter.mloss

        self.writer.add_scalar('Val/Acc', pixAcc, self.epoch)
        self.writer.add_scalar('Val/mIoU', mIoU, self.epoch)
        self.writer.add_scalar('Val_Generator /loss', self.val_generato_loss_meter.mloss, self.epoch)
        self.writer.add_scalar('Val_adversarial/loss', self.val_adversarial_loss_meter.mloss, self.epoch)
        self.writer.add_scalar('Val_Pixel/loss', self.val_pixel_loss.mloss, self.epoch)

        # for early-stopping
        if self.best_generator_loss > cur_gen_loss or self.best_pix_loss < cur_pix_loss:
            self.patience = 0
        else:
            self.patience += 1

        if self.best_generator_loss > cur_gen_loss:
            self.save_best = True

        # Store best score
        self.best_pixAcc = pixAcc if self.best_pixAcc < pixAcc else self.best_pixAcc
        self.best_generator_loss = cur_gen_loss if self.best_generator_loss > cur_gen_loss else self.best_generator_loss
        self.best_discriminator_loss = cur_dis_loss if self.best_discriminator_loss > cur_gen_loss else self.best_discriminator_loss
        self.best_pix_loss = cur_pix_loss if self.best_pix_loss > cur_pix_loss else self.best_pix_loss
        self.best_adv_loss = cur_adv_loss if self.best_adv_loss > cur_adv_loss else self.best_adv_loss
        self.best_mIoU = mIoU if self.best_mIoU < mIoU else self.best_mIoU

def main():
    print("call main")
    train_network = Network()
    train_network.run()

if __name__ == '__main__':
    main()
