import os
import sys
import yaml
import time
import argparse

import time

from tqdm import tqdm
import torch.backends.cudnn as cudnn

sys.path.append('..')
from data import create_dataset
from util.utils import *
from models.load import get_segmentation_model
from util.metrics import *
from PIL import Image
from torch.autograd import Variable
from models.discriminator import *
import models.geno_searched as geno_types
from options.test_options import TestOptions

c = 0
pixel_acc, miou, total_time = 0,0,0
test_dir = " "# create a folder in the given path
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print('ERROR: Directory Exist', str(e))
        logging.info('ERROR: Directory Exist. ' + directory)



class TestNetwork(object):
    def __init__(self):
        self._init_configure()
        self._init_logger()
        self._init_device()
        self._init_dataset()
        self._init_model()
        if not self._check_resume():
            self.logger.error('The pre-trained model not exist!!!')
            exit(-1)

    def _init_configure(self):
        self.args = TestOptions().parse()
        with open(self.args.config) as fp:
            self.cfg = yaml.load(fp)
            print('load configure file at {}'.format(self.args.config))
        self.model_name = self.args.model
        print('Usage model :{}'.format(self.model_name))

    def _init_logger(self):
        global test_dir
        log_dir = './app/logs/'+ self.model_name + '/test' + '/{}'.format(self.cfg['data']['dataset']) \
                  +'/{}'.format(time.strftime('%Y%m%d-%H%M'))
        self.logger = get_logger(log_dir)
        print('RUNDIR: {}'.format(log_dir))
        #createFolder(log_dir)
        test_dir = log_dir

        print("RRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR", test_dir)
        self.logger.info('{}-Train'.format(self.model_name))
        self.save_path = log_dir
        self.save_image_path = os.path.join(self.save_path, 'saved_test_images')

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
        self.test_queue = create_dataset(self.args)

    def _init_model(self):

        # Setup loss function
        criterion_pixelwise = torch.nn.L1Loss()

        self.criterion_pixelwise = criterion_pixelwise.to(self.device)

        self.logger.info("Using criterion_pixelwise loss {}".format(self.criterion_pixelwise))

        # Setup Model
        try:
            genotype = eval("geno_types.%s" % self.cfg['training']['geno_type'])
            init_channels = self.cfg['training']['init_channels']
            depth = self.cfg['training']['depth']

        except:
            genotype = None
            init_channels = 0
            depth = 0

        self.aux = False
        model = get_segmentation_model(self.model_name,
                                           dataset=self.cfg['data']['dataset'],
                                           backbone=self.cfg['training']['backbone'],
                                           aux=self.aux,
                                           c=init_channels,
                                           depth=depth,
                                           # the below two are special for nasunet
                                           genotype=genotype,
                                           double_down_channel=self.cfg['training']['double_down_channel']
                                           )

        if torch.cuda.device_count() > 1 and self.cfg['training']['multi_gpus']:
            self.logger.info('use: %d gpus', torch.cuda.device_count())
            model = nn.DataParallel(model)
        else:
            self.logger.info('gpu device = %d' % self.device_id)
            torch.cuda.set_device(self.device_id)
        self.model = model.to(self.device)
        self.logger.info('param size = %fMB', calc_parameters_count(model))

    def _check_resume(self):
        resume = self.cfg['training']['resume'] if self.cfg['training']['resume'] is not None else None
        if resume is not None:
            if os.path.isfile(resume):
                self.logger.info("Loading model and optimizer from checkpoint '{}'".format(resume))
                checkpoint = torch.load(resume, map_location=self.device)
                self.model.load_state_dict(checkpoint['generator_model_state'])
                return True
            else:
                self.logger.info("No checkpoint found at '{}'".format(resume))
                return False
        return False

    def sample_images(self, real_A, real_B, fake_B, step, phase):

        """Saves a generated sample from the validation set"""

        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)
        # random_image = np.random.randint(15) # batch size of val data
        image_numpy = tensor2im(img_sample)
        img_path = self.save_image_path + "/%s_phase_%s_step.png" % (phase, step)
        save_image(image_numpy, img_path)

    def test(self, img_queue, split='val'):
        global total_time
        global pixel_acc, miou
        self.model.eval()
        report_freq =1
        tbar = tqdm(img_queue)
        create_exp_dir(self.save_image_path, desc='=>Save prediction image on')
        with torch.no_grad():
            for step, data in enumerate(tbar):

                AtoB = self.args.direction == 'AtoB'
                self.real_A = data['A' if AtoB else 'B'].to(self.device)
                self.real_B = data['B' if AtoB else 'A'].to(self.device)


                start_time = time.time()
                fake_B = self.model(self.real_A)
                time_taken = time.time() - start_time
                total_time = total_time + time_taken

                # save image
                if step % report_freq == 0:
                    self.sample_images(self.real_A, self.real_B, fake_B, step, 'test')

                if not isinstance(self.real_B, list):

                    loss_pixel = self.criterion_pixelwise(fake_B, self.real_B)
                    self.loss_meter.update(loss_pixel.item())
                    self.metric.update(self.real_B, fake_B)

                    pixAcc, mIoU = self.metric.get()
                    self.logger.info('{} loss: {}, pixAcc: {}, mIoU: {}'.format(
                        split, self.loss_meter.mloss, pixAcc, mIoU))

                    tbar.set_description('loss: %.6f, pixAcc: %.3f, mIoU: %.6f'
                                         % (self.loss_meter.mloss, pixAcc, mIoU))

                    pixel_acc= pixAcc
                    miou = mIoU
            
    def run(self):
        self.logger.info('args = %s', self.cfg)
        # Setup Metrics
        self.metric = SegmentationMetric(19)
        self.loss_meter = average_meter()
        run_start = time.time()
        # Set up results folder
        if not os.path.exists(self.save_image_path):
            os.makedirs(self.save_image_path)

        if len(self.test_queue) != 0:

            self.logger.info('Begin test set evaluation')
            self.test(self.test_queue, split='test')

        self.logger.info('Evaluation done!')


def main():
    print("call main")
    testNetwork = TestNetwork()
    testNetwork.run()

if __name__ == '__main__':
    main()






