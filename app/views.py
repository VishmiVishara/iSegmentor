import json
import os
import yaml
import sys
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template
import schedule
import time
from pathlib import Path
import json
from pathlib import Path
from genotype import Genotype
import plotly.offline as opy
import plotly.graph_objs as go
from experiment import n_train
import sys
# sys.path.append('..')
# from options.train_options import TrainOptions
#import train

BASE_DIR = Path(__file__).parent
CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEDIA_ROOT = os.path.join(CORE_DIR, 'media')
CONFIG_ROOT = os.path.join(BASE_DIR, 'configs')

setting_obj = ""
dataset_list = ["CityScapes", "PASCAL VOC 2012"]
original_dataset_path = MEDIA_ROOT+"/{}/original_data"
gt_dataset_path = MEDIA_ROOT+"/{}/gt_data"
ac = 0
x_value = []
y_value = []

media_folder = MEDIA_ROOT
config_path = CONFIG_ROOT

epoch, train_discriminator_loss_meter, train_generator_loss_meter, train_pixel_loss, train_adversarial_loss_meter, pixAcc = 0, 0, 0, 0, 0, 0


# create a folder in the given path
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print('ERROR: Directory Exist', str(e))
        logging.info('ERROR: Directory Exist. ' + directory)


def index(request):

    context = {}
    global setting_obj

    if request.method == 'POST':
        # get dataset name to create a folder to save data
        if 'save-dataset' in request.POST:
            datasetName = request.POST.get('datasetname')
            print(datasetName)

            fs = FileSystemStorage()

            original_data_path = os.path.join(
                original_dataset_path.format(datasetName))
            # create a folder for original images
            createFolder(original_data_path)

            gt_data_path = os.path.join(gt_dataset_path.format(datasetName))
            # create a folder for ground thruth images
            createFolder(gt_data_path)

            originalImage = request.FILES['originalImage']
            gtImage = request.FILES['GTImage']

            # save original image dataset file
            originalImage_filename = fs.save(
                original_data_path + r"/" + originalImage.name, originalImage)
            original_uploaded_file_url = fs.url(originalImage_filename)

            # save GT umage dataset file
            GTImage_filename = fs.save(
                gt_data_path + r"/" + gtImage.name, gtImage)
            gt_uploaded_file_url = fs.url(GTImage_filename)

            dataset_list.append(datasetName)
            print(dataset_list)

    if request.method == 'GET':
        # get dataset name to create a folder to save data
        if 'config_save' in request.GET:
            datasetName = request.GET.get('dataset')
            print(datasetName)
            split = request.GET.get('split')
            s_epoch = request.GET.get('s_epoch')
            s_batch_size = request.GET.get('s_batch_size')
            train_portion = request.GET.get('train_portion')
            arch_optimizer = request.GET.get('arch_optimizer')
            s_loss = request.GET.get('s_loss')

            geno_type = request.GET.get('geno_type')
            t_epoch = request.GET.get('t_epoch')
            t_batch_size = request.GET.get('t_batch_size')
            val_batch_size = request.GET.get('val_batch_size')
            model_optimizer = request.GET.get('model_optimizer')
            t_loss = request.GET.get('t_loss')

            save_config = setting_obj
            save_config['data']['dataset'] = datasetName
            save_config['data']['split'] = split

            save_config['searching']['epoch'] = s_epoch
            save_config['searching']['batch_size'] = s_batch_size
            save_config['searching']['train_portion'] = train_portion
            save_config['searching']['arch_optimizer'] = arch_optimizer
            save_config['searching']['loss'] = s_loss

            save_config['training']['geno_type'] = geno_type
            save_config['training']['epoch'] = t_epoch
            save_config['training']['batch_size'] = t_batch_size
            save_config['training']['val_batch_size'] = val_batch_size
            save_config['training']['model_optimizer'] = model_optimizer
            save_config['training']['loss'] = t_loss

            filename = config_path + "/" + datasetName + ".yml"
            with open(filename, 'w') as f:
                yaml.dump(save_config, f, allow_unicode=True)

        elif request.GET.get('tabs-icons-text-2-tab', True):
            # read default config file
            with open(config_path + r"/nas_unet_voc.yml", 'r') as yaml_in, open("voc.json", "w") as json_out:
                # print(json_out)
                yaml_object = yaml.safe_load(yaml_in)
                data = json.dump(yaml_object, json_out)

            # create a json using default config
            with open('voc.json') as f:
                data = json.load(f)
                # print(data["model"])
                setting_obj = data
                # save default config to the context
                context['config'] = data

    return render(request, 'index.html', context)


def search(request):
    html_template = loader.get_template('search.html')
    context = {}
    context["dataset_list"] = dataset_list
    return HttpResponse(html_template.render(context, request))


def train(request):
    print("Load train")
    context = {}
    context["dataset_list"] = dataset_list

    with open("geno.json") as json_out:
        data = json.load(json_out)
        architecture_list = []
        architecture_list = data.keys()
        list(architecture_list)

        context["architecture_list"] = architecture_list

        list_down_tuples = []
        list_up_tuples = []
        down_range = []
        up_range = []

        for obj in data["NAS_UNET_V2_En"]["down"]:
            for key, value in obj.items():
                list_down_tuples.append((key, value))

        for obj in data["NAS_UNET_V2_En"]["up"]:
            for key, value in obj.items():
                list_up_tuples.append((key, value))

        down_range = range(data["NAS_UNET_V2_En"]['down_concat']
                           [0], data["NAS_UNET_V2_En"]['down_concat'][1])
        up_range = range(data["NAS_UNET_V2_En"]['up_concat']
                         [0], data["NAS_UNET_V2_En"]['up_concat'][1])

        # print(down_range)
        # print(up_range)

        # print(list_down_tuples)
        # print(list_up_tuples)

        geno = Genotype(down=list_down_tuples, down_concat=down_range,
                        up=list_up_tuples, up_concat=up_range)

        # print(geno)

        if request.method == 'POST':
            if request.POST.get('btn-train-init', True):
                sys.argv = ["hello"]
                n_train.main()

            else:
                print(request.POST)
        return render(request, 'train.html', context)


def loadChart(request):

    context = {}
    global x_value, y_value
    global epoch, train_discriminator_loss_meter, train_generator_loss_meter, train_pixel_loss, train_adversarial_loss_meter, pixAcc

    epoch = n_train.epoch
    train_discriminator_loss_meter = n_train.train_discriminator_loss_meter
    train_generator_loss_meter = n_train.train_generator_loss_meter
    train_pixel_loss = n_train.train_pixel_loss
    train_adversarial_loss_meter = n_train.train_adversarial_loss_meter
    pixAcc = n_train.pixAcc_
    mIoU = n_train.mIoU_

    print(x_value)
    print(y_value)
    if train_generator_loss_meter > 0.0 :
        x_value.append(epoch)
        y_value.append(train_generator_loss_meter)

        x = x_value
        y = y_value
        trace1 = go.Scatter(x=x, y=y, marker={'color': 'red', 'symbol': 104, 'size': 10},
                            mode="lines",  name='1st Trace')

        print(x_value)
        print(y_value)

        data = go.Data([trace1])
        layout = go.Layout(title="Epoch vs Generator Loss", xaxis={
                        'title': 'x1'}, yaxis={'title': 'x2'})
        figure = go.Figure(data=data, layout=layout)
        div = opy.plot(figure, auto_open=False, output_type='div')
        context['graph'] = div

    html_template = loader.get_template('live-chart.html')
    return HttpResponse(html_template.render(context, request))


# class reciver:
#     global epoch_, train_discriminator_loss_meter_,train_generator_loss_meter_, train_pixel_loss_, train_adversarial_loss_meter_, pixAcc_
#     epoch_, train_discriminator_loss_meter_,train_generator_loss_meter_, train_pixel_loss_, train_adversarial_loss_meter_, pixAcc_ = 0,0,0,0,0,0

#     @staticmethod
#     def call(epoch, train_discriminator_loss_meter,
#                         train_generator_loss_meter,
#                         train_pixel_loss,
#                         train_adversarial_loss_meter,
#                         pixAcc,
#                         mIoU):
#         epoch_ = epoch
#         train_discriminator_loss_meter_ = train_discriminator_loss_meter
#         train_generator_loss_meter_ = train_generator_loss_meter
#         train_pixel_loss_  = train_pixel_loss
#         train_adversarial_loss_meter_ = train_adversarial_loss_meter
#         pixAcc_ = pixAcc

#         print("#################", epoch, train_discriminator_loss_meter,train_generator_loss_meter, train_pixel_loss, train_adversarial_loss_meter, pixAcc)

#     @staticmethod
#     def values():
#         print("******************",  epoch_)
