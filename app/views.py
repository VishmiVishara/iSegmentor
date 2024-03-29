from alert_service import Alerter
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
from app.experiment.n_train import Network
import schedule
import time
from pathlib import Path
import json
from genotype import Genotype
import plotly.offline as opy
import plotly.graph_objs as go
from experiment import n_train
from experiment import test
from NasUnet.experiment import search_cell
import threading
import sys
import shutil
import zipfile
from datetime import datetime
import pytz
tz_NY = pytz.timezone('Asia/Kolkata')
datetime_NY = datetime.now(tz_NY)


BASE_DIR = Path(__file__).parent
CORE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEDIA_ROOT = os.path.join(CORE_DIR, 'media')
CONFIG_ROOT = os.path.join(BASE_DIR, 'configs')
LOGS_ROOT = os.path.join(BASE_DIR, r'logs/')

setting_obj = ""
dataset_list = ["CityScapes"]
original_dataset_path = MEDIA_ROOT+"/datasets"+"/{}/original_data"
gt_dataset_path = MEDIA_ROOT+"/datasets"+"/{}/gt_data"
ac = 0
x_value = []
y_value = []

x_value_dis = []
y_value_dis = []

media_folder = MEDIA_ROOT
config_path = CONFIG_ROOT

test_dir = " "
zipname = " "
epoch, train_discriminator_loss_meter, train_generator_loss_meter, train_pixel_loss, train_adversarial_loss_meter, pixAcc = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
test_pic_acc, test_mIoU = 0, 0

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
    context["dataset_list"] = dataset_list
    if request.method == 'POST':
        # get dataset name to create a folder to save data
        if 'save-dataset' in request.POST:
            print("Calling save dataset............")
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

        # get dataset name to create a folder to save data
    if 'config_save' in request.GET:
        print("CONFIG SAVE.............")
        print(request.GET)

        datasetName = request.GET.get('dataset')
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
        #save_config['data']['split'] = float(split)

        save_config['searching']['epoch'] = int(s_epoch)
        save_config['searching']['batch_size'] = int(s_batch_size)
        save_config['searching']['train_portion'] = float(train_portion)
        save_config['searching']['arch_optimizer']['name'] = arch_optimizer
        save_config['searching']['loss']['name'] = s_loss

        save_config['training']['geno_type'] = geno_type
        save_config['training']['epoch'] = int(t_epoch)
        save_config['training']['batch_size'] = int(t_batch_size)
        save_config['training']['val_batch_size'] = int(val_batch_size)
        save_config['training']['model_optimizer']['name'] = model_optimizer
        save_config['training']['loss']['name'] = t_loss

        filename = config_path + "/" + datasetName + ".yml"
        with open(filename, 'w') as f:
            yaml.dump(save_config, f, allow_unicode=True)

    if request.GET.get('tabs-icons-text-2-tab', True):
        print("Calling config tab")
        # read default config file
        with open(config_path + r"/cityscapes.yml", 'r') as yaml_in, open("cityscpaes.json", "w") as json_out:
            # print(json_out)
            yaml_object = yaml.safe_load(yaml_in)
            data = json.dump(yaml_object, json_out)

        # create a json using default config
        with open('cityscpaes.json') as f:
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

    if request.method == 'POST':
        if 'btn-search-init' in request.POST:
            sys.argv = ["hello"]
            search_cell.main()

    return HttpResponse(html_template.render(context, request))


def train(request):

    print("Load train")
    t = threading.Thread(target=launchTensorBoard, args=([]))
    t.start()
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

        for obj in data["NAS_UNET_NEW_V3"]["down"]:
            for key, value in obj.items():
                list_down_tuples.append((key, value))

        for obj in data["NAS_UNET_NEW_V3"]["up"]:
            for key, value in obj.items():
                list_up_tuples.append((key, value))

        down_range = range(data["NAS_UNET_NEW_V3"]['down_concat']
                           [0], data["NAS_UNET_NEW_V3"]['down_concat'][1])
        up_range = range(data["NAS_UNET_NEW_V3"]['up_concat']
                         [0], data["NAS_UNET_NEW_V3"]['up_concat'][1])

        # print(down_range)
        # print(up_range)

        # print(list_down_tuples)
        # print(list_up_tuples)

        geno = Genotype(down=list_down_tuples, down_concat=down_range,
                        up=list_up_tuples, up_concat=up_range)

        # print(geno)
        print(request)

        if request.method == 'POST':
            # if 'btn-tensorboard' in request.POST:
            #     t = threading.Thread(target=launchTensorBoard, args=([]))
            #     t.start()

            print(request)

            if request.POST.get('btn-train-init', True):
                print("training starting.......")
                # alerter = Alerter()
                # alerter.send_emails("A New Model Training initiated on Cityscapes Dataset" +
                # "using Searched U-Net Architecture - \n\n" +  str(geno)
                # + "\n\n It will take few hours to complete." +
                # "We'll Update you once we are done with the Training!")
                sys.argv = ["hello"]
                n_train.isStop = False
                n_train.main()
                # # open tensorboard in another thread
                # t = threading.Thread(target=launchTensorBoard, args=([]))
                # t.start()

        # if request.GET.get('btn-train_stop', True):
        if request.method == 'GET':
            if 'btn-train_stop' in request.GET:
                print("training stop.......")
                n_train.isStop = True
                print(n_train.isStop)
        
        # print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
        # print(request)
        # if(request.GET.get('btn-train_stop')):
        #     print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$Button clicked')

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

    #print("BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB", train_discriminator_loss_meter)

    # print(x_value)
    # print(y_value)
    if train_generator_loss_meter > 0.0:
        # print("Trueeeeeeeeeeeeeeeeeeeeeeeeeee")
        x_value.append(epoch)
        y_value.append(train_generator_loss_meter)

        x = x_value
        y = y_value
        trace1 = go.Scatter(x=x, y=y, marker={'color': 'red', 'symbol': 104, 'size': 10},
                            mode="lines",  name='1st Trace')

        # print(x_value)
        # print(y_value)

        data = go.Data([trace1])
        layout = go.Layout(title="Epoch vs Generator Loss", xaxis={
            'title': 'epoch'}, yaxis={'title': 'generator Loss'})
        figure = go.Figure(data=data, layout=layout)
        div = opy.plot(figure, auto_open=False, output_type='div')
        context['graph'] = div

    html_template = loader.get_template('live-chart.html')
    return HttpResponse(html_template.render(context, request))


def loadChartDis(request):

    print("Load chart Discriminator")
    context = {}
    global x_value_dis, y_value_dis
    global epoch, train_discriminator_loss_meter, train_generator_loss_meter, train_pixel_loss, train_adversarial_loss_meter, pixAcc

    epoch = n_train.epoch
    train_discriminator_loss_meter = n_train.train_discriminator_loss_meter
    train_generator_loss_meter = n_train.train_generator_loss_meter
    train_pixel_loss = n_train.train_pixel_loss
    train_adversarial_loss_meter = n_train.train_adversarial_loss_meter
    pixAcc = n_train.pixAcc_
    mIoU = n_train.mIoU_

    print("train_discriminator_loss_meter ",
          n_train.train_discriminator_loss_meter)
    if train_discriminator_loss_meter > 0:
        x_value_dis.append(epoch)
        y_value_dis.append(train_discriminator_loss_meter)

        x = x_value_dis
        y = y_value_dis
        trace1 = go.Scatter(x=x, y=y, marker={'color': 'red', 'symbol': 104, 'size': 10},
                            mode="lines",  name='1st Trace')

        # print(x_value_dis)
        # print(y_value_dis)

        data = go.Data([trace1])
        layout = go.Layout(title="Epoch vs Discriminator Loss", xaxis={
            'title': 'epoch'}, yaxis={'title': 'discriminator loss'})
        figure = go.Figure(data=data, layout=layout)
        div = opy.plot(figure, auto_open=False, output_type='div')
        context['graph_dis'] = div

    html_template = loader.get_template('live-chart-dis.html')
    return HttpResponse(html_template.render(context, request))


def evaluate(request):
    global test_dir
    global zipname
    html_template = loader.get_template('evaluate.html')
    context = {}
    context["dataset_list"] = dataset_list

    if request.method == 'POST':
        if 'btn-evaluat-init' in request.POST:
            sys.argv = ["hello"]
            test.main()

            test_dir = test.test_dir

            print(test_dir)

            test_mIoU = test.miou
            test_pic_acc = test.pixel_acc
            time = test.total_time / 500

            context["test_mIoU"] = str(round(test_mIoU * 100, 2))
            context["test_pic_acc"] = str(round(test_pic_acc * 100, 2))
            context["time"] = str(round(time, 2))

            print(test_mIoU)
            print(test_pic_acc)
            print(time)

    # if 'btn-download' in request.GET:

    return HttpResponse(html_template.render(context, request))


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file),
                       os.path.relpath(os.path.join(root, file),
                                       os.path.join(path, '..')))


def download(request):

    model_path = test_dir

    if model_path is not None:
        zip_file_name = test_dir+"_results_"
        #zip_file_path = os.path.join(folder_path_train + '/' + zip_file_name)

        print(zip_file_name)
        print(model_path)

        print("Download Results")

        zipname = zip_file_name + ".zip"
        zipf = zipfile.ZipFile(zipname, 'w', zipfile.ZIP_DEFLATED)
        zipdir(model_path, zipf)
        zipf.close()
        #shutil.make_archive(zip_file_name, 'zip', model_path)
        # shutil.rmtree(model_path)
        #context["test_download"] = zip_file_name

    if zipname != ' ':
        with open(zipname, 'rb') as f:
            contents = f.read()
        # Set the return value of the HttpResponse
        response = HttpResponse(contents, content_type="application\zip")
        # Set the HTTP header for sending to browser
        filename = 'results' + str(datetime.now()) + ".zip"
        response['Content-Disposition'] = "attachment; filename=%s" % filename
        # Return the response value
        return response


def launchTensorBoard():
    print(LOGS_ROOT)
    os.system('tensorboard --logdir ./app/logs/')
    return
