import json
import os
import yaml
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render, get_object_or_404, redirect
from django.template import loader
from django.http import HttpResponse
from django import template
import core.settings as sett
import json

setting_obj = ""
original_dataset_path = sett.MEDIA_ROOT+"/{}/original_data"
gt_dataset_path = sett.MEDIA_ROOT+"/{}/gt_data"

config_path = sett.CONFIG_ROOT

# create a folder in the given path


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError as e:
        print('ERROR: Directory Exist', str(e))
        logging.info('ERROR: Directory Exist. ' + directory)


def pages(request):
    context = {}
    # All resource paths end in .html.
    # Pick out the html file name from the url. And load that template.
    try:

        load_template = request.path.split('/')[-1]
        context['segment'] = load_template

        html_template = loader.get_template(load_template)
        return HttpResponse(html_template.render(context, request))

    except template.TemplateDoesNotExist:

        html_template = loader.get_template('page-404.html')
        return HttpResponse(html_template.render(context, request))

    except:

        html_template = loader.get_template('page-500.html')
        return HttpResponse(html_template.render(context, request))


def index(request):
    context = {}
    global setting_obj
    
    print(request.method)

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

            filename = config_path + "/"+ datasetName + ".yml"
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
                #print(data["model"])
                setting_obj = data
                # save default config to the context
                context['config'] = data
                print(context['config'])

    return render(request, 'index.html', context)
