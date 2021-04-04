import sys
sys.path.append('..')
from .nas_unet import *

def get_segmentation_model(name, **kwargs):
    models = {
        'nasunet': get_nas_unet,
        'cityscapes': get_nas_unet,
    }
    return models[name.lower()](**kwargs)
