import torch
from torchvision import models
def Deeplabv3():

    deeplabv3plus_mobilenet = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    return models
