import torch

import torchvision
import torchvision.models as models
import torch.nn as nn

vgg16_true = models.vgg16(pretrained=True)
vgg16_true

vgg16_true.add_module('add_linear', nn.Linear(1000,10))
vgg16_true.classifier[5] = nn.Linear(4096,10)
print(vgg16_true)


