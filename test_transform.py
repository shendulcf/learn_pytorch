from torch import tensor_split
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

# python的用法 -> tensor数据类型
# 通过ToTensor来看两个问题

img_path = r"xiaotudui\dataset\dataset1\train\ants\6743948_2b8c096dda.jpg"
img_path_abs = r"D:\workspace\Study_Pytorch\xiaotudui\dataset\dataset1\train\ants\6743948_2b8c096dda.jpg"

# 1、 transforms该如何使用
img = Image.open(img_path)
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# print(tensor_img)

# 2、 为什么我们需要Tensor数据类型

writer = SummaryWriter("logs")
writer.add_image("Tensor_img",tensor_img)