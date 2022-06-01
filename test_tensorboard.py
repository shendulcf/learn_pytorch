from torch.utils.tensorboard import SummaryWriter
import cv2 # opencv 图像就是numpy的类型
from PIL import Image
import numpy as np
from dataset import read_data

writer = SummaryWriter("logs")
# 
'''Args:
    tag (string): Data identifier
    img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
    global_step (int): Global step value to record
    walltime (float): Optional override default walltime (time.time())
      seconds after epoch of event
'''
img_path = r'D:\workspace\Study_Pytorch\xiaotudui\dataset\dataset1\train\ants\649026570_e58656104b.jpg'
img = Image.open(img_path)
img_array = np.array(img)
print(type(img_array))
print(img_array.shape)
writer.add_image("img", img_array, 1, dataformats='HWC')


# y = 2x
'''
Args:
    tag (string): Data identifier
    scalar_value (float or string/blobname): Value to save
    global_step (int): Global step value to record
    walltime (float): Optional override default walltime (time.time())
      with seconds after epoch of event
    new_style (boolean): Whether to use new style (tensor field) or old
      style (simple_value field). New style could lead to faster data loading.
'''
# for i in range(100):
#     writer.add_scalar("y=x", i, i)
'''
tensorboard --logdir=logs --port=6007
''' 

writer.close()