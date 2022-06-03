from numpy import pad
import torch
import torch.nn as nn
import torch.nn.functional as F

''' 要重写__init__和
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5) ;;

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
'''

class FanNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # self.conv1 = nn.Conv2d()
    
    def forward(self, x):
        x = torch.sigmoid(x)
        return x

fanfan = FanNet()
input = torch.tensor((3,2))

out = fanfan(input)
# print(out)

# ----> convolution layers

# nn.functional.conv2d()
input = torch.randint(0,5,(5,5))

kernel = torch.tensor([[1,2,1],
                      [0,1,0],
                      [2,1,0]])
input = torch.reshape(input,(1,1,5,5))                     
kernel = torch.reshape(kernel,(1,1,3,3))

# print(input)
# print(input.shape)
# print(kernel.shape)

output = F.conv2d(input, kernel, stride=1)
output2 = F.conv2d(input, kernel, stride=2, padding=0)
output3 = F.conv2d(input, kernel, stride=2, padding=1)

# print(output)
# print(output2)
# print(output3)
# print(output.shape)
# print(output2.shape)
# print(output3.shape)

## ----> nn.Conv2d
'''
in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t = 1, padding: _size_2_t | str = 0, 
dilation: _size_2_t = 1, groups: int = 1, bias: bool = True, padding_mode: str = 'zeros', device: Any | None = None,
dtype: Any | None = None) -> None
其中dilation定义了卷积核之间的距离, 即是空洞卷积
'''
nn.Conv2d()

